"""EnrichmentAgent: Rescores matches using pgvector resume RAG.

The LLM metadata extraction (company_description, role_summary, location, salary)
now lives exclusively in node_matcher. This agent ONLY handles vector rescoring.

Embeddings are cached by content hash in ``embed_cache`` so identical JD text
is never re-sent to the (shared) llama-server across sweeps.
"""  # noqa: E501

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from typing import TYPE_CHECKING, Any

from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger

if TYPE_CHECKING:
    from src.memory.pgvector_store import MemoryStore

logger = get_logger("enrichment_agent")

EMBED_TEXT_CAP = 2000


def _embed_text_hash(text: str) -> str:
    """Content hash for the exact text sent to the embed server."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _get_embedding(text: str, store: MemoryStore | None = None) -> list[float] | None:
    """Fetch text embedding from local llama-server. Returns None on failure.

    When *store* is given, embeddings are cached by content hash so identical
    text is fetched once and replayed from Postgres afterwards.
    """
    text = text[:EMBED_TEXT_CAP]
    if not text.strip():
        return None
    text_hash = _embed_text_hash(text)
    if store is not None:
        cached = await store.get_cached_embedding(text_hash)
        if cached is not None:
            return cached
    cfg = get_config().embed
    try:
        client = await get_client("enrichment_agent", timeout=cfg.timeout)
        resp = await client.post(
            f"{cfg.url}/embeddings",
            json={"input": text},
        )
        if resp.status_code == 200:
            data = resp.json()
            emb = data.get("data", [{}])[0].get("embedding", [])
            if isinstance(emb, list) and len(emb) > 0:
                embedding = [float(v) for v in emb]
                if store is not None:
                    with contextlib.suppress(Exception):
                        # cache is best-effort; never break embedding
                        await store.put_cached_embedding(text_hash, embedding)
                return embedding
    except Exception as e:
        logger.error(
            "Embedding fetch failed, returning None",
            exception=str(e),
        )
    return None


class EnrichmentAgent:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    async def enrich_and_rescore(self, job: dict[str, Any]) -> dict[str, Any]:
        """Rescore match against pgvector resume embeddings.

        Matcher already set company_description, role_summary, etc.
        This only does vector similarity rescoring.
        """
        role = str(job.get("role") or "Position")
        company = str(job.get("company") or "Company")

        jd_text = " ".join(
            str(job.get(f, ""))
            for f in ("jd_summary", "role_summary", "company_description")
            if job.get(f)
        )
        if not jd_text:
            jd_text = f"{role} {company}"

        jd_vector = await _get_embedding(jd_text, self.store)
        if jd_vector is None:
            return job

        resume_chunks = await self.store.search_similar_chunks(jd_vector, top_k=5)
        if resume_chunks:
            similarities = [1.0 - ch.get("distance", 1.0) for ch in resume_chunks]
            similarities = [s for s in similarities if s >= 0.0]

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                calculated_match = int(min(98, max(30, (avg_sim - 0.4) * 200)))
                job["match_percent"] = max(job.get("match_percent", 0), calculated_match)
                # shortlist_probability is now the calibrated P(screen/interview/offer)
                # from ml/calibration.py (separate binary classifiers). Do not
                # derive it as match_percent*0.85 — that fake is removed. Preserve
                # any LLM-set value in shadow mode; the calibrated value will
                # overwrite it when the model is live (Phase 4).
                job.setdefault("shortlist_probability", job["match_percent"])

        return job

    async def batch_enrich_and_rescore(
        self,
        jobs: list[dict[str, Any]],
        concurrency: int = 8,
    ) -> list[dict[str, Any]]:
        """Parallel rescoring of candidate jobs."""
        if not jobs:
            return []

        sem = asyncio.Semaphore(concurrency)

        async def _worker(j: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                try:
                    return await self.enrich_and_rescore(j)
                except Exception:
                    return j

        return await asyncio.gather(*(_worker(j) for j in jobs))
