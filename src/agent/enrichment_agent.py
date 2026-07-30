"""EnrichmentAgent: Rescores matches using pgvector resume RAG.

The LLM metadata extraction (company_description, role_summary, location, salary)
now lives exclusively in node_matcher. This agent ONLY handles vector rescoring.
"""  # noqa: E501

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from src.configuration import get_config
from src.logging import get_logger

if TYPE_CHECKING:
    from src.memory.pgvector_store import MemoryStore

logger = get_logger("enrichment_agent")


async def _get_embedding(text: str) -> list[float] | None:
    """Fetch text embedding from local llama-server. Returns None on failure."""
    cfg = get_config().embed
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.post(
                f"{cfg.url}/embeddings",
                json={"input": text[:2000]},
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("data", [{}])[0].get("embedding", [])
                if isinstance(emb, list) and len(emb) > 0:
                    return [float(v) for v in emb]
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

        jd_vector = await _get_embedding(jd_text)
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
                job["shortlist_probability"] = int(job["match_percent"] * 0.85)

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
