"""EnrichmentAgent: Rescores matches using pgvector resume RAG.

The LLM metadata extraction (company_description, role_summary, location, salary)
now lives exclusively in node_matcher. This agent ONLY handles vector rescoring.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx

from src.memory.pgvector_store import MemoryStore

EMBED_URL = "http://127.0.0.1:8900/v1"


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


async def _get_embedding(text: str) -> list[float]:
    """Fetch text embedding from local llama-server on :8900."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(
                f"{EMBED_URL}/embeddings",
                json={"input": text[:2000]},
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("data", [{}])[0].get("embedding", [])
                if isinstance(emb, list) and len(emb) > 0:
                    return [float(v) for v in emb]
    except Exception:
        pass

    # Fallback deterministic pseudo-embedding (dim=1024)
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [(float(b) / 255.0) for b in h]
    while len(vec) < 1024:
        vec.extend(vec[: 1024 - len(vec)])
    return vec[:1024]


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

        resume_chunks = await self.store.search_similar_chunks(jd_vector, top_k=5)
        if resume_chunks:
            similarities = []
            for chunk in resume_chunks:
                emb = chunk.get("embedding")
                if isinstance(emb, list):
                    sim = _cosine_similarity(jd_vector, emb)
                    similarities.append(sim)

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                calculated_match = int(min(98, max(30, (avg_sim - 0.5) * 160)))
                job["match_percent"] = max(
                    job.get("match_percent", 0), calculated_match
                )
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
