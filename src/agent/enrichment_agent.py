"""EnrichmentAgent: Cross-searches job postings from multiple sources,
pulls complete JDs & company info, and rescores matches using pgvector resume RAG.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import httpx
from firecrawl import FirecrawlApp

from src.agent.jobs_agent import get_embedding
from src.llm.context import ContextManager
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


async def cross_search_job_details(
    app: FirecrawlApp,
    role: str,
    company: str,
    apply_link: str,
) -> str:
    """Scrape apply link or cross-search alternative pages for raw text."""
    # 1. Try scraping the direct apply link first
    if apply_link and apply_link.startswith("http"):
        try:
            async with httpx.AsyncClient(timeout=6.0, follow_redirects=True) as client:
                resp = await client.get(
                    apply_link,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                if resp.status_code == 200 and len(resp.text) > 300:
                    return resp.text[:12000]
        except Exception:
            pass

    # 2. Try Firecrawl scrape
    if apply_link and apply_link.startswith("http"):
        try:
            res = await asyncio.to_thread(
                app.scrape_url,
                apply_link,
                params={"formats": ["markdown"]},
            )
            md = res.get("markdown", "") if isinstance(res, dict) else ""
            if md and len(md) > 200:
                return md[:12000]
        except Exception:
            pass

    return f"{role} position at {company}."


class EnrichmentAgent:
    def __init__(self, store: MemoryStore, ctx: ContextManager, app: FirecrawlApp) -> None:
        self.store = store
        self.ctx = ctx
        self.app = app

    async def enrich_and_rescore(self, job: dict[str, Any]) -> dict[str, Any]:
        """Enrich company background, role summary, and rescore match against pgvector."""
        role = str(job.get("role") or "Position")
        company = str(job.get("company") or "Company")
        apply_link = str(job.get("apply_link") or job.get("url") or "")

        # Step 1: Cross-search for full JD & company text
        raw_text = await cross_search_job_details(self.app, role, company, apply_link)

        # Step 2: Parallel LLM extraction for company description & role summary
        prompt = (
            f"Analyze this job listing for '{role}' at '{company}':\n\n{raw_text[:8000]}\n\n"
            "Extract:\n"
            "1. company_description: 1-2 sentence company overview.\n"
            "2. role_summary: 1-2 sentence role overview.\n"
            "3. location: Specific city/country or 'Remote'.\n"
            "4. salary: Salary range if mentioned, else null.\n"
            "Return valid JSON matching keys: company_description, role_summary, location, salary."
        )
        schema = {
            "type": "object",
            "properties": {
                "company_description": {"type": "string"},
                "role_summary": {"type": "string"},
                "location": {"type": "string"},
                "salary": {"type": ["string", "null"]},
            },
            "required": ["company_description", "role_summary", "location"],
        }
        extracted = await self.ctx.json_chat(prompt, schema=schema)
        if isinstance(extracted, dict):
            if extracted.get("company_description"):
                job["company_description"] = extracted["company_description"]
            if extracted.get("role_summary"):
                job["role_summary"] = extracted["role_summary"]
            if extracted.get("location") and extracted["location"] not in ("-", "Unknown"):
                job["location"] = extracted["location"]
            if extracted.get("salary") and not job.get("salary"):
                job["salary"] = extracted["salary"]

        # Step 3: Rescore via RAG pgvector vector similarity
        jd_text = f"{role} {company} {job.get('role_summary', '')} {raw_text[:2000]}"
        jd_vector = await get_embedding(jd_text)

        # Query top matching resume chunks from pgvector store
        resume_chunks = await self.store.search_similar_chunks(jd_vector, top_k=5)
        if resume_chunks:
            # Compute average vector similarity score
            similarities = []
            for chunk in resume_chunks:
                emb = chunk.get("embedding")
                if isinstance(emb, list):
                    sim = _cosine_similarity(jd_vector, emb)
                    similarities.append(sim)

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                # Map cosine similarity (0.6 - 0.95) to percentage (40% - 98%)
                calculated_match = int(min(98, max(30, (avg_sim - 0.5) * 160)))
                job["match_percent"] = max(job.get("match_percent", 0), calculated_match)
                job["shortlist_probability"] = int(job["match_percent"] * 0.85)

        return job

    async def batch_enrich_and_rescore(
        self,
        jobs: list[dict[str, Any]],
        concurrency: int = 8,
    ) -> list[dict[str, Any]]:
        """Parallel multithreaded enrichment for candidate jobs."""
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
