"""StartupAgent: Researches startup founders, co-founders, social links,
funding rounds (Seed/Series A), and recent company news via SearXNG & Cloud LLM.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from src.llm.context import ContextManager


async def _searxng_search(query: str) -> list[str]:
    """Execute search query against local SearXNG."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(
                "http://localhost:8080/search",
                params={"q": query, "format": "json"},
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return [
                    f"{r.get('title', '')}: {r.get('content', '')} ({r.get('url', '')})"
                    for r in results[:4]
                    if r.get("content") or r.get("title")
                ]
    except Exception as e:
        print(f"  [dim]SearXNG query '{query}': {e}[/dim]")
    return []


class StartupAgent:
    """Agent that researches startup founders, funding, and outreach info."""

    def __init__(self, ctx: ContextManager) -> None:
        self.ctx = ctx

    async def analyze_startup(self, job: dict[str, Any]) -> dict[str, Any]:
        """Research company founders, funding stage, socials, and recent news."""
        company = str(job.get("company") or "").strip()
        if not company or company in ("N/A", "Unknown", "Company"):
            return job

        # Execute parallel SearXNG queries for founder & funding info
        queries = [
            f"{company} founders cofounders CEO LinkedIn Twitter",
            f"{company} seed series A funding valuation news",
        ]
        results_list = await asyncio.gather(*(_searxng_search(q) for q in queries))
        combined_snippets = "\n".join(snippet for sublist in results_list for snippet in sublist)

        if not combined_snippets:
            return job

        prompt = (
            f"Analyze web info for '{company}':\n\n{combined_snippets[:6000]}\n\n"
            "Extract:\n"
            "1. is_startup: boolean (true if startup/venture backed, false if enterprise).\n"
            "2. founders: list of founder/co-founder names and titles.\n"
            "3. funding_stage: e.g. 'Seed ($3.5M)', 'Series A', 'Bootstrapped', or 'N/A'.\n"
            "4. founder_socials: list of LinkedIn/X profiles for outreach.\n"
            "5. company_news: 1 sentence summary of recent funding or news.\n"
            "Return valid JSON matching these keys."
        )

        schema = {
            "type": "object",
            "properties": {
                "is_startup": {"type": "boolean"},
                "founders": {"type": "array", "items": {"type": "string"}},
                "funding_stage": {"type": "string"},
                "founder_socials": {"type": "array", "items": {"type": "string"}},
                "company_news": {"type": "string"},
            },
            "required": ["is_startup", "founders", "funding_stage"],
        }

        extracted = await self.ctx.json_chat(prompt, schema=schema)
        if isinstance(extracted, dict):
            job["is_startup"] = extracted.get("is_startup", False)
            if extracted.get("founders"):
                job["founders"] = extracted["founders"]
            if extracted.get("funding_stage") and extracted["funding_stage"] != "N/A":
                job["funding_stage"] = extracted["funding_stage"]
            if extracted.get("founder_socials"):
                job["founder_socials"] = extracted["founder_socials"]
            if extracted.get("company_news"):
                job["company_news"] = extracted["company_news"]

            # Prioritize startups by boosting match percent and shortlist score slightly
            if job["is_startup"]:
                job["match_percent"] = min(99, job.get("match_percent", 0) + 10)
                job["shortlist_probability"] = min(95, job.get("shortlist_probability", 0) + 10)

        return job

    async def batch_analyze_startups(
        self, jobs: list[dict[str, Any]], concurrency: int = 8
    ) -> list[dict[str, Any]]:
        """Parallel analysis of startup intelligence for candidate jobs."""
        if not jobs:
            return []

        sem = asyncio.Semaphore(concurrency)

        async def _worker(j: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                try:
                    return await self.analyze_startup(j)
                except Exception:
                    return j

        return await asyncio.gather(*(_worker(j) for j in jobs))
