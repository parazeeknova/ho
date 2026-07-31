"""DorkingEngine (Pillar 2): Deep company and ATS job discovery via SearXNG search dorking."""

from __future__ import annotations

import asyncio

import httpx

from src.logging import get_logger
from src.radar.models import JobObservation

logger = get_logger("dorking_engine")

_DORK_QUERIES = [
    'site:boards.greenhouse.io "Engineer" OR "Developer" OR "Infrastructure"',
    'site:ashbyhq.com "Engineer" OR "Staff" OR "Founding"',
    'site:jobs.lever.co "Engineer" OR "Backend" OR "AI"',
    'site:apply.workable.com "Engineer" OR "Senior"',
]


class DorkingEngine:
    """Queries SearXNG with specialized search engine dorks to uncover
    freshly indexed ATS job postings across the web.
    """

    def __init__(self, searxng_url: str = "http://localhost:8080") -> None:
        self.searxng_url = searxng_url.rstrip("/")
        self._seen_urls: set[str] = set()

    async def execute_dorks(self, queries: list[str] | None = None) -> list[JobObservation]:
        """Runs targeted dork queries against SearXNG and returns job observations."""
        target_queries = queries or _DORK_QUERIES
        observations: list[JobObservation] = []

        async with httpx.AsyncClient(timeout=8.0) as client:
            for q in target_queries:
                try:
                    resp = await client.get(
                        f"{self.searxng_url}/search",
                        params={"q": q, "format": "json", "language": "en"},
                    )
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    results = data.get("results", [])
                    for r in results:
                        link = r.get("url", "")
                        title = r.get("title", "")
                        content = r.get("content", "")

                        if not link or link in self._seen_urls:
                            continue

                        self._seen_urls.add(link)
                        comp_guess = self._extract_company_from_url(link)
                        observations.append(
                            JobObservation(
                                url=link,
                                source=f"dork-{comp_guess}",
                                title=title or "Software Engineer",
                                snippet=content,
                            )
                        )
                except Exception as e:
                    logger.debug(f"SearXNG dork query '{q}' failed: {e}")

                await asyncio.sleep(0.5)

        logger.info(f"Dorking engine discovered {len(observations)} job postings from SearXNG")
        return observations

    @staticmethod
    def _extract_company_from_url(url: str) -> str:
        low = url.lower()
        if "boards.greenhouse.io/" in low:
            parts = low.split("boards.greenhouse.io/")[-1].split("/")
            return parts[0] if parts else "unknown"
        if "ashbyhq.com/" in low:
            parts = low.split("ashbyhq.com/")[-1].split("/")
            return parts[0] if parts else "unknown"
        if "jobs.lever.co/" in low:
            parts = low.split("jobs.lever.co/")[-1].split("/")
            return parts[0] if parts else "unknown"
        if "apply.workable.com/" in low:
            parts = low.split("apply.workable.com/")[-1].split("/")
            return parts[0] if parts else "unknown"
        return "searxng-discovered"
