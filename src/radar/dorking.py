"""DorkingEngine (Pillar 2): Deep company and ATS job discovery via SearXNG search dorking."""

from __future__ import annotations

import asyncio

import httpx

from src.logging import get_logger
from src.radar.models import JobObservation

logger = get_logger("dorking_engine")

_DORK_QUERIES = [
    (
        "site:boards.greenhouse.io OR site:jobs.lever.co OR site:jobs.ashbyhq.com OR"
        ' site:apply.workable.com intitle:"intern" OR intitle:"new grad" OR'
        ' intitle:"junior" "software" "2026"'
    ),
    'site:boards.greenhouse.io "Junior" OR "Entry Level" OR "Associate" OR "Graduate"',
    'site:jobs.ashbyhq.com "Junior" OR "Entry Level" OR "Early Career" OR "University"',
    'site:jobs.lever.co "Junior" OR "Entry Level" OR "Graduate" OR "Associate"',
    'site:apply.workable.com "Junior" OR "Entry Level" OR "Associate"',
    'site:boards.greenhouse.io "New Grad" OR "2026" OR "Intern" OR "Internship"',
    'site:jobs.ashbyhq.com "New Grad" OR "2026" OR "Intern" OR "Internship"',
    'site:jobs.lever.co "New Grad" OR "2026" OR "Intern" OR "Internship"',
    'site:apply.workable.com "New Grad" OR "2026" OR "Intern" OR "Internship"',
    # Targeted Software / Developer Junior Dorks
    'site:boards.greenhouse.io ("Junior Developer" OR "Associate Software Engineer")',
    'site:jobs.ashbyhq.com ("Junior Software Engineer" OR "Entry Level Engineer")',
    'site:jobs.lever.co ("Junior Software Engineer" OR "Associate Engineer")',
]


class DorkingEngine:
    """Queries SearXNG with specialized search engine dorks to uncover
    freshly indexed ATS job postings across the web.
    """

    def __init__(self, searxng_url: str = "http://localhost:8080") -> None:
        self.searxng_url = searxng_url.rstrip("/")
        self._seen_urls: set[str] = set()

    async def execute_dorks(
        self, queries: list[str] | None = None, time_range: str = "day"
    ) -> list[JobObservation]:
        """Runs targeted 48h time-restricted dork queries against SearXNG."""
        target_queries = queries or _DORK_QUERIES
        observations: list[JobObservation] = []

        async with httpx.AsyncClient(timeout=8.0) as client:
            for q in target_queries:
                try:
                    resp = await client.get(
                        f"{self.searxng_url}/search",
                        params={
                            "q": q,
                            "format": "json",
                            "time_range": time_range,
                            "language": "en",
                        },
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
