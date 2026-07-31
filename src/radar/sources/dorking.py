"""DorkingEngine (Pillar 2): Deep company and ATS job discovery via SearXNG search dorking.

Queries are time-restricted: qdr:d2 (past 2 days) for Google/Bing syntax,
plus SearXNG's own time_range=day parameter. Both filters are applied so
results are strictly limited to the last 48 hours, preventing 6-month-old
ghost jobs from contaminating the pipeline.
"""

from __future__ import annotations

import asyncio

from src.http_client import get_client
from src.logging import get_logger
from src.radar.core.models import JobObservation

logger = get_logger("dorking_engine")

_TIME_SYNTAX = " qdr:d2"

_DORK_QUERIES = [
    (
        "site:boards.greenhouse.io OR site:jobs.lever.co OR site:jobs.ashbyhq.com OR"
        ' site:apply.workable.com intitle:"intern" OR intitle:"new grad" OR'
        f' intitle:"junior" "software" "2026"{_TIME_SYNTAX}'
    ),
    (
        'site:boards.greenhouse.io "Junior" OR "Entry Level" OR "Associate" OR'
        f' "Graduate"{_TIME_SYNTAX}'
    ),
    (
        'site:jobs.ashbyhq.com "Junior" OR "Entry Level" OR "Early Career" OR'
        f' "University"{_TIME_SYNTAX}'
    ),
    (f'site:jobs.lever.co "Junior" OR "Entry Level" OR "Graduate" OR "Associate"{_TIME_SYNTAX}'),
    f'site:apply.workable.com "Junior" OR "Entry Level" OR "Associate"{_TIME_SYNTAX}',
    f'site:boards.greenhouse.io "New Grad" OR "2026" OR "Intern" OR "Internship"{_TIME_SYNTAX}',
    f'site:jobs.ashbyhq.com "New Grad" OR "2026" OR "Intern" OR "Internship"{_TIME_SYNTAX}',
    f'site:jobs.lever.co "New Grad" OR "2026" OR "Intern" OR "Internship"{_TIME_SYNTAX}',
    f'site:apply.workable.com "New Grad" OR "2026" OR "Intern" OR "Internship"{_TIME_SYNTAX}',
    (
        'site:boards.greenhouse.io ("Junior Developer" OR'
        f' "Associate Software Engineer"){_TIME_SYNTAX}'
    ),
    (f'site:jobs.ashbyhq.com ("Junior Software Engineer" OR "Entry Level Engineer"){_TIME_SYNTAX}'),
    (f'site:jobs.lever.co ("Junior Software Engineer" OR "Associate Engineer"){_TIME_SYNTAX}'),
]


class DorkingEngine:
    """Queries SearXNG with specialized search engine dorks to uncover
    freshly indexed ATS job postings across the web.

    Each query carries both the SearXNG time_range=day parameter AND
    the qdr:d2 syntax understood by Google/Bing, ensuring dual-layered
    time filtering. Results are deduplicated across runs via _seen_urls.
    """

    def __init__(self, searxng_url: str = "http://localhost:8080") -> None:
        self.searxng_url = searxng_url.rstrip("/")
        self._seen_urls: set[str] = set()

    async def execute_dorks(
        self, queries: list[str] | None = None, time_range: str = "day"
    ) -> list[JobObservation]:
        """Runs 48h time-restricted dork queries against SearXNG.

        Two-layer time filter:
          1. SearXNG time_range=day (maps to d/w/m/y in the metasearch engine)
          2. qdr:d2 appended to each query string (Google/Bing-native filter)

        This double gating ensures even SearXNG engines that ignore time_range
        still return only recent results, and engines like Google that respect
        qdr:d2 add their own 48-hour cutoff.
        """
        target_queries = queries or _DORK_QUERIES
        observations: list[JobObservation] = []

        client = await get_client("dorking", timeout=12.0)
        for q in target_queries:
            try:
                resp = await client.get(
                    f"{self.searxng_url}/search",
                    params={
                        "q": q,
                        "format": "json",
                        "time_range": time_range,
                        "language": "en",
                        "safesearch": "0",
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
                            extra={"discovery_method": "searxng_dork", "time_filter": "48h"},
                        )
                    )
            except Exception as e:
                logger.debug(f"SearXNG dork query failed: {e}")

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
