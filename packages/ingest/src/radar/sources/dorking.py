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

_TIME_SYNTAX = ""

_ATS_SITES = [
    "site:boards.greenhouse.io",
    "site:jobs.lever.co",
    "site:jobs.ashbyhq.com",
    "site:apply.workable.com",
    "site:jobs.smartrecruiters.com",
    "site:myworkdayjobs.com",
    "site:app.rippling.com",
    "site:jobs.teamtailor.com",
    "site:jobs.recruitee.com",
    "site:jobs.comeet.com",
    "site:jobs.jobscore.com",
    "site:jobs.jazzhr.com",
]

_DORK_TERMS = [
    '"Junior" OR "Entry Level" OR "Associate" OR "Graduate"',
    '"New Grad" OR "2026" OR "Intern" OR "Internship"',
    '"Junior Software Engineer" OR "Entry Level Engineer"',
]

_DORK_QUERIES = [f"{site} {terms}{_TIME_SYNTAX}" for site in _ATS_SITES for terms in _DORK_TERMS]

# Second discovery lane: arbitrary company career pages (companies with neither
# Greenhouse nor Ashby). These queries surface /careers, /jobs, /openings
# across the open web — the review's "scan more of the internet" ask.
_WEB_LANE_QUERIES = [
    'intitle:"careers" "software engineer" (jobs OR openings)',
    'intitle:"jobs" "we are hiring" software',
    '"/careers" "software engineer" "apply" -site:linkedin.com',
    '"career opportunities" software engineer',
]

_DORK_QUERIES += [f"{q}{_TIME_SYNTAX}" for q in _WEB_LANE_QUERIES]


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
        self, queries: list[str] | None = None, time_range: str = "week"
    ) -> list[JobObservation]:
        """Runs time-restricted dork queries against SearXNG.

        Uses a week-long window: a strict day filter returns nothing because
        job-board pages aren't re-crawled that often by search engines.
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
        if "jobs.smartrecruiters.com/" in low:
            parts = low.split("jobs.smartrecruiters.com/")[-1].split("/")
            return parts[0] if parts else "unknown"
        if "jobs.teamtailor.com/" in low:
            parts = low.split("jobs.teamtailor.com/")[-1].split("/")
            return parts[0] if parts else "unknown"
        if "jobs.recruitee.com/" in low:
            parts = low.split("jobs.recruitee.com/")[-1].split("/")
            return parts[0] if parts else "unknown"
        if "myworkdayjobs.com/" in low:
            parts = low.split("myworkdayjobs.com/")[-1].split("/")
            return parts[0] if parts else "unknown"
        # Generic career pages: extract the hostname as the company guess.
        from urllib.parse import urlparse

        host = urlparse(url).netloc or ""
        return host.replace("www.", "").split(".")[0] if host else "searxng-discovered"
