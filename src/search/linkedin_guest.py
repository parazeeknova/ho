"""LinkedIn Guest API Miner — scrapes unauthenticated job listings from LinkedIn's
publicly accessible guest API without cookies or logins.

The endpoint at /jobs-guest/jobs/api/seeMoreJobPostings/search returns raw HTML,
not JSON, so we parse it with BeautifulSoup.
"""

from __future__ import annotations

import asyncio
import random
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from src.pipeline.queue import JobPipeline, QueuedJob

GUEST_API = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"

LINKEDIN_PARAMS = {
    "trk": "public_jobs_jobs-search-bar_search-submit",
    "start": 0,
}

# -- Domains that Linkedin often wraps in redirects we don't want ------------
LINKEDIN_TRACKING = re.compile(r"https?://(?:www\.)?linkedin\.com/jobs/view/.*")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
}


def _strip_tracking(url: str) -> str:
    """Remove query parameters and tracking garbage from a job URL."""
    if not url:
        return ""
    p = urlparse(url)
    clean = f"{p.scheme}://{p.netloc}{p.path}"
    if m := LINKEDIN_TRACKING.search(clean):
        return m.group(0)
    return clean


async def scrape_linkedin_guest_jobs(
    keyword: str,
    location: str = "Remote",
    pipeline: JobPipeline | None = None,
    max_pages: int = 4,
) -> list[dict[str, str]]:
    """Paginate the LinkedIn guest API and push scraped listings into
    *pipeline* (if given).  Returns the full list of discovered job dicts.

    Parameters
    ----------
    keyword : str
        Search query, e.g. ``"Backend Engineer"``.
    location : str
        Location filter, e.g. ``"Remote"`` or ``"India"``.
    pipeline : JobPipeline or None
        If provided, each job is pushed as a ``QueuedJob`` as it is discovered.
    max_pages : int
        Number of pages (25 results each) to scrape.  Default 4 (100 results).
    """
    discovered: list[dict[str, str]] = []

    async def _fetch_page(start: int) -> None:
        params = {**LINKEDIN_PARAMS, "start": str(start)}
        if keyword:
            params["keywords"] = keyword
        if location:
            params["location"] = location

        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                resp = await client.get(GUEST_API, params=params, headers=HEADERS)
                if resp.status_code != 200:
                    return
                soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"  [dim]LinkedIn guest page {start}: {e}[/dim]")
            return

        cards = soup.select("a.base-card__full-link")
        for card in cards:
            url = _strip_tracking(card.get("href", ""))
            if not url:
                continue

            title_el = card.find("h3", class_="base-search-card__title")
            company_el = card.find("h4", class_="base-search-card__subtitle")

            title = title_el.get_text(strip=True) if title_el else ""
            company = company_el.get_text(strip=True) if company_el else ""

            if not title or not company:
                continue

            job = {
                "role": title,
                "company": company,
                "apply_link": url,
                "url": url,
                "source": "linkedin_guest",
            }
            discovered.append(job)

            if pipeline is not None:
                desc = f"**{title}** at {company}\nLinkedIn Guest API listing.\nApply: {url}"
                await pipeline.push(QueuedJob(markdown=desc, url=url, title=title))

    for start in range(0, max_pages * 25, 25):
        await _fetch_page(start)
        if start < (max_pages - 1) * 25:
            delay = random.uniform(1.5, 3.5)
            await asyncio.sleep(delay)

    return discovered
