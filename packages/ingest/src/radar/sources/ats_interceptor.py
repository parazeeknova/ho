"""Layer 1: Direct ATS API Interceptor.

Provides high-speed, 0-token direct JSON API fetching for Greenhouse, Lever,
Ashby, Workable, and SmartRecruiters.
"""

from __future__ import annotations

import re
from typing import Any

from src.http_cache import cached_get
from src.logging import get_logger
from src.radar.core.models import JobObservation

logger = get_logger("ats_interceptor")


def parse_ats_slug(url: str) -> tuple[str, str] | None:
    """Extract (platform, slug) from an ATS board URL.

    Supported platforms: greenhouse, lever, ashby, workable, smartrecruiters.
    """
    if not url:
        return None
    u = url.lower().strip()

    # Greenhouse: boards.greenhouse.io/{slug} or greenhouse.io/{slug}
    m = re.search(r"greenhouse\.io/(?:embed/job_board\?for=)?([^/\?#]+)", u)
    if m and m.group(1) not in ("embed", "v1"):
        return ("greenhouse", m.group(1))

    # Lever: jobs.lever.co/{slug}
    m = re.search(r"jobs\.lever\.co/([^/\?#]+)", u)
    if m:
        return ("lever", m.group(1))

    # Ashby: jobs.ashbyhq.com/{slug}
    m = re.search(r"jobs\.ashbyhq\.com/([^/\?#]+)", u)
    if m:
        return ("ashby", m.group(1))

    # Workable: apply.workable.com/{slug}
    m = re.search(r"apply\.workable\.com/([^/\?#]+)", u)
    if m:
        return ("workable", m.group(1))

    # SmartRecruiters: jobs.smartrecruiters.com/{slug}
    m = re.search(r"jobs\.smartrecruiters\.com/([^/\?#]+)", u)
    if m:
        return ("smartrecruiters", m.group(1))

    return None


async def _fetch_ats_jobs_result(platform: str, slug: str) -> list[dict[str, Any]] | None:
    """Fetch an ATS API response, returning ``None`` only when it is unavailable.

    An empty list is a successful response from a board with no openings (or
    an invalid slug).  Keeping that distinct from a transport failure lets the
    poller avoid an expensive browser fallback for every empty ATS endpoint.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    timeout = 10.0
    jobs: list[dict[str, Any]] = []

    from src.http_client import get_client

    client = await get_client("ats_interceptor", timeout=timeout)
    try:
        if platform == "greenhouse":
            # GET https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
            api_url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
            resp = await cached_get(client, api_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("jobs", []):
                    jobs.append(
                        {
                            "url": item.get("absolute_url", ""),
                            "title": item.get("title", ""),
                            "location": (item.get("location") or {}).get("name", ""),
                            "updated_at": item.get("updated_at", ""),
                            "raw_markdown": item.get("content", ""),
                        }
                    )

        elif platform == "lever":
            # GET https://api.lever.co/v0/postings/{slug}
            api_url = f"https://api.lever.co/v0/postings/{slug}"
            resp = await cached_get(client, api_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        categories = item.get("categories") or {}
                        jobs.append(
                            {
                                "url": item.get("hostedUrl") or item.get("applyUrl", ""),
                                "title": item.get("text", ""),
                                "location": categories.get("location", ""),
                                "updated_at": item.get("createdAt", ""),
                                "raw_markdown": (
                                    item.get("descriptionPlain", "") or item.get("description", "")
                                ),
                            }
                        )

        elif platform == "ashby":
            # GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true
            api_url = (
                f"https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true"
            )
            resp = await cached_get(client, api_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("jobs", []):
                    job_id = item.get("id", "")
                    job_url = f"https://jobs.ashbyhq.com/{slug}/{job_id}" if job_id else ""
                    jobs.append(
                        {
                            "url": job_url,
                            "title": item.get("title", ""),
                            "location": item.get("locationName", ""),
                            "updated_at": item.get("publishedAt", ""),
                            "raw_markdown": item.get("descriptionHtml", ""),
                        }
                    )

        elif platform == "workable":
            # GET https://apply.workable.com/api/v1/widget/accounts/{slug}
            api_url = f"https://apply.workable.com/api/v1/widget/accounts/{slug}"
            resp = await cached_get(client, api_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("jobs", []):
                    shortcode = item.get("shortcode", "")
                    job_url = (
                        f"https://apply.workable.com/{slug}/j/{shortcode}/" if shortcode else ""
                    )
                    jobs.append(
                        {
                            "url": job_url,
                            "title": item.get("title", ""),
                            "location": (item.get("location") or {}).get("city", ""),
                            "updated_at": item.get("published", ""),
                            "raw_markdown": item.get("description", ""),
                        }
                    )

        elif platform == "smartrecruiters":
            # GET https://api.smartrecruiters.com/v1/companies/{slug}/postings
            api_url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
            resp = await cached_get(client, api_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("content", []):
                    job_id = item.get("id", "")
                    job_url = f"https://jobs.smartrecruiters.com/{slug}/{job_id}" if job_id else ""
                    jobs.append(
                        {
                            "url": job_url,
                            "title": item.get("name", ""),
                            "location": (item.get("location") or {}).get("city", ""),
                            "updated_at": item.get("releasedDate", ""),
                            "raw_markdown": "",
                        }
                    )

    except Exception as exc:
        logger.debug(f"ATS API fetch failed for {platform}:{slug}: {exc}")
        return None

    return jobs


async def fetch_ats_jobs(platform: str, slug: str) -> list[dict[str, Any]]:
    """Fetch all open job postings directly via official JSON API endpoints.

    This public helper retains its historical list-only contract.  Callers
    that need to distinguish an empty board from an unreachable API use the
    internal result helper above.
    """
    return await _fetch_ats_jobs_result(platform, slug) or []


async def intercept_ats_board(board_url: str, source_id: str) -> list[JobObservation] | None:
    """Attempt to intercept ATS board using direct JSON APIs.

    Returns list[JobObservation] if successfully intercepted, or None to fallback.
    """
    parsed = parse_ats_slug(board_url)
    if not parsed:
        return None

    platform, slug = parsed
    raw_jobs = await _fetch_ats_jobs_result(platform, slug)
    if raw_jobs is None:
        return None

    observations: list[JobObservation] = []
    for item in raw_jobs:
        url = item.get("url", "")
        if not url or not url.startswith("http"):
            continue
        obs = JobObservation(
            url=url,
            source=source_id,
            title=item.get("title", ""),
            snippet=f"Location: {item.get('location', 'Remote')} | ATS: {platform}",
            raw_markdown=item.get("raw_markdown", ""),
            source_freshness_evidence=item.get("updated_at", ""),
        )
        obs.extra["official_source"] = True
        obs.extra["ats_platform"] = platform
        obs.extra["company_slug"] = slug
        observations.append(obs)

    if observations:
        logger.info(
            f"ATS Interceptor ({platform}:{slug}): {len(observations)} jobs fetched via direct API"
        )
    return observations
