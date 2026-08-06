"""Layer 1: ATS Mass Poller -- High-speed 10,000+ slug JSON API polling.

Harvests company slugs from GitHub README.md repositories, then runs an
async loop that pings Greenhouse, Lever, and Ashby JSON endpoints every
4 hours. Only returns jobs posted within the last 48 hours whose titles
contain intern/grad/junior keywords.

Costs zero tokens, takes milliseconds per slug, and gives exact Unix
timestamps of when each job was posted.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from src.http_client import get_client
from src.logging import get_logger
from src.radar.core.models import JobObservation

logger = get_logger("ats_mass_poller")

GREENHOUSE_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
LEVER_API = "https://api.lever.co/v0/postings/{slug}"
ASHBY_API = "https://api.ashbyhq.com/posting-api/job-board/{slug}"

# Per-request ceiling for a single slug poll. A handful of boards (notably
# Lever) can take 15-35s to respond; without a hard cap one slow slug stalls
# the entire platform's asyncio.gather and the sweep hangs.
MASS_POLL_REQUEST_TIMEOUT = 20.0


async def _timed_request(client: Any, method: str, url: str, **kwargs: Any) -> Any:
    """Issue one ATS request under a hard wall-clock cap.

    The httpx client timeout covers socket ops but the whole request (DNS +
    connect + read + json) can still exceed it if the server dribbles bytes.
    Wrapping in asyncio.timeout bounds the total so a slow board can never
    block the sweep.
    """
    try:
        async with asyncio.timeout(MASS_POLL_REQUEST_TIMEOUT):
            return await getattr(client, method)(url, **kwargs)
    except TimeoutError:
        return None
    except Exception:
        return None


_COMMUNITY_README_URLS = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
    "https://raw.githubusercontent.com/pittcsc/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
    "https://raw.githubusercontent.com/cvrve/Summer2025-Internships/main/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2025-Internships/dev/README.md",
    "https://raw.githubusercontent.com/Coding-Crashkurse/Summer-2026-Internships/main/README.md",
    "https://raw.githubusercontent.com/AlanChen4/Summer-2026-SWE-Internships/main/README.md",
]

_TITLE_KEYWORDS = re.compile(
    r"intern|internship|new grad|graduate|university|early career|"
    r"entry.level|junior|associate|campus|co.op|coop",
    re.IGNORECASE,
)

_ATS_SLUG_RE = re.compile(
    r"https?://(?:boards\.greenhouse\.io|jobs\.lever\.co|jobs\.ashbyhq\.com|"
    r"apply\.workable\.com)/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)

_SLUG_STORE: dict[str, set[str]] = {"greenhouse": set(), "lever": set(), "ashby": set()}
_SLUG_LAST_HARVEST = 0.0
_SLUG_COUNT = 0


async def harvest_slugs_from_github_readmes() -> int:
    """Pull all community GitHub README.md files and regex-extract company slugs.

    Scrapes every ATS URL from the raw markdown, extracts the {company_slug},
    and deduplicates them into the global _SLUG_STORE.

    Returns the total number of unique slugs harvested across all platforms.
    """
    global _SLUG_STORE, _SLUG_LAST_HARVEST, _SLUG_COUNT

    client = await get_client("ats_mass_poller", timeout=15.0)
    new_slugs = 0
    for url in _COMMUNITY_README_URLS:
        try:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                },
                timeout=15.0,
            )
            if resp.status_code != 200:
                logger.debug(f"Slug harvest: {url} returned {resp.status_code}")
                continue

            matches = _ATS_SLUG_RE.findall(resp.text)
            for slug in matches:
                slug = slug.lower().strip().rstrip("/")
                if not slug or slug in ("embed", "v1", "api", "www"):
                    continue

                is_gh = "boards.greenhouse.io" in resp.text.lower()
                is_lev = "jobs.lever.co" in resp.text.lower() or "lever.co" in resp.text.lower()
                is_ash = "ashbyhq.com" in resp.text.lower()

                if is_gh and slug not in _SLUG_STORE["greenhouse"]:
                    _SLUG_STORE["greenhouse"].add(slug)
                    new_slugs += 1
                if is_lev and slug not in _SLUG_STORE["lever"]:
                    _SLUG_STORE["lever"].add(slug)
                    new_slugs += 1
                if is_ash and slug not in _SLUG_STORE["ashby"]:
                    _SLUG_STORE["ashby"].add(slug)
                    new_slugs += 1
        except Exception as exc:
            logger.debug(f"Slug harvest failed for {url}: {exc}")

    _SLUG_LAST_HARVEST = time.monotonic()
    _SLUG_COUNT = sum(len(v) for v in _SLUG_STORE.values())
    logger.info(f"Slug harvest: {new_slugs} new slugs (total: {_SLUG_COUNT} across 3 platforms)")
    return new_slugs


async def get_all_slugs() -> dict[str, set[str]]:
    """Return the current slug database. Triggers harvest if empty."""
    total = sum(len(v) for v in _SLUG_STORE.values())
    if total == 0:
        await harvest_slugs_from_github_readmes()
    return _SLUG_STORE


async def _poll_greenhouse_slugs(slugs: set[str]) -> list[dict[str, Any]]:
    """Ping all Greenhouse slugs concurrently and filter by freshness + title."""
    jobs: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(30)

    async def _fetch(slug: str) -> None:
        async with sem:
            try:
                client = await get_client("ats_mass_poller", timeout=25.0)
                url = GREENHOUSE_API.format(slug=slug)
                resp = await _timed_request(
                    client,
                    "get",
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "Accept": "application/json",
                    },
                )
                if resp is None or resp.status_code != 200:
                    return
                data = resp.json()
                for item in data.get("jobs", []):
                    title = item.get("title", "")
                    if not _TITLE_KEYWORDS.search(title):
                        continue
                    updated = item.get("updated_at", "")
                    if not _is_within_48h(updated):
                        continue
                    jobs.append(
                        {
                            "url": item.get("absolute_url", ""),
                            "title": title,
                            "location": (item.get("location") or {}).get("name", ""),
                            "updated_at": updated,
                            "platform": "greenhouse",
                            "slug": slug,
                        }
                    )
            except Exception:
                pass

    tasks = [asyncio.create_task(_fetch(s)) for s in slugs]
    await asyncio.gather(*tasks, return_exceptions=True)
    return jobs


async def _poll_lever_slugs(slugs: set[str]) -> list[dict[str, Any]]:
    """Ping all Lever slugs concurrently and filter by freshness + title."""
    jobs: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(30)

    async def _fetch(slug: str) -> None:
        async with sem:
            try:
                client = await get_client("ats_mass_poller", timeout=25.0)
                url = LEVER_API.format(slug=slug)
                resp = await _timed_request(
                    client,
                    "get",
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "Accept": "application/json",
                    },
                )
                if resp is None or resp.status_code != 200:
                    return
                data = resp.json()
                if not isinstance(data, list):
                    return
                for item in data:
                    title = item.get("text", "")
                    if not _TITLE_KEYWORDS.search(title):
                        continue
                    created = item.get("createdAt", "")
                    if not _is_within_48h(created):
                        continue
                    jobs.append(
                        {
                            "url": item.get("hostedUrl") or item.get("applyUrl", ""),
                            "title": title,
                            "location": (item.get("categories") or {}).get("location", ""),
                            "updated_at": created,
                            "platform": "lever",
                            "slug": slug,
                        }
                    )
            except Exception:
                pass

    tasks = [asyncio.create_task(_fetch(s)) for s in slugs]
    await asyncio.gather(*tasks, return_exceptions=True)
    return jobs


async def _poll_ashby_slugs(slugs: set[str]) -> list[dict[str, Any]]:
    """Ping all Ashby slugs concurrently and filter by freshness + title."""
    jobs: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(30)

    async def _fetch(slug: str) -> None:
        async with sem:
            try:
                client = await get_client("ats_mass_poller", timeout=25.0)
                url = ASHBY_API.format(slug=slug)
                resp = await _timed_request(
                    client,
                    "post",
                    url,
                    json={"includeCompensation": True},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "Accept": "application/json",
                    },
                )
                if resp is None or resp.status_code != 200:
                    return
                data = resp.json()
                for item in data.get("jobs", []):
                    title = item.get("title", "")
                    if not _TITLE_KEYWORDS.search(title):
                        continue
                    published = item.get("publishedAt", "")
                    if not _is_within_48h(published):
                        continue
                    job_id = item.get("id", "")
                    job_url = f"https://jobs.ashbyhq.com/{slug}/{job_id}" if job_id else ""
                    jobs.append(
                        {
                            "url": job_url,
                            "title": title,
                            "location": item.get("locationName", ""),
                            "updated_at": published,
                            "platform": "ashby",
                            "slug": slug,
                        }
                    )
            except Exception:
                pass

    tasks = [asyncio.create_task(_fetch(s)) for s in slugs]
    await asyncio.gather(*tasks, return_exceptions=True)
    return jobs


def _is_within_48h(timestamp: str) -> bool:
    """Check if a UTC timestamp is within the last 48 hours."""
    if not timestamp:
        return False
    try:
        from datetime import UTC, datetime, timedelta

        ts = timestamp.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    continue
            else:
                return False

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        cutoff = datetime.now(UTC) - timedelta(hours=48)
        return dt > cutoff
    except Exception:
        return False


async def poll_all_mass_slugs() -> list[JobObservation]:
    """Ping the full slug database across all 3 ATS platforms.

    Harvests slugs if needed, then concurrently polls Greenhouse, Lever,
    and Ashby endpoints. Only returns jobs with intern/grad/junior titles
    posted within the last 48 hours.

    Returns list of JobObservations ready for gating and LLM matching.
    """
    slugs = await get_all_slugs()
    total_slugs = sum(len(v) for v in slugs.values())
    if total_slugs == 0:
        logger.warning("Mass poller: no slugs available")
        return []

    logger.info(f"Mass poller: polling {total_slugs} slugs across 3 platforms...")
    t0 = time.monotonic()

    gh_task = asyncio.create_task(_poll_greenhouse_slugs(slugs.get("greenhouse", set())))
    lv_task = asyncio.create_task(_poll_lever_slugs(slugs.get("lever", set())))
    ab_task = asyncio.create_task(_poll_ashby_slugs(slugs.get("ashby", set())))

    results = await asyncio.gather(gh_task, lv_task, ab_task, return_exceptions=True)

    all_jobs: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, list):
            all_jobs.extend(r)

    elapsed = time.monotonic() - t0
    logger.info(
        f"Mass poller complete: {len(all_jobs)} fresh junior-job hits "
        f"from {total_slugs} slugs in {elapsed:.1f}s"
    )

    observations: list[JobObservation] = []
    for item in all_jobs:
        url = item.get("url", "")
        if not url or not url.startswith("http"):
            continue
        obs = JobObservation(
            url=url,
            source=f"mass_poller:{item.get('platform', 'unknown')}",
            title=item.get("title", ""),
            snippet=(
                f"Location: {item.get('location', 'Remote')} | "
                f"ATS: {item.get('platform', '')} | "
                f"Posted: {item.get('updated_at', '')}"
            ),
            raw_markdown=item.get("raw_markdown", ""),
            source_freshness_evidence=item.get("updated_at", ""),
        )
        obs.extra["official_source"] = True
        obs.extra["ats_platform"] = item.get("platform", "")
        obs.extra["company_slug"] = item.get("slug", "")
        obs.extra["mass_polled"] = True
        observations.append(obs)

    return observations


async def mass_poll_loop(
    interval_seconds: float = 14400,
    callback=None,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Run a continuous loop that polls all ATS slugs every 4 hours.

    Args:
        interval_seconds: Poll interval (default 4 hours = 14400s).
        callback: Async callable receiving list[JobObservation] from each poll.
        shutdown_event: Event to signal shutdown.
    """
    first_run = True
    while True:
        if shutdown_event and shutdown_event.is_set():
            return

        try:
            if first_run:
                await harvest_slugs_from_github_readmes()
                first_run = False

            jobs = await poll_all_mass_slugs()
            if jobs and callback:
                await callback(jobs)
        except Exception as exc:
            logger.warning(f"Mass poll loop iteration failed: {exc}")

        if shutdown_event:
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
                return
            except TimeoutError:
                continue
        else:
            await asyncio.sleep(interval_seconds)
