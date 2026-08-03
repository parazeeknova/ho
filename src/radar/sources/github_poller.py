"""Layer 2: Open-Source GitHub Index ETag Poller.

Polls top community GitHub README repositories (SimplifyJobs, PittCSC, etc.)
through the shared HTTP response cache, which issues conditional requests
(If-None-Match / If-Modified-Since) and returns 304s at zero bandwidth.
Unchanged files are served from the cache and skipped without re-parsing.
"""

from __future__ import annotations

import asyncio

from src.http_cache import cached_get
from src.http_client import get_client
from src.logging import get_logger
from src.radar.core.extractors import extract_github_index_markdown
from src.radar.core.models import JobObservation

logger = get_logger("github_poller")

# Hardcoded raw markdown URLs for top open-source job repositories
COMMUNITY_GITHUB_INDEXES: list[str] = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
    "https://raw.githubusercontent.com/pittcsc/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
]

_LAST_OBS_HASHES: dict[str, set[str]] = {}


async def poll_github_index_etag(url: str) -> tuple[list[JobObservation], bool]:
    """Poll a single GitHub raw markdown URL through the shared HTTP cache.

    Returns:
        (observations, was_modified)
        If the response came from cache (304 / unchanged body): ([], False)
        If 200 OK: returns (new_observations, True)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/plain, text/markdown",
    }

    try:
        client = await get_client("github_poller", timeout=15.0)
        resp = await cached_get(client, url, headers=headers)

        if resp.status_code != 200:
            return [], False

        if resp.extensions.get("cached"):
            logger.debug(f"GitHub ETag 304 Not Modified: {url}")
            return [], False

        markdown = resp.text
        repo_name = url.split("githubusercontent.com/")[-1].rsplit("/", 1)[0]
        source_id = f"github_index:{repo_name}"

        all_obs = extract_github_index_markdown(markdown, source_id)

        # Diff against last seen hashes for this repo
        prev_hashes = _LAST_OBS_HASHES.get(url, set())
        current_hashes = {o.canonical_url_hash() for o in all_obs}
        _LAST_OBS_HASHES[url] = current_hashes

        new_obs = [o for o in all_obs if o.canonical_url_hash() not in prev_hashes]
        logger.info(f"GitHub ETag 200 OK ({source_id}): {len(all_obs)} total, {len(new_obs)} new")
        return new_obs if prev_hashes else all_obs, True

    except Exception as exc:
        logger.warning(f"GitHub ETag polling failed for {url}: {exc}")

    return [], False


async def poll_all_github_indexes_etag() -> list[JobObservation]:
    """Poll all community GitHub indexes concurrently with ETag caching."""
    tasks = [poll_github_index_etag(url) for url in COMMUNITY_GITHUB_INDEXES]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined_obs: list[JobObservation] = []
    for r in results:
        if isinstance(r, tuple) and len(r) == 2:
            obs, modified = r
            if obs:
                combined_obs.extend(obs)

    logger.info(f"GitHub ETag poller complete: {len(combined_obs)} new observations fetched")
    return combined_obs
