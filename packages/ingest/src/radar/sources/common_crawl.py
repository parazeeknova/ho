"""Common Crawl discovery (the review's primary architecture recommendation).

The Azure relic harvested ATS slugs from Common Crawl. We bring that into the
local pipeline directly: query the Common Crawl Index API
(https://index.commoncrawl.org) for URLs that match ATS board patterns across
EVERY family (greenhouse, lever, ashby, workable, smartrecruiters, workday,
rippling, teamtailor, recruitee, comeet, jobscore, jazzhr). This discovers
boards without a manually curated company list — the crawl IS the company list.

The index is a public read-only API; no key needed. Each result is a WARC/URL
record. We only need the URL (the company slug is in the path).
"""

from __future__ import annotations

import json

from src.http_client import get_client
from src.logging import get_logger

logger = get_logger("common_crawl_discovery")

# URL substrings that identify a job-board page on each ATS family. The index
# search matches on the URL; a board slug (company) is the first path segment.
_ATS_BOARD_MARKERS = [
    "boards.greenhouse.io/",
    "jobs.lever.co/",
    "jobs.ashbyhq.com/",
    "apply.workable.com/",
    "jobs.smartrecruiters.com/",
    "myworkdayjobs.com/",
    "app.rippling.com/",
    "jobs.teamtailor.com/",
    "jobs.recruitee.com/",
    "jobs.comeet.com/",
    "jobs.jobscore.com/",
    "jobs.jazzhr.com/",
]

_INDEX_HOST = "https://index.commoncrawl.org"
_INDEXES = ["CC-MAIN-2025-30", "CC-MAIN-2025-26", "CC-MAIN-2025-22"]

# marker prefix -> ATS platform name (for the source record).
_MARKER_PLATFORM = {
    "boards.greenhouse.io/": "greenhouse",
    "jobs.lever.co/": "lever",
    "jobs.ashbyhq.com/": "ashby",
    "apply.workable.com/": "workable",
    "jobs.smartrecruiters.com/": "smartrecruiters",
    "myworkdayjobs.com/": "workday",
    "app.rippling.com/": "rippling",
    "jobs.teamtailor.com/": "teamtailor",
    "jobs.recruitee.com/": "recruitee",
    "jobs.comeet.com/": "comeet",
    "jobs.jobscore.com/": "jobscore",
    "jobs.jazzhr.com/": "jazzhr",
}


def _ats_marker_for_url(url: str) -> str | None:
    low = url.lower()
    for m in _ATS_BOARD_MARKERS:
        if m in low:
            return m
    return None


def _slug_from_url(url: str, marker: str) -> str | None:
    """Company slug = the first path segment after the ATS domain.

    Strips trailing query strings and fragments ('?gh_src=...' on greenhouse)
    and rejects pure-numeric / junk slugs."""
    try:
        tail = url.split(marker, 1)[1]
        slug = tail.split("/")[0].strip().split("?")[0].split("#")[0]
        if not slug:
            return None
        # Reject obvious junk: all-digits, or slugs with no letters.
        if not any(ch.isalpha() for ch in slug):
            return None
        return slug
    except Exception:
        return None


async def discover_from_common_crawl(
    limit: int = 1500, indexes: list[str] | None = None
) -> list[dict[str, str]]:
    """Query the Common Crawl URL index for ATS board pages across all families.

    For each index snapshot, search for each ATS marker (a collquery/urlsearch
    style index query). Deduplicate by (marker, slug) so one board -> one source.
    """
    client = await get_client("commoncrawl", timeout=30.0)
    target_indexes = indexes or _INDEXES
    found: dict[str, str] = {}  # key: (marker,slug) -> board_url
    for idx in target_indexes:
        for marker in _ATS_BOARD_MARKERS:
            if len(found) >= limit:
                break
            try:
                # url index search: matchUrls contains the marker.
                resp = await client.get(
                    f"{_INDEX_HOST}/{idx}-index?url={marker}*&output=json&limit=500"
                )
                if resp.status_code != 200:
                    continue
                # The API returns newline-delimited JSON records.
                for line in resp.text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    url = rec.get("url") or rec.get("filename") or ""
                    if not url:
                        continue
                    m = _ats_marker_for_url(url)
                    if not m:
                        continue
                    slug = _slug_from_url(url, m)
                    if not slug:
                        continue
                    key = (m, slug)
                    if key in found:
                        continue
                    # Rebuild a canonical board URL from the marker + slug.
                    domain = m.rstrip("/")
                    board_url = f"https://{domain}/{slug}"
                    found[key] = {
                        "name": slug,
                        "website": board_url,
                        "ats_url": board_url,
                        "source": "common_crawl",
                        "platform": _MARKER_PLATFORM.get(m, "unknown"),
                    }
                    if len(found) >= limit:
                        break
            except Exception as e:
                logger.debug(f"Common Crawl index {idx} query {marker} failed: {e}")
        if len(found) >= limit:
            break
    logger.info(f"common_crawl discovery: {len(found)} boards from {len(target_indexes)} indexes")
    return list(found.values())


def is_common_crawl_configured() -> bool:
    """The public index needs no config; always available. (Kept for symmetry.)"""
    return True
