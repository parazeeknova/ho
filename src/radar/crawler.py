"""SearchDiscoveryCrawler: dynamic query-template rotation for global job discovery.

Generates and rotates SearXNG queries from persona-derived templates:
- Role families × eligibility × freshness × company signals
- Classifies results as official ATS, startup signal, founder post, or aggregator
- Rejects aggregators as final sources but uses them as discovery evidence
- Resolves official domains, detects career/ATS endpoints, persists new sources
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import Any
from urllib.parse import urlparse

import httpx

from src.configuration import get_config
from src.logging import get_logger
from src.radar.discovery import (
    is_aggregator_domain,
)

logger = get_logger("search_crawler")

_ROLE_FAMILIES = [
    "backend engineer",
    "fullstack developer",
    "platform engineer",
    "infrastructure engineer",
    "devops engineer",
    "site reliability engineer",
    "AI engineer",
    "machine learning engineer",
    "data engineer",
    "software engineer",
    "developer tools engineer",
]

_ELIGIBILITY = [
    "visa sponsorship",
    "relocation support",
    "EOR",
    "global remote",
    "work from anywhere",
    "sponsorship available",
]

_FRESHNESS = [
    "hiring now",
    "join our team",
    "we're hiring",
    "new role",
]

_COMPANY_SIGNALS = [
    "seed stage",
    "Series A",
    "YC backed",
    "backed by",
    "recently raised",
    "launched",
    "startup hiring",
]


def _build_query_templates() -> list[str]:
    """Generate a diverse set of search queries rotated each sweep."""
    templates: list[str] = []
    for role in _ROLE_FAMILIES[:4]:  # rotate subset
        for elig in _ELIGIBILITY[:2]:
            templates.append(f'"{role}" "{elig}"')
        for fresh in _FRESHNESS[:2]:
            templates.append(f'"{role}" "{fresh}"')
        for sig in _COMPANY_SIGNALS[:2]:
            templates.append(f'"{role}" "{sig}"')

    for _ in range(3):
        r1, r2 = random.sample(_ROLE_FAMILIES[:6], 2)
        e1 = random.choice(_ELIGIBILITY)
        templates.append(f'"{r1}" OR "{r2}" "{e1}"')

    return templates


def _classify_result(url: str, title: str, snippet: str) -> str:
    text = f"{title} {snippet}".lower()
    url_lower = url.lower()

    ats_signs = (
        "boards.greenhouse.io",
        "jobs.lever.co",
        "jobs.ashbyhq.com",
        "apply.workable.com",
        "jobs.smartrecruiters.com",
        "myworkdayjobs.com",
        "app.rippling.com",
    )
    if any(s in url_lower for s in ats_signs) or any(
        kw in text
        for kw in ("apply", "job description", "qualifications", "requirements", "responsibilities")
    ):
        if is_aggregator_domain(_extract_domain(url)):
            return "aggregator"
        return "ats_job"

    news_sites = ("techcrunch.com", "crunchbase.com", "producthunt.com")
    if any(s in url_lower for s in news_sites) or any(
        kw in text for kw in ("raised", "funding", "announced", "launched", "startup")
    ):
        return "startup_signal"

    linkedin_post = "linkedin.com/posts" in url_lower or "linkedin.com/feed" in url_lower
    hiring_words = ("hiring", "looking for", "join us", "dm me")
    if linkedin_post and any(kw in text for kw in hiring_words):
        return "founder_post"

    if is_aggregator_domain(_extract_domain(url)):
        return "aggregator"

    return "unknown"


async def run_search_discovery(
    max_results_per_query: int = 10,
    max_total_results: int = 80,
) -> list[dict[str, Any]]:
    """Run search discovery: generate queries, fetch, classify, deduplicate."""
    cfg = get_config().searxng
    templates = _build_query_templates()
    queries = random.sample(templates, min(len(templates), 20))

    raw_results: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(3)

    async def _query_one(q: str) -> None:
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                    resp = await client.get(
                        cfg.url,
                        params={
                            "q": q,
                            "format": "json",
                            "time_range": "week",
                        },
                    )
                    if resp.status_code == 200:
                        for r in resp.json().get("results", [])[:max_results_per_query]:
                            url = r.get("url", "")
                            title = r.get("title", "")
                            snippet = r.get("content", "")
                            if not url or not url.startswith("http"):
                                continue
                            classification = _classify_result(url, title, snippet)
                            raw_results.append(
                                {
                                    "url": url,
                                    "title": title,
                                    "snippet": snippet,
                                    "classification": classification,
                                    "query": q,
                                }
                            )
            except Exception:
                pass

    tasks = [asyncio.create_task(_query_one(q)) for q in queries]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Deduplicate by canonical URL
    seen_urls: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in raw_results:
        canonical = _canonical_url(r["url"])
        if canonical not in seen_urls:
            seen_urls.add(canonical)
            deduped.append(r)

    results = deduped[:max_total_results]

    classified: dict[str, list[dict[str, Any]]] = {
        "ats_job": [],
        "startup_signal": [],
        "founder_post": [],
        "aggregator": [],
        "unknown": [],
    }
    for r in results:
        classified[r["classification"]].append(r)

    logger.info(
        f"Search discovery: {len(results)} results "
        f"(ats={len(classified['ats_job'])}, "
        f"signal={len(classified['startup_signal'])}, "
        f"founder={len(classified['founder_post'])}, "
        f"aggregator={len(classified['aggregator'])})"
    )

    # Resolve domains and ATS endpoints for non-aggregator results
    discoveries: list[dict[str, Any]] = []
    for r in classified["ats_job"]:
        domain = _extract_domain(r["url"])
        name = _extract_company_from_title(r["title"])
        if domain and name:
            discoveries.append(
                {
                    "name": name,
                    "website": f"https://{domain}",
                    "source": "search_ats",
                    "provenance_url": r["url"],
                }
            )

    for r in classified["startup_signal"]:
        name = _extract_company_from_title(r["title"])
        if name:
            discoveries.append(
                {
                    "name": name,
                    "website": "",
                    "source": "search_startup",
                    "provenance_url": r["url"],
                }
            )

    for r in classified["founder_post"]:
        name = _extract_company_from_title(r["title"])
        if name:
            discoveries.append(
                {
                    "name": name,
                    "website": "",
                    "source": "search_founder",
                    "provenance_url": r["url"],
                    "founder_signal": True,
                }
            )

    # Resolve domains for name-only discoveries
    from src.radar.discovery import _resolve_official_domain

    for d in discoveries:
        if not d.get("website") or not d["website"].startswith("http"):
            domain = await _resolve_official_domain(d["name"])
            if domain and not is_aggregator_domain(domain):
                d["website"] = f"https://{domain}"

    logger.info(f"Search discovery resolved: {len(discoveries)} potential sources")
    return discoveries


def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        return hashlib.sha256(f"{p.hostname or ''}{p.path.rstrip('/')}".encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(url.encode()).hexdigest()[:16]


def _extract_domain(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _extract_company_from_title(title: str) -> str:
    """Extract likely company name from a title/snippet."""
    for sep in (" is hiring", " hiring ", " at ", " — ", " | ", " - "):
        parts = title.split(sep, 1)
        if len(parts) > 1:
            name = parts[0].strip()
            if 2 < len(name) < 60:
                return name
    # Try H1/H2 pattern: "Company Name raises $X"
    for pat in (" raises ", " raised ", " announces ", " launches "):
        idx = title.lower().find(pat)
        if idx > 0:
            return title[:idx].strip()
    return title[:60].strip()
