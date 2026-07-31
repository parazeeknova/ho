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
from src.radar.sources.discovery import (
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


def _build_query_templates() -> list[str]:
    """Generate bing-compatible search queries — plain words, OR operators."""
    templates: list[str] = []
    for role in _ROLE_FAMILIES[:6]:
        for elig in _ELIGIBILITY[:3]:
            templates.append(f"{role} {elig} hiring")
        for fresh in _FRESHNESS[:2]:
            templates.append(f"{role} {fresh}")
    for _ in range(5):
        r1, r2 = random.sample(_ROLE_FAMILIES[:6], 2)
        e1 = random.choice(_ELIGIBILITY[:3])
        templates.append(f"{r1} OR {r2} {e1}")
    for role in _ROLE_FAMILIES[:4]:
        templates.append(f"{role} remote hiring 2026")
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
        ".teamtailor.com/jobs",
        ".recruitee.com/",
        ".comeet.com/jobs",
        ".jobscore.com/jobs",
        ".jazzhr.com",
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
    max_results_per_query: int = 15,
    max_total_results: int = 150,
) -> list[dict[str, Any]]:
    """Run search discovery: generate queries, fetch, classify, deduplicate."""
    cfg = get_config().searxng
    templates = _build_query_templates()
    queries = random.sample(templates, min(len(templates), 30))

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
                            "engines": "bing,bing news,github",
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
        board_url = _extract_board_root(r["url"])
        name = _extract_company_from_title(r["title"])
        if board_url and name:
            discoveries.append(
                {
                    "name": name,
                    "website": board_url,
                    "source": "search_ats",
                    "provenance_url": r["url"],
                    "direct_job": True,
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
    from src.radar.sources.discovery import _resolve_official_domain

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


def _extract_board_root(job_url: str) -> str:
    """Extract the company-specific board URL from an ATS job URL.

    boards.greenhouse.io/acme/jobs/123 → https://boards.greenhouse.io/acme
    jobs.lever.co/acme/456 → https://jobs.lever.co/acme
    acme.myworkdayjobs.com/careers/job/1 → https://acme.myworkdayjobs.com
    acme.teamtailor.com/jobs/123 → https://acme.teamtailor.com
    acme.recruitee.com/jobs/456 → https://acme.recruitee.com
    acme.comeet.com/jobs/789 → https://acme.comeet.com
    acme.jobscore.com/jobs/123 → https://acme.jobscore.com
    acme.jazzhr.com → https://acme.jazzhr.com
    """
    try:
        p = urlparse(job_url)
    except Exception:
        return ""

    host = (p.hostname or "").lower()
    path = p.path.rstrip("/")

    # Workday: company is the subdomain
    if "myworkdayjobs.com" in host:
        return f"https://{host}"

    # Subdomain-based ATS: company is the subdomain
    for subdomain_ats in (
        ".teamtailor.com",
        ".recruitee.com",
        ".comeet.com",
        ".jobscore.com",
        ".jazzhr.com",
    ):
        if subdomain_ats in host:
            return f"https://{host}"

    # Greenhouse: /{company}/jobs/{id} → /{company}
    if "greenhouse.io" in host:
        parts = [x for x in path.split("/") if x]
        if len(parts) >= 1:
            return f"https://{host}/{parts[0]}"
        return f"https://{host}"

    # icims: /jobs/{id}?company={companyId} — too opaque, skip

    # Lever/Ashby/Workable/SmartRecruiters/Rippling: /{company}/{id}
    parts = [x for x in path.split("/") if x]
    if len(parts) >= 1:
        if "rippling.com" in host and len(parts) >= 2:
            return f"https://{host}/{parts[0]}/{parts[1]}"
        skip = {"jobs", "careers", "job", "postings", "apply"}
        company = parts[0]
        if company.lower() in skip and len(parts) >= 2:
            company = parts[1]
        return f"https://{host}/{company}"
    return f"https://{host}"


def _extract_company_from_title(title: str) -> str:
    """Extract likely company name from a title/snippet.

    Order matters: try unambiguous suffixes first (' is hiring'),
    then symmetric separators (' — ', ' | ', ' - '), then ' at '
    (where company is on the RIGHT: 'SWE at Acme' → 'Acme').
    """
    # Company-first patterns: suffix makes company unambiguous
    for sep in (
        " is hiring",
        " hiring ",
    ):
        parts = title.split(sep, 1)
        if len(parts) > 1:
            name = parts[0].strip()
            if 2 < len(name) < 60:
                return name

    # Role-first pattern: "Role at Company" → company is on the right
    role_at_idx = title.lower().find(" at ")
    if role_at_idx > 0 and role_at_idx < 60:
        name = title[role_at_idx + 4 :].strip()
        if len(name) >= 1:
            return name

    # Symmetric separators: take first side as company
    for sep in (" — ", " | ", " - "):
        parts = title.split(sep, 1)
        if len(parts) > 1:
            name = parts[0].strip()
            if 2 < len(name) < 60:
                return name

    # News headline patterns: company before verb
    for pat in (" raises ", " raised ", " announces ", " launches "):
        idx = title.lower().find(pat)
        if idx > 0:
            return title[:idx].strip()
    return title[:60].strip()
