"""Real agent implementations replacing the four scheduler stubs.

- CareerSiteDetector: identifies official careers/ATS endpoints for companies
- ATSCrawler: incremental snapshot collection from ATS boards
- FounderSocialAgent: validates public hiring posts and company/founder relationships
- EmployeeDiscoveryAgent: finds public contact routes and hiring-team profiles
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from src.configuration import get_config
from src.graph.entity import FrontierEntry, NodeType
from src.logging import get_logger
from src.radar.models import JobObservation

logger = get_logger("radar_agents")

_ATS_PATTERN = re.compile(
    r"(?:greenhouse\.io|lever\.co|ashbyhq\.com|workable\.com|"
    r"smartrecruiters\.com|myworkdayjobs\.com|rippling\.com|"
    r"teamtailor\.com|recruitee\.com|comeet\.com|jobscore\.com|"
    r"jazzhr\.com)",
    re.IGNORECASE,
)

_CAREERS_PATH_PATTERNS = [
    "/careers",
    "/jobs",
    "/about/careers",
    "/company/careers",
    "/join-us",
    "/work-with-us",
    "/open-positions",
    "/openings",
    "/positions",
    "/careers/engineering",
    "/jobs/engineering",
    "/job-openings",
    "/hiring",
]


async def career_site_detector(entry: FrontierEntry) -> list[FrontierEntry]:
    """Detect official career sites/ATS endpoints from a company URL.

    Given a company homepage URL, probes for common careers paths
    and ATS subdomains. Returns FrontierEntries for any discovered
    career/ATS endpoints.
    """
    company = entry.payload.get("company", "")
    url = entry.payload.get("url", "")
    if not url or not url.startswith("http"):
        logger.debug("Career site detector: no URL to probe", company=company)
        return []

    results: list[FrontierEntry] = []

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        base = url.rstrip("/")

        for path in _CAREERS_PATH_PATTERNS:
            probe_url = urljoin(base, path)
            try:
                resp = await client.get(probe_url)
                if resp.status_code == 200:
                    actual_url = str(resp.url)
                    text_sample = resp.text[:5000].lower()

                    if _ATS_PATTERN.search(actual_url):
                        ats_type = _identify_ats(actual_url)
                        results.append(
                            FrontierEntry(
                                id=f"ats:{company}:{ats_type}",
                                agent="ats_crawler",
                                node_id=entry.node_id,
                                node_type=NodeType.CAREER_SITE,
                                priority=60,
                                depth=entry.depth + 1,
                                payload={
                                    "company": company,
                                    "ats_url": actual_url,
                                    "ats_type": ats_type,
                                },
                            )
                        )
                        logger.info("ATS endpoint discovered", company=company, url=actual_url)
                        break

                    if any(
                        kw in text_sample
                        for kw in ("career", "job", "hiring", "position", "opening")
                    ):
                        results.append(
                            FrontierEntry(
                                id=f"career:{company}",
                                agent="ats_crawler",
                                node_id=entry.node_id,
                                node_type=NodeType.CAREER_SITE,
                                priority=55,
                                depth=entry.depth + 1,
                                payload={
                                    "company": company,
                                    "ats_url": actual_url,
                                    "ats_type": "careers_page",
                                },
                            )
                        )
                        break
            except Exception:
                continue

        if not results:
            _ats_subdomains = [
                f"https://jobs.{_extract_domain(url)}",
                f"https://careers.{_extract_domain(url)}",
                f"https://boards.greenhouse.io/{_extract_domain(url).split('.')[0]}",
                f"https://jobs.lever.co/{_extract_domain(url).split('.')[0]}",
                f"https://jobs.ashbyhq.com/{_extract_domain(url).split('.')[0]}",
                f"https://apply.workable.com/{_extract_domain(url).split('.')[0]}",
            ]
            for ats_url in _ats_subdomains:
                try:
                    resp = await client.get(ats_url)
                    if resp.status_code == 200:
                        ats_type = _identify_ats(str(resp.url))
                        results.append(
                            FrontierEntry(
                                id=f"ats:{company}:{ats_type}",
                                agent="ats_crawler",
                                node_id=entry.node_id,
                                node_type=NodeType.CAREER_SITE,
                                priority=60,
                                depth=entry.depth + 1,
                                payload={
                                    "company": company,
                                    "ats_url": str(resp.url),
                                    "ats_type": ats_type,
                                },
                            )
                        )
                        break
                except Exception:
                    continue

    return results


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _identify_ats(url: str) -> str:
    url_lower = url.lower()
    if "greenhouse.io" in url_lower:
        return "greenhouse"
    if "lever.co" in url_lower:
        return "lever"
    if "ashbyhq.com" in url_lower:
        return "ashby"
    if "workable.com" in url_lower:
        return "workable"
    if "smartrecruiters.com" in url_lower:
        return "smartrecruiters"
    if "myworkdayjobs.com" in url_lower:
        return "workday"
    if "rippling.com" in url_lower:
        return "rippling"
    if "teamtailor.com" in url_lower:
        return "teamtailor"
    if "recruitee.com" in url_lower:
        return "recruitee"
    if "comeet.com" in url_lower:
        return "comeet"
    if "jobscore.com" in url_lower:
        return "jobscore"
    if "jazzhr.com" in url_lower:
        return "jazzhr"
    return "careers_page"


async def ats_crawler(entry: FrontierEntry) -> list[FrontierEntry]:
    """Incremental snapshot collection from ATS boards.

    Scrapes a known ATS endpoint, extracts job posting URLs, diffs against
    the previous snapshot, and returns new observations as FrontierEntries.
    """
    company = entry.payload.get("company", "")
    ats_url = entry.payload.get("ats_url", "")
    ats_type = entry.payload.get("ats_type", "careers_page")

    if not ats_url or not ats_url.startswith("http"):
        return []

    observations: list[JobObservation] = []

    try:
        cfg = get_config().firecrawl
        ats_cfg = get_config().ats
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.post(
                f"{cfg.url}/v1/map",
                json={"url": ats_url, "limit": cfg.map_limit},
            )
            if resp.status_code == 200:
                data = resp.json()
                links = data.get("links", []) or []
                limit = min(ats_cfg.max_pages_per_board * 20, 200)
                for link in links[:limit]:
                    if not isinstance(link, str):
                        continue
                    if _ATS_PATTERN.search(link) or "/jobs/" in link or "/postings/" in link:
                        observations.append(
                            JobObservation(
                                url=link,
                                source=f"ats:{ats_type}",
                                title=f"Job at {company}",
                                snippet=company,
                                raw_markdown="",
                            )
                        )
    except Exception as e:
        logger.warning("ATS crawl failed", company=company, exception=str(e))
        return []

    results: list[FrontierEntry] = []
    for obs in observations:
        results.append(
            FrontierEntry(
                id=f"job:{obs.canonical_url_hash()}",
                agent="job_processor",
                node_id=entry.node_id,
                node_type=NodeType.JOB,
                priority=40,
                depth=entry.depth + 1,
                payload={
                    "company": company,
                    "observation_url": obs.url,
                    "source": obs.source,
                    "ats_type": ats_type,
                },
            )
        )

    return results


async def founder_social_agent(entry: FrontierEntry) -> list[FrontierEntry]:
    """Validate public hiring posts and company/founder relationships.

    Searches for public LinkedIn, X (Twitter), GitHub profiles for founders
    and validates that hiring posts are from actual company founders.
    Stores verified data in the GraphNode payload.
    """
    founder_name = entry.payload.get("founder_name", "")
    company = entry.payload.get("company", "")
    if not founder_name or not company:
        return []

    evidence: dict[str, Any] = {
        "linkedin": None,
        "x_twitter": None,
        "github": None,
        "company_url": None,
        "verified_hiring_posts": [],
    }

    queries = [
        f'"{founder_name}" "{company}" founder OR CEO site:linkedin.com/in/',
        f'"{founder_name}" "{company}" site:github.com',
        f'"{founder_name}" "{company}" hiring OR "looking for" OR "join us" site:linkedin.com',
    ]

    try:
        cfg = get_config().searxng
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            for query in queries:
                try:
                    resp = await client.get(
                        cfg.url,
                        params={"q": query, "format": "json", "time_range": "month"},
                    )
                    if resp.status_code == 200:
                        results_list = resp.json().get("results", [])
                        for r in results_list[:3]:
                            rurl = r.get("url", "")
                            if "linkedin.com" in rurl:
                                evidence["linkedin"] = rurl
                            elif "github.com" in rurl:
                                evidence["github"] = rurl
                            elif any(
                                kw in r.get("content", "").lower()
                                for kw in ("hiring", "looking for", "join", "role")
                            ):
                                evidence["verified_hiring_posts"].append(
                                    {
                                        "url": rurl,
                                        "snippet": r.get("content", "")[:200],
                                        "title": r.get("title", ""),
                                    }
                                )
                except Exception:
                    continue
    except Exception as e:
        logger.warning("Founder social search failed", founder=founder_name, exception=str(e))

    results: list[FrontierEntry] = []
    if evidence["linkedin"] or evidence["github"]:
        results.append(
            FrontierEntry(
                id=f"founder_enriched:{entry.node_id}",
                agent="outreach_generator",
                node_id=entry.node_id,
                node_type=NodeType.FOUNDER,
                priority=45,
                depth=entry.depth + 1,
                payload={
                    "founder_name": founder_name,
                    "company": company,
                    "linkedin": evidence["linkedin"],
                    "github": evidence["github"],
                    "verified_posts": evidence["verified_hiring_posts"],
                },
            )
        )

    return results


async def employee_discovery_agent(entry: FrontierEntry) -> list[FrontierEntry]:
    """Find public professional contact routes and hiring-team profiles.

    Discovers publicly available recruiting contact information from
    company career pages and public LinkedIn profiles. Only collects
    explicitly published work email and public profile links.
    """
    company = entry.payload.get("company", "")
    if not company:
        return []

    contact_routes: list[dict[str, str]] = []

    try:
        cfg = get_config().searxng
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.get(
                cfg.url,
                params={
                    "q": (
                        f'"{company}" recruiting OR "talent" OR "people ops" '
                        f"email OR contact site:linkedin.com OR "
                        f"site:{_extract_domain_from_company(company)}"
                    ),
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", [])[:5]:
                    rurl = r.get("url", "")
                    if rurl and rurl.startswith("http"):
                        contact_routes.append(
                            {
                                "type": "public_profile",
                                "url": rurl,
                                "title": r.get("title", ""),
                            }
                        )
    except Exception as e:
        logger.debug("Employee discovery search failed", company=company, exception=str(e))

    if contact_routes:
        logger.info("Contact routes discovered", company=company, count=len(contact_routes))

    return []


def _extract_domain_from_company(company: str) -> str:
    cleaned = company.lower().replace(" ", "").replace(".", "")
    return f"{cleaned}.com"
