"""Startup Discovery Pipeline — parallel producer that discovers companies from
YC batches, VC portfolios, Product Hunt, GitHub organizations, HN who's hiring,
and other startup datasets independently of job listings.

Every discovered startup becomes a first-class entity enriched with founders,
funding, careers URLs, and hiring signals — regardless of whether a job has
already been found.
"""

from __future__ import annotations

import asyncio
import random

from src.configuration import get_config
from src.http_client import get_client
from src.logging import get_logger

logger = get_logger("startup_discoverer")

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",  # noqa: E501
    "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0",  # noqa: E501
]


YC_COMPANIES_API = "https://api.ycombinator.com/v0/companies"
VC_PORTFOLIOS = {
    "a16z": "https://a16z.com/portfolio/",
    "sequoia": "https://www.sequoiacap.com/our-companies/",
    "accel": "https://www.accel.com/companies",
    "benchmark": "https://www.benchmark.com/portfolio/",
    "foundersfund": "https://foundersfund.com/portfolio/",
    "lightspeed": "https://lsvp.com/companies/",
    "greylock": "https://greylock.com/portfolio/",
    "khosla": "https://www.khoslaventures.com/portfolio/",
    "insight": "https://www.insightpartners.com/portfolio/",
    "bessemer": "https://www.bvp.com/companies",
}
PRODUCT_HUNT_API = "https://api.producthunt.com/v2/api/graphql"
GITHUB_TRENDING_API = "https://api.github.com/search/repositories"


async def _searxng_discover(query: str) -> list[dict[str, str]]:
    """Use SearXNG to discover startups from a domain-specific query."""
    results: list[dict[str, str]] = []
    try:
        cfg = get_config().searxng
        client = await get_client("startup_discoverer", timeout=cfg.timeout)
        resp = await client.get(
            cfg.url,
            params={
                "q": query,
                "format": "json",
                "time_range": "month",
            },
            headers={"User-Agent": random.choice(_USER_AGENTS)},
        )
        if resp.status_code == 200:
            for r in resp.json().get("results", [])[:10]:
                url = r.get("url", "")
                if url and url.startswith("http"):
                    results.append(
                        {
                            "company": r.get("title", "").split("|")[0].strip(),
                            "description": r.get("content", ""),
                            "url": url,
                            "source": "searxng",
                        }
                    )
    except Exception as e:
        logger.warning("Startup discoverer SearXNG failed", source="searxng", exception=str(e))
    return results


async def discover_yc_companies() -> list[dict[str, str]]:
    """Pull recent YC batches. Falls back to SearXNG if API not reachable."""
    discovered: list[dict[str, str]] = []
    try:
        client = await get_client("startup_discoverer", timeout=10.0)
        resp = await client.get(
            YC_COMPANIES_API,
            params={"batch": "W25", "limit": "50"},
            headers={"User-Agent": random.choice(_USER_AGENTS)},
        )
        if resp.status_code == 200:
            data = resp.json()
            for c in data.get("companies", [])[:50]:
                discovered.append(
                    {
                        "company": c.get("name", ""),
                        "description": c.get("short_description", ""),
                        "url": f"https://www.ycombinator.com/companies/{c.get('slug', '')}",
                        "source": "yc",
                    }
                )
            if discovered:
                return discovered
    except Exception as e:
        logger.warning("YC API failed", source="yc", exception=str(e))

    return await _searxng_discover('site:ycombinator.com/companies "founded" "team size"')


async def discover_vc_portfolio(vc_name: str, vc_url: str) -> list[dict[str, str]]:
    """Scrape a VC portfolio page for company names."""
    return await _searxng_discover(
        f'site:{vc_url.split("/")[2]} "portfolio" OR "companies" startup'
    )


async def discover_product_hunt() -> list[dict[str, str]]:
    """Discover trending products/startups on Product Hunt via SearXNG."""
    return await _searxng_discover('site:producthunt.com "launched" OR "maker" "upvotes"')


async def discover_github_startups() -> list[dict[str, str]]:
    """Discover active startup GitHub orgs and repos."""
    queries = [
        'site:github.com "open source" "funding" OR "backed by" startup',
        'site:github.com "we are hiring" OR "join us" "seed" OR "series a" startup',
    ]
    results: list[dict[str, str]] = []
    for q in queries:
        results.extend(await _searxng_discover(q))
    return results


async def discover_hn_hiring() -> list[dict[str, str]]:
    """Discover startups from HN 'Who is Hiring' threads."""
    return await _searxng_discover('site:news.ycombinator.com "who is hiring" startup hiring')


async def discover_founder_hiring_posts() -> list[dict[str, str]]:
    """Discover founders actively hiring on social media."""
    return await _searxng_discover(
        '("hiring" OR "looking for" OR "join us") ("founder" OR "CEO" OR "CTO") '
        '("seed" OR "series a" OR "pre-seed" OR "stealth") startup'
    )


async def discover_startups(positions: list[str]) -> list[dict[str, str]]:
    """Discover startups from all sources in parallel.

    Returns a deduplicated list of company dicts with keys:
    company, description, url, source.
    """
    tasks: list[asyncio.Task[list[dict[str, str]]]] = []

    tasks.append(asyncio.create_task(discover_yc_companies()))
    tasks.append(asyncio.create_task(discover_product_hunt()))
    tasks.append(asyncio.create_task(discover_github_startups()))
    tasks.append(asyncio.create_task(discover_hn_hiring()))
    tasks.append(asyncio.create_task(discover_founder_hiring_posts()))

    for vc_name, vc_url in list(VC_PORTFOLIOS.items())[:4]:
        tasks.append(asyncio.create_task(discover_vc_portfolio(vc_name, vc_url)))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen: set[str] = set()
    all_companies: list[dict[str, str]] = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for company in result:
            name = company.get("company", "").strip().lower()
            if name and name not in seen and len(name) > 1:
                seen.add(name)
                all_companies.append(company)

    logger.info(f"Discovered {len(all_companies)} startups from {len(tasks)} sources")
    return all_companies
