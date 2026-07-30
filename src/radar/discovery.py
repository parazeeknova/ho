"""Continuous company discovery from public startup ecosystems.

Discovers companies from:
- YC accelerator directories
- VC portfolio pages (a16z, Sequoia, Benchmark, Accel)
- Wellfound/startup-job pages
- HN "Who is Hiring"
- SearXNG hiring/funding/launch queries

Every discovered company's website is probed for its ATS/career page.
Generic vendor roots are never used.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import httpx

from src.configuration import get_config
from src.logging import get_logger

logger = get_logger("discovery")

_CAREERS_PATHS = (
    "/careers",
    "/jobs",
    "/about/careers",
    "/company/careers",
    "/join-us",
    "/work-with-us",
    "/open-positions",
    "/openings",
)

ATS_SIGNATURES = {
    "greenhouse": "boards.greenhouse.io",
    "lever": "jobs.lever.co",
    "ashby": "jobs.ashbyhq.com",
    "workable": "apply.workable.com",
    "smartrecruiters": "jobs.smartrecruiters.com",
    "workday": "myworkdayjobs.com",
    "rippling": "app.rippling.com",
}

_VC_PORTFOLIOS = [
    "https://a16z.com/portfolio/",
    "https://www.sequoiacap.com/our-companies/",
    "https://www.benchmark.com/portfolio/",
    "https://www.accel.com/companies",
]


async def discover_from_yc(limit: int = 50) -> list[dict[str, str]]:
    companies: list[dict[str, str]] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get("https://www.ycombinator.com/companies")
            if resp.status_code != 200:
                return companies
            links = re.findall(r'href="(/companies/[^"]+)"', resp.text)
            seen = set()
            for path in links[:limit]:
                if path in seen:
                    continue
                seen.add(path)
                try:
                    detail = await client.get(f"https://www.ycombinator.com{path}")
                    if detail.status_code == 200:
                        name = _extract_name(detail.text, path)
                        site = _extract_website(detail.text)
                        if name:
                            companies.append(
                                {"name": name, "website": site or "", "source": "yc_directory"}
                            )
                except Exception:
                    pass
    except Exception as e:
        logger.warning("YC discovery failed", exception=str(e))
    return companies


async def discover_from_vc_portfolios(limit: int = 40) -> list[dict[str, str]]:
    companies: list[dict[str, str]] = []
    for portfolio_url in _VC_PORTFOLIOS:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(portfolio_url)
                if resp.status_code != 200:
                    continue
                names = _extract_portfolio_company_names(resp.text, limit)
                for name in names:
                    # Resolve official domain for portfolio companies
                    domain = await _resolve_company_domain(name)
                    companies.append(
                        {
                            "name": name,
                            "website": f"https://{domain}" if domain else "",
                            "source": "vc_portfolio",
                        }
                    )
        except Exception:
            continue
    return companies[:limit]


def _extract_portfolio_company_names(html: str, limit: int = 30) -> list[str]:
    names: list[str] = []
    seen = set()
    # Try common portfolio page patterns
    for pat in (
        r'"name"\s*:\s*"([^"]+)"',
        r'alt="([^"]+)"[^>]*class="[^"]*logo',
        r"<h3[^>]*>([^<]+)</h3>",
    ):
        for m in re.finditer(pat, html, re.IGNORECASE):
            name = m.group(1).strip()
            if (
                2 < len(name) < 60
                and name.lower() not in ("home", "about", "contact")
                and name not in seen
            ):
                seen.add(name)
                names.append(name)
    return names[:limit]


async def _resolve_company_domain(name: str) -> str:
    """Resolve official domain for a company name."""
    slug = name.lower().replace(" ", "").replace(".", "").replace("-", "")
    candidates = [
        f"https://{slug}.com",
        f"https://{slug}.io",
        f"https://{slug}.co",
        f"https://www.{slug}.com",
        f"https://{slug}.ai",
    ]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in candidates:
                try:
                    resp = await client.head(url)
                    if resp.status_code < 400:
                        return urlparse(url).hostname or slug
                except Exception:
                    continue
    except Exception:
        pass
    return ""


async def discover_from_hackernews(limit: int = 30) -> list[dict[str, str]]:
    companies: list[dict[str, str]] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://hn.algolia.com/api/v1/search_by_date",
                params={
                    "query": "Ask HN: Who is hiring",
                    "tags": "story",
                    "hitsPerPage": min(limit, 30),
                },
            )
            if resp.status_code == 200:
                for hit in resp.json().get("hits", []):
                    title = hit.get("title", "")
                    if not title or "who is hiring" not in title.lower():
                        continue
                    # Fetch the actual thread page and extract companies from comments
                    obj_id = hit.get("objectID", "")
                    thread_url = f"https://news.ycombinator.com/item?id={obj_id}" if obj_id else ""
                    if thread_url:
                        thread_resp = await client.get(thread_url)
                        if thread_resp.status_code == 200:
                            for line in thread_resp.text.split("\n"):
                                # HN comment format has company names with URLs
                                m = re.search(r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>', line)
                                if m:
                                    name = m.group(2).strip()
                                    company_url = m.group(1)
                                    if 2 < len(name) < 60 and "http" in company_url:
                                        domain = _extract_domain(company_url)
                                        if domain:
                                            companies.append(
                                                {
                                                    "name": name,
                                                    "website": f"https://{domain}",
                                                    "source": "hackernews",
                                                }
                                            )
        companies = companies[:limit]
    except Exception:
        pass
    return companies


async def discover_from_wellfound(limit: int = 30) -> list[dict[str, str]]:
    companies: list[dict[str, str]] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get("https://wellfound.com/jobs")
            if resp.status_code != 200:
                return companies
            seen: set[str] = set()
            for m in re.finditer(r'href="/company/([^"]+)"', resp.text):
                slug = m.group(1)
                if slug in seen:
                    continue
                seen.add(slug)
                name = slug.replace("-", " ").title()
                # Try to resolve company page for website
                try:
                    company_resp = await client.get(f"https://wellfound.com/company/{slug}")
                    if company_resp.status_code == 200:
                        site = _extract_website(company_resp.text)
                        companies.append(
                            {
                                "name": name,
                                "website": site or f"https://wellfound.com/company/{slug}",
                                "source": "wellfound",
                            }
                        )
                        continue
                except Exception:
                    pass
                companies.append({"name": name, "website": "", "source": "wellfound"})
    except Exception:
        pass
    return companies[:limit]


async def discover_from_searxng(kind: str = "hiring") -> list[dict[str, str]]:
    companies: list[dict[str, str]] = []
    cfg = get_config().searxng

    queries = {
        "funding": (
            'site:techcrunch.com OR site:crunchbase.com "raised" '
            '"seed" OR "series a" startup hiring'
        ),
        "hiring": ('site:linkedin.com/posts "hiring" "software engineer" "startup" "remote"'),
        "launch": (
            'site:producthunt.com OR site:wellfound.com "hiring" "software engineer" remote'
        ),
    }

    query = queries.get(kind, queries["hiring"])
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.get(
                cfg.url, params={"q": query, "format": "json", "time_range": "week"}
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", [])[:20]:
                    title = r.get("title", "")
                    url_str = r.get("url", "")
                    if not title or not url_str:
                        continue
                    domain = _extract_domain(url_str)
                    if domain:
                        companies.append(
                            {
                                "name": _clean_name(title),
                                "website": f"https://{domain}",
                                "source": f"searxng_{kind}",
                            }
                        )
    except Exception:
        pass
    return companies


async def detect_ats_for_company(website: str) -> str | None:
    if not website.startswith("http"):
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            base = website.rstrip("/")
            for path in _CAREERS_PATHS:
                try:
                    resp = await client.get(urljoin(base, path))
                    if resp.status_code == 200:
                        actual = str(resp.url)
                        for _sig_name, sig in ATS_SIGNATURES.items():
                            if sig in actual.lower():
                                return actual
                        return actual
                except Exception:
                    continue
            domain = _extract_domain(website)
            base_name = domain.split(".")[0]
            for guess in (
                f"https://boards.greenhouse.io/{base_name}",
                f"https://jobs.lever.co/{base_name}",
                f"https://jobs.ashbyhq.com/{base_name}",
                f"https://apply.workable.com/{base_name}",
            ):
                try:
                    resp = await client.get(guess)
                    if resp.status_code == 200:
                        return str(resp.url)
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _extract_name(html: str, path: str) -> str:
    m = re.search(r"<title>([^<]+)</title>", html)
    if m:
        return m.group(1).replace(" | Y Combinator", "").replace(" | YC", "").strip()
    return path.rsplit("/", 1)[-1].replace("-", " ").title()


def _extract_website(html: str) -> str:
    m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*(?:website|Visit)\s*<', html, re.IGNORECASE)
    return m.group(1) if m else ""


def _clean_name(title: str) -> str:
    for sep in (" - ", " | ", " — ", " hiring", " raises"):
        idx = title.lower().find(sep)
        if idx > 0:
            return title[:idx].strip()
    return title.strip()[:80]


def _extract_domain(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""
