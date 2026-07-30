"""Continuous company discovery from public startup ecosystems.

Discovered via live endpoint scan (2026-07-30):
  READABLE: Crunchbase, Dealroom, Failory, Antler, Plug and Play,
    a16z, Sequoia, Khosla, BVP, Index Ventures, Insight Partners, GV,
    Salesforce Ventures, We Work Remotely, Arc, BetaList, TechCrunch
  JS-SPA (via Firecrawl): YC, Techstars, 500 Global, Accel, Greylock,
    NEA, First Round, Felicis, Tracxn, Himalayas, DevHunt, Lever, Workable
  BLOCKED 403: PitchBook, OpenVC, F6S, StartupBlink, EF, Lightspeed,
    Founders Fund, Sapphire, M12, Wellfound, ProductHunt, Sifted
  404: Benchmark, General Catalyst, Alchemist (no portfolio page)

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
    "teamtailor": ".teamtailor.com",
    "recruitee": ".recruitee.com",
    "comeet": ".comeet.com",
    "jobscore": ".jobscore.com",
    "jazzhr": ".jazzhr.com",
}

_VC_PORTFOLIOS = [
    "https://a16z.com/portfolio/",
    "https://www.sequoiacap.com/our-companies/",
    "https://www.accel.com/companies",
    "https://www.khoslaventures.com/portfolio/",
    "https://www.bvp.com/portfolio",
    "https://www.indexventures.com/companies/",
    "https://www.insightpartners.com/portfolio/",
    "https://www.gv.com/portfolio/",
    "https://www.salesforceventures.com/portfolio/",
    "https://greylock.com/portfolio/",
    "https://www.nea.com/portfolio",
    "https://firstround.com/companies/",
    "https://www.felicis.com/companies",
    "https://500.co/companies",
    "https://www.techstars.com/portfolio/",
    "https://www.coatue.com/portfolio/",
    "https://www.intelcapital.com/portfolio/",
    "https://www.qualcommventures.com/portfolio/",
    "https://visionfund.com/portfolio/",
    "https://www.balderton.com/companies/",
    "https://northzone.com/portfolio/",
    "https://initialized.com/companies/",
    "https://luxcapital.com/companies/",
    "https://www.crv.com/companies/",
]


async def discover_from_yc(limit: int = 50) -> list[dict[str, str]]:
    """Discover YC companies via SearXNG search for YC-backed startups.

    YC's official directory is a React SPA blocked by anti-bot
    protection — even Firecrawl Playwright returns empty HTML.
    We use SearXNG to find YC company list mirrors and directories.
    """
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    cfg = get_config().searxng
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.get(
                cfg.url,
                params={
                    "q": "YC W25 YC S25 YC-backed startup companies list 2026",
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return companies
            for r in resp.json().get("results", [])[:30]:
                title = r.get("title", "")
                snippet = r.get("content", "")
                text = f"{title} {snippet}"
                for m in re.finditer(r"([A-Z][A-Za-z0-9 .&,-]{3,50})", text):
                    name = m.group(1).strip()
                    if (
                        name not in seen
                        and len(name) > 3
                        and not any(
                            n in name.lower()
                            for n in (
                                "home",
                                "about",
                                "list",
                                "search",
                                "login",
                                "signup",
                                "apply now",
                            )
                        )
                    ):
                        seen.add(name)
                        companies.append({"name": name, "website": "", "source": "yc_directory"})
                        if len(companies) >= limit:
                            return companies
    except Exception:
        pass

    for c in companies:
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
    return companies


async def discover_from_vc_portfolios(limit: int = 40) -> list[dict[str, str]]:
    """Discover companies from VC portfolio pages via Firecrawl.

    All VC portfolio pages are JS-rendered (React/WordPress); static HTTP
    won't produce company names. We use Firecrawl's Playwright-backed
    scrape to get rendered content.
    """
    companies: list[dict[str, str]] = []
    cfg = get_config().firecrawl
    async with httpx.AsyncClient(timeout=60.0) as client:
        for portfolio_url in _VC_PORTFOLIOS:
            try:
                resp = await client.post(
                    f"{cfg.url}/v1/scrape",
                    json={"url": portfolio_url, "formats": ["html"], "onlyMainContent": True},
                )
                if resp.status_code != 200:
                    continue
                html = (resp.json().get("data") or {}).get("html", "") or ""
                if not html:
                    continue
                names = _extract_portfolio_company_names(html, limit)
                for name in names:
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
    seen: set[str] = set()
    noise = {
        "home",
        "about",
        "contact",
        "careers",
        "portfolio",
        "menu",
        "jobs",
        "team",
        "news",
        "blog",
        "press",
        "events",
        "privacy",
        "terms",
    }

    for pat in (
        r"<a[^>]*href=\"https?://([^\"]+)\"[^>]*>([^<]{2,40})</a>",
        r"<h3[^>]*>([^<]{2,60})</h3>",
        r"<span[^>]*>([^<]{2,60})</span>",
        r"<p[^>]*>([^<]{2,60})</p>",
    ):
        for m in re.finditer(pat, html, re.IGNORECASE):
            name = (m.lastindex and m.group(m.lastindex)) or m.group(1)
            name = _clean_name(name)
            if (
                2 < len(name) < 60
                and name.lower() not in noise
                and not name.startswith("http")
                and name not in seen
            ):
                seen.add(name)
                names.append(name)
            if len(names) >= limit:
                return names[:limit]
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


async def discover_from_dealroom(limit: int = 50) -> list[dict[str, str]]:
    """Discover funded startups from Dealroom.co's public API."""
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Get fresh market map IDs — try multiple queries
            all_map_ids: list[str] = []
            for q in ("ai", "saas", "fintech", "health"):
                resp = await client.get(
                    "https://dealroom.co/api/marketmaps",
                    params={"q": q, "limit": 3},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for m in data.get("results", []):
                        mid = m.get("id", "")
                        if mid and mid not in all_map_ids:
                            all_map_ids.append(mid)

            for map_id in all_map_ids[:6]:
                resp2 = await client.get(f"https://dealroom.co/api/marketmap?id={map_id}")
                if resp2.status_code != 200:
                    continue
                data = resp2.json()
                for comp in data.get("companies", []):
                    name = (comp.get("name") or "").strip()
                    website = (comp.get("website") or "").strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    companies.append(
                        {
                            "name": name,
                            "website": website,
                            "source": "dealroom",
                            "funding_amount": _extract_funding(comp),
                            "hq_city": _extract_hq_city(comp),
                        }
                    )
                    if len(companies) >= limit:
                        break
    except Exception:
        pass

    # Resolve domains for companies without websites
    for c in companies:
        if not c["website"] or not c["website"].startswith("http"):
            domain = await _resolve_official_domain(c["name"])
            if domain and not is_aggregator_domain(domain):
                c["website"] = f"https://{domain}"
    return companies


def _extract_funding(comp: dict) -> float:
    funding = comp.get("totalFunding")
    if isinstance(funding, dict):
        amt = funding.get("amount", 0)
        return float(amt) if amt else 0.0
    return 0.0


def _extract_hq_city(comp: dict) -> str:
    hq = comp.get("hq")
    if isinstance(hq, dict):
        return (hq.get("city") or "").strip()
    return ""


async def discover_from_remoteok(limit: int = 30) -> list[dict[str, str]]:
    """Discover company names from the RemoteOK public API.

    RemoteOK provides structured JSON with company, position, location, salary,
    and tags. We extract company names and resolve their official domains.
    """
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://remoteok.com/api?tag=dev")
            if resp.status_code != 200:
                return companies
            data = resp.json()
            for entry in data[1:]:  # first entry is legal notice
                company = (entry.get("company") or "").strip()
                position = (entry.get("position") or "").strip()
                if not company or company in seen:
                    continue
                # Only take tech roles
                tech_terms = (
                    "engineer",
                    "developer",
                    "backend",
                    "frontend",
                    "fullstack",
                    "devops",
                    "software",
                    "platform",
                    "infrastructure",
                    "data",
                    "ml",
                    "ai",
                    "machine learning",
                    "sre",
                )
                if not any(t in position.lower() for t in tech_terms):
                    continue
                seen.add(company)
                companies.append(
                    {
                        "name": company,
                        "website": "",
                        "source": "remoteok",
                    }
                )
                if len(companies) >= limit:
                    break
    except Exception:
        pass

    # Resolve official domains for discovered companies
    for c in companies:
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
    return companies


async def discover_from_weworkremotely(limit: int = 30) -> list[dict[str, str]]:
    """Discover companies from We Work Remotely's readable HTML listings.

    WWR is a plain-HTML remote job board — no JS rendering needed.
    Each job card contains a company name and apply link.
    """
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://weworkremotely.com/")
            if resp.status_code != 200:
                return companies
            html = resp.text
            for m in re.finditer(
                r'<span class="company">([^<]+)</span>',
                html,
                re.IGNORECASE,
            ):
                name = m.group(1).strip()
                if name and name not in seen and 2 < len(name) < 80:
                    seen.add(name)
                    companies.append({"name": name, "website": "", "source": "weworkremotely"})
                    if len(companies) >= limit:
                        break
    except Exception:
        pass

    for c in companies:
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
    return companies


async def discover_from_betalist(limit: int = 30) -> list[dict[str, str]]:
    """Discover startups from BetaList's readable HTML listing.

    BetaList showcases recently launched startups with company names,
    descriptions, and website links.
    """
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://betalist.com/")
            if resp.status_code != 200:
                return companies
            html = resp.text
            for m in re.finditer(
                r'<a[^>]*href="(/startup/[^"]+)"[^>]*>([^<]+)</a>',
                html,
                re.IGNORECASE,
            ):
                name = m.group(2).strip()
                if name and name not in seen and 2 < len(name) < 80:
                    seen.add(name)
                    companies.append({"name": name, "website": "", "source": "betalist"})
                    if len(companies) >= limit:
                        break
    except Exception:
        pass

    for c in companies:
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
    return companies


async def _resolve_official_domain(name: str) -> str:
    """Resolve a company name to its official domain via SearXNG."""
    cfg = get_config().searxng
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.get(
                cfg.url,
                params={
                    "q": f'"{name}" official website OR careers',
                    "format": "json",
                },
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", [])[:5]:
                    url_str = r.get("url", "")
                    domain = _extract_domain(url_str)
                    if domain and not is_aggregator_domain(domain):
                        return domain
    except Exception:
        pass
    # Fallback: direct HTTP probe
    return await _resolve_company_domain(name)


def is_aggregator_domain(domain: str) -> bool:
    from src.radar.governor import is_aggregator_domain as _check

    return _check(domain)


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
