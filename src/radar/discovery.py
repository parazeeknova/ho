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

import asyncio
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
    import datetime

    now = datetime.datetime.now()
    yr = int(str(now.year)[-2:])
    batches = [f"YC W{yr - i} YC S{yr - i} YC{yr - i}" for i in range(4)]
    query = f"{' '.join(batches)} YC-backed startup companies list"

    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.get(
                cfg.url,
                params={
                    "q": query,
                    "format": "json",
                    "engines": "bing,bing news,github",
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

    logger.info(f"YC discovery: resolving domains for {len(companies)} companies...")
    for i, c in enumerate(companies):
        if i > 0 and i % 10 == 0:
            logger.info(f"YC domain resolution: {i}/{len(companies)}")
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
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(timeout=60.0) as client:

        async def _scrape_one(p_url):
            async with sem:
                try:
                    resp = await client.post(
                        f"{cfg.url}/v1/scrape",
                        json={"url": p_url, "formats": ["html"], "onlyMainContent": True},
                    )
                    if resp.status_code != 200:
                        return ""
                    return (resp.json().get("data") or {}).get("html", "") or ""
                except Exception:
                    return ""

        tasks = [_scrape_one(url) for url in _VC_PORTFOLIOS]
        total_vc = len(_VC_PORTFOLIOS)
        logger.info(f"Scraping {total_vc} VC portfolio pages via Firecrawl...")
        htmls = await asyncio.gather(*tasks)
        done = sum(1 for h in htmls if h)
        logger.info(f"VC portfolios: {done}/{total_vc} scraped successfully")
        total_names = 0
        resolved = 0
        processed = 0
        for html in htmls:
            if not html:
                continue
            names = _extract_portfolio_company_names(html, limit)
            total_names += len(names)
            for name in names:
                domain = await _resolve_company_domain(name)
                processed += 1
                if domain:
                    resolved += 1
                companies.append(
                    {
                        "name": name,
                        "website": f"https://{domain}" if domain else "",
                        "source": "vc_portfolio",
                    }
                )
                if processed % 10 == 0:
                    logger.info(
                        f"VC domain resolution: "
                        f"{processed}/{total_names} companies, "
                        f"{resolved} resolved",
                    )
        logger.info(
            f"VC portfolios: {total_names} companies extracted, {resolved} domains resolved",
        )
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
                        host = urlparse(url).hostname or slug
                        logger.debug(f"Domain resolved: '{name}' -> {host}")
                        return host
                except Exception:
                    continue
    except Exception:
        pass
    logger.debug(f"Domain resolution failed for '{name}'")
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
        logger.info(f"HN discovery: {len(companies)} companies from stories")
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
                        # Only landscape-* IDs have company data; skip others
                        if mid and mid not in all_map_ids and mid.startswith("landscape-"):
                            all_map_ids.append(mid)

            for map_id in all_map_ids[:8]:
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
    need_resolve = [c for c in companies if not c["website"] or not c["website"].startswith("http")]
    if need_resolve:
        logger.info(f"Dealroom: resolving domains for {len(need_resolve)} companies...")
    for i, c in enumerate(companies):
        if not c["website"] or not c["website"].startswith("http"):
            domain = await _resolve_official_domain(c["name"])
            if domain and not is_aggregator_domain(domain):
                c["website"] = f"https://{domain}"
            if (i + 1) % 10 == 0:
                logger.info(f"Dealroom domain resolution: {i + 1}/{len(companies)}")
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

    logger.info(f"RemoteOK: {len(companies)} companies with tech roles")
    # Resolve official domains for discovered companies
    if companies:
        logger.info(f"RemoteOK: resolving domains for {len(companies)} companies...")
    for i, c in enumerate(companies):
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
        if (i + 1) % 10 == 0:
            logger.info(f"RemoteOK domain resolution: {i + 1}/{len(companies)}")
    return companies


async def discover_from_weworkremotely(limit: int = 30) -> list[dict[str, str]]:
    """Discover companies from We Work Remotely via Firecrawl.

    WWR is Cloudflare-protected; direct httpx gets 403. Use Firecrawl's
    Playwright-backed scrape to bypass the challenge and get rendered HTML.
    """
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    cfg = get_config().firecrawl
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{cfg.url}/v1/scrape",
                json={
                    "url": "https://weworkremotely.com/",
                    "formats": ["html"],
                    "onlyMainContent": True,
                },
            )
            if resp.status_code != 200:
                return companies
            html = (resp.json().get("data") or {}).get("html", "") or ""
            if not html:
                return companies
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

    if companies:
        logger.info(f"WeWorkRemotely: resolving domains for {len(companies)} companies...")
    for i, c in enumerate(companies):
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
        if (i + 1) % 10 == 0:
            logger.info(f"WWR domain resolution: {i + 1}/{len(companies)}")
    return companies


async def discover_from_betalist(limit: int = 30) -> list[dict[str, str]]:
    """Discover startups from BetaList's readable HTML listing.

    BetaList uses <a href="/startups/{slug}"> overlay links with empty inner
    text. Company names live in <div class="font-medium ...">Name</div> blocks.
    We extract slugs from hrefs and derive names from them.
    """
    companies: list[dict[str, str]] = []
    seen: set[str] = set()
    slugs: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://betalist.com/")
            if resp.status_code != 200:
                return companies
            html = resp.text
            for m in re.finditer(
                r'<a[^>]*href="/startups/([^/"]+)"',
                html,
                re.IGNORECASE,
            ):
                slug = m.group(1).strip().lower()
                if slug and slug not in ("follow", "edit", "stats", "new"):
                    slugs.append(slug)

            seen_slugs: set[str] = set()
            for slug in slugs:
                if slug in seen_slugs:
                    continue
                seen_slugs.add(slug)
                name = slug.replace("-", " ").title()
                if name and name not in seen and 2 < len(name) < 80:
                    seen.add(name)
                    companies.append({"name": name, "website": "", "source": "betalist"})
                    if len(companies) >= limit:
                        break
    except Exception:
        pass

    logger.info(f"BetaList: {len(companies)} startups from {len(slugs)} slugs")
    if companies:
        logger.info(f"BetaList: resolving domains for {len(companies)} companies...")
    for i, c in enumerate(companies):
        domain = await _resolve_official_domain(c["name"])
        if domain and not is_aggregator_domain(domain):
            c["website"] = f"https://{domain}"
        if (i + 1) % 10 == 0:
            logger.info(f"BetaList domain resolution: {i + 1}/{len(companies)}")
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
                    "engines": "bing,bing news,github",
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
                                logger.info(f"ATS found: {website} -> {actual} ({_sig_name})")
                                return actual
                        logger.info(f"Careers page found: {website} -> {actual} (no known ATS)")
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
                        logger.debug(f"ATS vendor match: {resp.url}")
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
