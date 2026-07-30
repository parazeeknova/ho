"""Continuous company discovery from public startup ecosystems.

Discovers companies from:
- YC accelerator directories
- VC portfolio pages
- Public funding/launch announcements
- SearXNG hiring/funding queries
- Wellfound, HN "Who is Hiring"

Every discovered company's website is probed for its specific ATS/career page.
Generic vendor roots are never used.
"""

from __future__ import annotations

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


async def discover_from_yc(limit: int = 50) -> list[dict[str, str]]:
    """Discover YC companies from YC directory pages."""
    companies: list[dict[str, str]] = []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get("https://www.ycombinator.com/companies")
            if resp.status_code != 200:
                return companies
            import re

            links = re.findall(r'href="(/companies/[^"]+)"', resp.text)
            seen = set()
            for path in links[:limit]:
                if path in seen:
                    continue
                seen.add(path)
                try:
                    detail_resp = await client.get(
                        f"https://www.ycombinator.com{path}",
                    )
                    if detail_resp.status_code == 200:
                        name = _extract_name(detail_resp.text, path)
                        site = _extract_website(detail_resp.text)
                        if name:
                            companies.append(
                                {
                                    "name": name,
                                    "website": site or "",
                                    "source": "yc_directory",
                                }
                            )
                except Exception:
                    pass
    except Exception as e:
        logger.warning("YC discovery failed", exception=str(e))
    return companies


async def discover_from_searxng(kind: str = "hiring") -> list[dict[str, str]]:
    """Discover companies via SearXNG search."""
    companies: list[dict[str, str]] = []
    cfg = get_config().searxng

    queries = {
        "funding": (
            'site:techcrunch.com OR site:crunchbase.com "raised" '
            '"seed" OR "series a" OR "series b" funding startup '
            "engineering hiring"
        ),
        "hiring": (
            'site:linkedin.com/posts "hiring" "software engineer" '
            '"YC" OR "backed by" OR "seed" OR "series a"'
        ),
        "launch": (
            "site:producthunt.com OR site:wellfound.com "
            '"software engineer" "hiring" "remote" startup'
        ),
    }

    query = queries.get(kind, queries["hiring"])
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            resp = await client.get(
                cfg.url,
                params={"q": query, "format": "json", "time_range": "week"},
            )
            if resp.status_code == 200:
                for r in resp.json().get("results", [])[:20]:
                    title = r.get("title", "")
                    url = r.get("url", "")
                    if not title or not url:
                        continue
                    domain = _extract_domain(url)
                    if domain:
                        companies.append(
                            {
                                "name": _clean_company_name(title),
                                "website": f"https://{domain}",
                                "source": f"searxng_{kind}",
                            }
                        )
    except Exception as e:
        logger.debug("SearXNG discovery failed", kind=kind, exception=str(e))
    return companies


async def detect_ats_for_company(website: str) -> str | None:
    """Probe a company's website for its specific ATS/career page."""
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
                        for _ats_name, sig in ATS_SIGNATURES.items():
                            if sig in actual.lower():
                                return actual
                        return actual
                except Exception:
                    continue

            domain = _extract_domain(website)
            base_name = domain.split(".")[0]
            ats_guesses = [
                f"https://boards.greenhouse.io/{base_name}",
                f"https://jobs.lever.co/{base_name}",
                f"https://jobs.ashbyhq.com/{base_name}",
                f"https://apply.workable.com/{base_name}",
            ]
            for guess in ats_guesses:
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
    import re

    m = re.search(r"<title>([^<]+)</title>", html)
    if m:
        title = m.group(1)
        for suffix in (" | Y Combinator", " | YC", " - Y Combinator"):
            title = title.replace(suffix, "")
        return title.strip()
    return path.rsplit("/", 1)[-1].replace("-", " ").title()


def _extract_website(html: str) -> str:
    import re

    m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*website\s*<', html, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*Visit\s*<', html, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _clean_company_name(title: str) -> str:
    for sep in (" - ", " | ", " — ", " hiring", " raises"):
        idx = title.lower().find(sep)
        if idx > 0:
            return title[:idx].strip()
    return title.strip()[:80]


def _extract_domain(url: str) -> str:
    try:
        p = urlparse(url)
        h = p.hostname or ""
        if h.startswith("www."):
            h = h[4:]
        return h
    except Exception:
        return ""
