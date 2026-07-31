"""Wikipedia founder lookup.

SearXNG/Bing rarely surfaces founder pages, so we pull the founders
field straight from a company's Wikipedia infobox. Exact-title first,
then the search API with a company-page preference, then parse the
wikitext ``founders`` field.
"""

from __future__ import annotations

import asyncio
import re

from src.http_client import get_client
from src.logging import get_logger

logger = get_logger("wikipedia")

_API = "https://en.wikipedia.org/w/api.php"
_UA = "Mozilla/5.0 (X11; Linux x86_64) ho-radar/1.0"

# Wikipedia asks for < 1 req/s; the rest of the pipeline calls this from
# concurrent workers, so serialize and pace requests globally.
_WIKI_SEM = asyncio.Semaphore(2)
_WIKI_PACE = asyncio.Lock()
_WIKI_LAST_REQ = 0.0
_WIKI_MIN_INTERVAL = 1.1


async def _wiki_get(client, **params) -> object | None:
    """One paced Wikipedia API request; returns the JSON or None on 429."""
    global _WIKI_LAST_REQ
    async with _WIKI_SEM:
        async with _WIKI_PACE:
            import time

            wait = _WIKI_LAST_REQ + _WIKI_MIN_INTERVAL - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                resp = await client.get(
                    _API, params=params, headers={"User-Agent": _UA}, timeout=15.0
                )
                _WIKI_LAST_REQ = time.time()
            except Exception:
                return None
        if resp.status_code == 429:
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except Exception:
            return None


# wikitext infobox: | founders = A, B or {{plainlist|* A * B}} etc.
_INFOBOX_RE = re.compile(
    r"\|\s*founders?\s*=\s*(.+?)(?=\n\s*\||\n\s*\}\})",
    re.IGNORECASE | re.S,
)
_TEMPLATE_SPLIT_RE = re.compile(r"<br\s*/?>|\n|;|{{[^}]*}}|\*\s*|\|")
_NAME_CLEAN_RE = re.compile(r"[\[\]{}|]")


def _clean_name(raw: str) -> str:
    name = _NAME_CLEAN_RE.sub("", raw).strip()
    name = re.sub(r"\(.*?\)", "", name).strip()
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _parse_founders(wikitext: str) -> list[str]:
    m = _INFOBOX_RE.search(wikitext)
    if not m:
        return []
    body = m.group(1)
    body = re.sub(r"{{plainlist\|", "", body, flags=re.IGNORECASE)
    body = re.sub(r"{{unbulleted list\|", "", body, flags=re.IGNORECASE)
    parts = [p for p in _TEMPLATE_SPLIT_RE.split(body) if p.strip()]
    names: list[str] = []
    for p in parts:
        cleaned = _clean_name(p)
        if not cleaned or cleaned.lower() in ("founders", "founder"):
            continue
        if cleaned not in names and len(names) < 8:
            names.append(cleaned)
    return names


async def _fetch_wikitext(title: str) -> str | None:
    try:
        client = await get_client("wikipedia", timeout=15.0)
        data = await _wiki_get(
            client,
            action="query",
            prop="revisions",
            rvprop="content",
            rvslots="main",
            format="json",
            formatversion="2",
            titles=title,
            redirects="1",
        )
        if data is None:
            return None
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if "missing" in page or "revisions" not in page:
            return None
        return page.get("revisions", [{}])[0].get("slots", {}).get("main", {}).get("content", "")
    except Exception as e:
        logger.warning("Wikipedia wikitext fetch failed", title=title, exception=str(e))
        return None


async def _resolve_title(company: str) -> str | None:
    """Find the Wikipedia page title for a company."""
    try:
        client = await get_client("wikipedia", timeout=15.0)
        for candidate in (company, f"{company} (company)", f"{company} (software)"):
            data = await _wiki_get(
                client,
                action="query",
                prop="info",
                format="json",
                formatversion="2",
                titles=candidate,
                redirects="1",
            )
            if data is None:
                continue
            pages = data.get("query", {}).get("pages", [])
            if pages and "missing" not in pages[0]:
                return pages[0]["title"]
        # Search fallback: prefer an exact-ish title mentioning the company.
        data = await _wiki_get(
            client,
            action="query",
            list="search",
            srsearch=f'"{company}" company',
            srlimit="5",
            format="json",
            formatversion="2",
        )
        if data is not None:
            hits = data.get("query", {}).get("search", [])
            for h in hits:
                title = h.get("title", "")
                if title.lower().startswith(company.lower().split()[0]):
                    return title
    except Exception as e:
        logger.warning("Wikipedia title resolution failed", company=company, exception=str(e))
    return None


async def get_wikipedia_founders(company: str) -> list[dict[str, str]]:
    """Return founder dicts (name, title, linkedin_url/github_url/email None).

    Empty list when Wikipedia has no page or no founders field.
    """
    if not company:
        return []
    title = await _resolve_title(company)
    if not title:
        return []
    wikitext = await _fetch_wikitext(title)
    if not wikitext:
        return []
    names = _parse_founders(wikitext)
    return [
        {"name": n, "title": "Founder", "linkedin_url": None, "github_url": None, "email": None}
        for n in names
    ]
