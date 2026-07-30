"""Job searcher: GitHub internship indexes + web search via Firecrawl SDK."""

import asyncio
import random
import re
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx
from firecrawl import FirecrawlApp

from src.configuration import get_config
from src.llm.context import ContextManager
from src.logging import get_logger
from src.pipeline.queue import JobPipeline, QueuedJob

if TYPE_CHECKING:
    from src.memory.pgvector_store import MemoryStore

logger = get_logger("searcher")

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",  # noqa: E501
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",  # noqa: E501
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",  # noqa: E501
    "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0",  # noqa: E501
]

GITHUB_INDEXES = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/zapplyjobs/Research-Internships-for-Undergraduates/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
    "https://raw.githubusercontent.com/SimplifyJobs/New-Grad-Positions/dev/README.md",
    "https://raw.githubusercontent.com/ReaVNaiL/New-Grad-2026/main/README.md",
]

SEARCH_QUERIES_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 8,
            "maxItems": 8,
        }
    },
    "required": ["queries"],
}

TARGET_POSITIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "roles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 4,
        }
    },
    "required": ["roles"],
}

INDEX_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "listings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "location": {"type": "string"},
                    "apply_link": {"type": "string"},
                    "posted": {"type": ["string", "null"]},
                },
                "required": ["company", "role", "apply_link"],
            },
        }
    },
    "required": ["listings"],
}


async def _scrape_index(url: str, app: FirecrawlApp, pipeline: JobPipeline) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and resp.text:
                await pipeline.push(
                    QueuedJob(markdown=resp.text, url=url, title=f"INDEX:{url.split('/')[-1]}")
                )
                return
    except Exception:
        pass
    try:
        cfg = get_config().firecrawl
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{cfg.url}/v1/scrape",
                json={"url": url, "formats": ["markdown"]},
            )
            if resp.status_code == 200:
                md = (resp.json().get("data") or {}).get("markdown", "") or ""
                if md:
                    await pipeline.push(
                        QueuedJob(markdown=md, url=url, title=f"INDEX:{url.split('/')[-1]}")
                    )
    except Exception as e:
        logger.warning("Index scrape failed", source=url, exception=str(e))


async def scrape_url_to_pipeline(item: dict, app: FirecrawlApp, pipeline: JobPipeline) -> None:
    url = item.get("url", "")
    if not url:
        return
    try:
        cfg = get_config().firecrawl
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(
                f"{cfg.url}/v1/scrape",
                json={"url": url, "formats": ["markdown"]},
            )
            if resp.status_code == 200:
                md = (resp.json().get("data") or {}).get("markdown", "") or ""
                if md and len(md) >= 300:
                    md_lower = md.lower()
                    if not any(kw in md_lower for kw in _JOB_KEYWORDS):
                        return
                    await pipeline.push(
                        QueuedJob(markdown=md, url=url, title=item.get("title", ""))
                    )
    except Exception:
        pass


_JOB_KEYWORDS = (
    "requirements",
    "qualifications",
    "experience",
    "responsibilities",
    "apply",
    "salary",
)


_url_blacklist = (
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "linkedin.com",
    "simplyhired.com",
    "remoteok.com",
    "remoterocketship.com",
    "dailyremote.com",
    "glassdoor.",
)


async def _search_searxng(queries: list[str]) -> list[dict[str, str]]:
    """Query self-hosted SearXNG metasearch engine."""
    sem = asyncio.Semaphore(5)
    hits: list[dict[str, str]] = []
    cfg = get_config().searxng

    async def _query_one(q: str) -> None:
        async with sem:
            try:
                async with httpx.AsyncClient(
                    timeout=cfg.timeout, headers={"User-Agent": random.choice(_USER_AGENTS)}
                ) as client:
                    resp = await client.get(
                        cfg.url,
                        params={"q": q, "format": "json", "time_range": "day"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data.get("results", [])[:5]:
                        url = r.get("url", "")
                        if (
                            url
                            and url.startswith("http")
                            and not any(bad in url.lower() for bad in _url_blacklist)
                        ):
                            hits.append(
                                {
                                    "url": url,
                                    "title": r.get("title", "") or "",
                                    "type": "searxng",
                                }
                            )
            except Exception:
                pass

    await asyncio.gather(*(_query_one(q) for q in queries))
    return hits


async def _search_web(
    app: FirecrawlApp, positions: list[str], ctx: ContextManager
) -> list[dict[str, str]]:
    queries: list[str] = [
        "software engineer intern remote",
        "backend engineer new grad",
        "junior developer remote hiring",
        "entry level software engineer",
    ]
    for pos in positions[:3]:
        queries.append(f"{pos} intern remote")
        queries.append(f"{pos} new grad visa sponsorship")
    if not queries:
        queries = [f"{p} intern remote" for p in positions[:2]]

    sem = asyncio.Semaphore(5)
    cfg = get_config().firecrawl

    async def _fetch_query(q: str) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=6.0) as client:
                    resp = await client.post(
                        f"{cfg.url}/v1/search",
                        json={"query": q},
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", []) or []
                        for r in data[:5]:
                            url = r.get("url", "")
                            if (
                                url
                                and url.startswith("http")
                                and not any(bad in url.lower() for bad in _url_blacklist)
                            ):
                                hits.append(
                                    {
                                        "url": url,
                                        "title": r.get("title", "") or "",
                                        "type": "web_search",
                                    }
                                )
            except Exception:
                pass
        return hits

    firecrawl_task = asyncio.gather(*(_fetch_query(q) for q in queries), return_exceptions=True)
    searxng_task = _search_searxng(queries)

    hit_lists, searxng_hits = await asyncio.gather(firecrawl_task, searxng_task)

    results: list[dict[str, str]] = []
    for hl in hit_lists:
        if isinstance(hl, list):
            results.extend(hl)
    results.extend(searxng_hits)

    logger.info(
        f"Search: {len(results)} URLs from {len(queries)} queries",
        extra={
            "firecrawl_hits": sum(len(hl) for hl in hit_lists if isinstance(hl, list)),
            "searxng_hits": len(searxng_hits),
        },
    )
    return results


async def scrape_all(
    app: FirecrawlApp,
    positions: list[str],
    ctx: ContextManager,
    pipeline: JobPipeline,
    max_workers: int = 12,
) -> None:
    tasks = []

    for url in GITHUB_INDEXES:
        tasks.append(lambda u=url: _scrape_index(u, app, pipeline))

    web_hits = await _search_web(app, positions, ctx)
    for hit in web_hits:
        tasks.append(lambda h=hit: scrape_url_to_pipeline(h, app, pipeline))

    sem = asyncio.Semaphore(max_workers)

    async def _limited_run(coro_factory):
        async with sem:
            return await coro_factory()

    await asyncio.gather(*(_limited_run(t) for t in tasks))


async def fetch_direct_json_feeds(positions: list[str], pipeline: JobPipeline) -> None:
    """Hit free public JSON endpoints: Remotive + Hacker News Algolia."""
    pos_lower = [p.lower() for p in positions]

    async def _fetch_remotive() -> None:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://remotive.com/api/remote-jobs", params={"limit": 50}
                )
                resp.raise_for_status()
                data = resp.json()
                for job in data.get("jobs", []):
                    title = (job.get("title") or "").lower()
                    category = (job.get("category") or "").lower()
                    if any(p in title or p in category for p in pos_lower):
                        _senior_kws = (
                            "senior",
                            "sr.",
                            "staff ",
                            "lead ",
                            "principal",
                            "architect",
                            "manager",
                            "director",
                            "head of",
                        )
                        if any(kw in title for kw in _senior_kws):
                            continue
                        desc = job.get("description", "")
                        clean_md = (
                            f"**{job.get('title', '')}** at {job.get('company_name', '')}\n\n"
                            f"{desc[:5000]}"
                        )
                        await pipeline.push(
                            QueuedJob(
                                markdown=clean_md,
                                url=job.get("url", ""),
                                title=job.get("title", ""),
                            )
                        )
        except Exception as e:
            logger.warning("Remotive feed error", source="remotive", exception=str(e))

    async def _fetch_hn() -> None:
        try:
            cutoff_ts = int(time.time()) - 86400
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "https://hn.algolia.com/api/v1/search_by_date",
                    params={
                        "tags": "job",
                        "query": "remote intern entry junior",
                        "hitsPerPage": 30,
                        "numericFilters": f"created_at_i>{cutoff_ts}",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                for hit in data.get("hits", []):
                    comment = hit.get("comment_text", "")
                    if not comment or len(comment) < 100:
                        continue
                    comment_lower = comment.lower()
                    if any(p in comment_lower for p in pos_lower):
                        title = hit.get("title", "Startup Role")[:50]
                        await pipeline.push(
                            QueuedJob(
                                markdown=comment[:5000],
                                url=(
                                    "https://news.ycombinator.com/item?id="
                                    f"{hit.get('objectID', '')}"
                                ),
                                title=f"HN Who Is Hiring: {title}",
                            )
                        )
        except Exception as e:
            logger.warning("HN Algolia feed error", source="hn_algolia", exception=str(e))

    await asyncio.gather(_fetch_remotive(), _fetch_hn())


async def map_company_careers(
    app: FirecrawlApp, target_domains: list[str], keyword: str = ""
) -> list[dict[str, str]]:
    """Use Firecrawl /map to discover job listing URLs across ATS platforms and career portals."""
    discovered: list[dict[str, str]] = []
    sem = asyncio.Semaphore(24)
    done_lock = asyncio.Lock()
    done_count = 0
    total = len(target_domains)
    cfg = get_config().firecrawl

    non_job_slugs = (
        "/about",
        "/team",
        "/terms",
        "/privacy",
        "/login",
        "/contact",
        "/culture",
        "/blog",
        "/press",
    )

    job_patterns = (
        "/jobs/",
        "/job/",
        "/careers/",
        "/positions/",
        "/openings/",
        "/embed/job",
        "myworkdayjobs.com",
    )

    async def _map_one(domain: str) -> None:
        nonlocal done_count
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                    payload: dict[str, str] = {"url": domain}
                    if keyword:
                        payload["search"] = keyword
                    resp = await client.post(
                        f"{cfg.url}/v1/map",
                        json=payload,
                    )
                    if resp.status_code == 200:
                        links = resp.json().get("links", []) or []
                        if isinstance(links, list):
                            for url in links:
                                if isinstance(url, str) and url.startswith("http"):
                                    u_lower = url.lower()
                                    if any(pat in u_lower for pat in job_patterns) and not any(
                                        bad in u_lower for bad in non_job_slugs
                                    ):
                                        discovered.append(
                                            {"url": url, "title": url.split("/")[-1], "type": "map"}
                                        )
            except Exception as e:
                logger.debug("map error", source=domain, exception=str(e))
            async with done_lock:
                done_count += 1
                if done_count % 5 == 0 or done_count == total:
                    logger.debug(f"Mapping domains... {done_count}/{total}")

    await asyncio.gather(*(_map_one(d) for d in target_domains))
    return discovered


async def extract_index_jobs(jobs: list[QueuedJob], ctx: ContextManager) -> list[dict]:
    extracted: list[dict] = []
    sem = asyncio.Semaphore(8)

    async def _extract_one(job: QueuedJob) -> list[dict]:
        async with sem:
            lines = [
                ln
                for ln in job.markdown.split("\n")
                if "|" in ln and ("remote" in ln.lower() or "---" in ln or "http" in ln.lower())
            ]
            if not lines:
                return []

            sub_listings: list[dict] = []
            chunk_size = 40
            for i in range(0, min(len(lines), 160), chunk_size):
                chunk_lines = lines[i : i + chunk_size]
                clean_md = "\n".join(chunk_lines)
                if len(clean_md) < 50:
                    continue
                prompt = (
                    "Extract ALL job/internship listings from this markdown table. "
                    "You MUST extract the raw http or https URL from any markdown "
                    "reference link (e.g. [Apply](https://url.com) -> https://url.com) "
                    "into apply_link. Do not leave apply_link empty. "
                    "Return valid JSON matching the required schema. "
                    "Be exhaustive — extract every single row/listing."
                )
                raw = await ctx.json_chat(prompt, INDEX_EXTRACT_SCHEMA, clean_md, limit=6000)
                if (
                    isinstance(raw, dict)
                    and "listings" in raw
                    and isinstance(raw["listings"], list)
                ):
                    for item in raw["listings"]:
                        if isinstance(item, dict):
                            link = item.get("apply_link")
                            if not link or not str(link).startswith("http"):
                                item["apply_link"] = job.url
                            sub_listings.append(item)

            return sub_listings

    tasks = [asyncio.create_task(_extract_one(j)) for j in jobs]
    results = await asyncio.gather(*tasks)
    for r in results:
        extracted.extend(r)

    return extracted


_ATS_PATTERN = re.compile(
    r"https?://(?:[a-zA-Z0-9-]+\.)*(?:"
    r"greenhouse\.io/[a-zA-Z0-9_-]+"
    r"|lever\.co/[a-zA-Z0-9_-]+"
    r"|ashbyhq\.com/[a-zA-Z0-9_-]+"
    r"|workable\.com/[a-zA-Z0-9_-]+"
    r"|smartrecruiters\.com/[a-zA-Z0-9_-]+"
    r"|myworkdayjobs\.com/[a-zA-Z0-9_-]+"
    r"|rippling\.com/careers/[a-zA-Z0-9_-]+"
    r"|jobs\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"
    r"|careers\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"
    r")",
    re.IGNORECASE,
)


def extract_career_domain(url: str) -> str | None:
    """Aggressively extracts ATS root or company career portal URL from any job link."""
    if not url or not url.lower().startswith("http"):
        return None
    match = _ATS_PATTERN.search(url)
    if match:
        return match.group(0).rstrip("/")

    try:
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]
        if any(part.lower() in ("jobs", "careers", "about/careers") for part in path_parts):
            return f"{parsed.scheme}://{parsed.netloc}/{path_parts[0]}"
    except Exception:
        pass
    return None


async def harvest_and_save_domains(urls: list[str], store: MemoryStore) -> int:
    """Extract candidate ATS root domains from job URLs and persist to PostgreSQL.

    Returns the count of newly discovered (not previously seen) domains.
    """
    new_count = 0
    for url in urls:
        domain = extract_career_domain(url)
        if domain:
            added = await store.add_discovered_domain(domain, url)
            if added:
                new_count += 1
    return new_count
