"""Job searcher: GitHub internship indexes + web search via Firecrawl SDK."""

import asyncio

import httpx
from firecrawl import FirecrawlApp

from src.llm.context import ContextManager
from src.pipeline.queue import JobPipeline, QueuedJob

GITHUB_INDEXES = [
    "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md",
    "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/main/README.md",
    "https://raw.githubusercontent.com/LorenzoLaCorte/european-tech-internships-2026/main/README.md",
    "https://raw.githubusercontent.com/zapplyjobs/Research-Internships-for-Undergraduates/main/README.md",
    "https://raw.githubusercontent.com/DereC4/internships-and-newgrad/main/README.md",
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
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, lambda: app.scrape_url(url, formats=["markdown"]))
        md = getattr(result, "markdown", "") or ""
        if md:
            await pipeline.push(
                QueuedJob(markdown=md, url=url, title=f"INDEX:{url.split('/')[-1]}")
            )
    except Exception as e:
        print(f"  [red]index fail {url}: {e}[/red]")


async def _scrape_url(item: dict, app: FirecrawlApp, pipeline: JobPipeline) -> None:
    loop = asyncio.get_running_loop()
    url = item.get("url", "")
    if not url:
        return
    try:
        result = await loop.run_in_executor(None, lambda: app.scrape_url(url, formats=["markdown"]))
        md = getattr(result, "markdown", "") or ""
        if md and len(md) > 100:
            await pipeline.push(QueuedJob(markdown=md, url=url, title=item.get("title", "")))
    except Exception:
        pass


_url_blacklist = (
    "indeed.com", "glassdoor.com", "ziprecruiter.com", "linkedin.com",
    "simplyhired.com", "remoteok.com", "remoterocketship.com",
    "dailyremote.com", "glassdoor.",
)


async def _search_searxng(queries: list[str]) -> list[dict[str, str]]:
    """Query self-hosted SearXNG metasearch engine."""
    sem = asyncio.Semaphore(3)
    hits: list[dict[str, str]] = []

    async def _query_one(q: str) -> None:
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(
                        "http://localhost:8080/search",
                        params={"q": q, "format": "json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    for r in data.get("results", [])[:5]:
                        url = r.get("url", "")
                        if url and url.startswith("http") and not any(
                            bad in url.lower() for bad in _url_blacklist
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

    await asyncio.gather(*(_query_one(q) for q in queries[:8]))
    return hits


async def _search_web(
    app: FirecrawlApp, positions: list[str], ctx: ContextManager
) -> list[dict[str, str]]:
    loop = asyncio.get_running_loop()

    ats_boards = [
        "site:greenhouse.io", "site:lever.co",
        "site:ashbyhq.com", "site:jobs.workable.com",
    ]
    queries: list[str] = []
    for pos in positions[:2]:
        for board in ats_boards:
            queries.append(f'{board} "remote" "{pos}"')
    if not queries:
        queries = [f"{p} intern remote" for p in positions[:2]]

    sem = asyncio.Semaphore(3)

    async def _fetch_query(q: str) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        async with sem:
            try:
                search_results = await loop.run_in_executor(None, app.search, q)
                data = getattr(search_results, "web", []) or []
                if isinstance(data, list):
                    for r in data[:5]:
                        url = getattr(r, "url", "")
                        if (
                            url
                            and url.startswith("http")
                            and not any(bad in url.lower() for bad in _url_blacklist)
                        ):
                            hits.append(
                                {
                                    "url": url,
                                    "title": getattr(r, "title", "") or "",
                                    "type": "web_search",
                                }
                            )
            except Exception:
                pass
        return hits

    firecrawl_task = asyncio.gather(
        *(_fetch_query(q) for q in queries[:8]), return_exceptions=True
    )
    searxng_task = _search_searxng(queries[:8])

    hit_lists, searxng_hits = await asyncio.gather(firecrawl_task, searxng_task)

    results: list[dict[str, str]] = []
    for hl in hit_lists:
        if isinstance(hl, list):
            results.extend(hl)
    results.extend(searxng_hits)

    print(
        f"  Search: {len(results)} URLs "
        f"(Firecrawl + {len(searxng_hits)} SearXNG) from {len(queries)} queries"
    )
    return results


async def scrape_all(
    app: FirecrawlApp,
    positions: list[str],
    ctx: ContextManager,
    pipeline: JobPipeline,
    max_workers: int = 6,
) -> None:
    tasks = []

    for url in GITHUB_INDEXES:
        tasks.append(_scrape_index(url, app, pipeline))

    web_hits = await _search_web(app, positions, ctx)
    for hit in web_hits:
        tasks.append(_scrape_url(hit, app, pipeline))

    sem = asyncio.Semaphore(max_workers)

    async def _limited_run(coro):
        async with sem:
            return await coro

    await asyncio.gather(*(_limited_run(t) for t in tasks))


async def fetch_direct_json_feeds(
    positions: list[str], pipeline: JobPipeline
) -> None:
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
            print(f"  [dim]Remotive feed error: {e}[/dim]")

    async def _fetch_hn() -> None:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    "http://hn.algolia.com/api/v1/search_by_date",
                    params={"tags": "job", "query": "remote", "hitsPerPage": 30},
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
            print(f"  [dim]HN Algolia feed error: {e}[/dim]")

    await asyncio.gather(_fetch_remotive(), _fetch_hn())


async def map_company_careers(
    app: FirecrawlApp, target_domains: list[str], keyword: str = "remote"
) -> list[dict[str, str]]:
    """Use Firecrawl /map to discover job listing URLs on ATS platforms."""
    loop = asyncio.get_running_loop()
    discovered: list[dict[str, str]] = []

    async def _map_one(domain: str) -> None:
        try:
            result = await loop.run_in_executor(
                None, lambda: app.map_url(domain, search=keyword)
            )
            links = getattr(result, "links", []) or []
            if isinstance(links, list):
                for url in links:
                    if isinstance(url, str) and url.startswith("http") and "/jobs/" in url.lower():
                        discovered.append({"url": url, "title": url.split("/")[-1], "type": "map"})
        except Exception as e:
            print(f"  [dim]Map {domain}: {e}[/dim]")

    await asyncio.gather(*(_map_one(d) for d in target_domains[:4]))
    return discovered


async def extract_index_jobs(jobs: list[QueuedJob], ctx: ContextManager) -> list[dict]:
    extracted: list[dict] = []
    sem = asyncio.Semaphore(2)

    async def _extract_one(job: QueuedJob) -> list[dict]:
        async with sem:
            lines = [
                ln
                for ln in job.markdown.split("\n")
                if "|" in ln and ("remote" in ln.lower() or "---" in ln)
            ]
            clean_md = "\n".join(lines[:100])
            if len(clean_md) < 50:
                return []
            prompt = (
                "Extract ALL job/internship listings from this markdown table. "
                "You MUST extract the raw http or https URL from any markdown "
                "reference link (e.g. [Apply](https://url.com) -> https://url.com) "
                "into apply_link. Do not leave apply_link empty. "
                "Return valid JSON matching the required schema. "
                "Be exhaustive — extract every single row/listing."
            )
            raw = await ctx.json_chat(prompt, INDEX_EXTRACT_SCHEMA, clean_md, limit=12000)
            if isinstance(raw, dict) and "listings" in raw:
                return raw["listings"]
            return []

    tasks = [asyncio.create_task(_extract_one(j)) for j in jobs]
    results = await asyncio.gather(*tasks)
    for r in results:
        extracted.extend(r)

    return extracted
