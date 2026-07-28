"""Job searcher: GitHub internship indexes + web search via Firecrawl SDK."""

import asyncio

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
                "required": ["company", "role"],
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


async def _search_web(
    app: FirecrawlApp, positions: list[str], ctx: ContextManager
) -> list[dict[str, str]]:
    loop = asyncio.get_running_loop()

    domains = ", ".join(positions)
    query_prompt = (
        f"Generate 8 natural-language search queries to find entry-level/"
        f"intern/new-grad remote jobs across these target domains: {domains}. "
        f"Distribute the 8 queries evenly among the domains. Target easy-to-scrape "
        f"job boards: Greenhouse, Lever, Ashby, Remotive, RemoteOK, Wellfound, "
        f"GitHub READMEs. Avoid indeed, glassdoor, ziprecruiter, upwork. "
        f"Return valid JSON matching the required schema."
    )

    raw = await ctx.json_chat(query_prompt, SEARCH_QUERIES_SCHEMA)
    queries: list[str] = raw.get("queries", []) if isinstance(raw, dict) else []
    if not queries:
        queries = [
            f"{p} intern remote" for p in positions[:4]
        ] + [
            f"entry level {p} remote" for p in positions[:4]
        ]
        queries = queries[:8]

    results: list[dict[str, str]] = []
    for q in queries[:8]:
        try:
            search_results = await loop.run_in_executor(None, app.search, q)
            data = getattr(search_results, "web", []) or []
            if isinstance(data, list):
                for r in data[:5]:
                    url = getattr(r, "url", "")
                    if url and url.startswith("http"):
                        results.append(
                            {
                                "url": url,
                                "title": getattr(r, "title", "") or "",
                                "type": "web_search",
                            }
                        )
            await asyncio.sleep(0.3)
        except Exception:
            pass

    print(f"  Web search: {len(results)} URLs from {len(queries)} queries")
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


async def extract_index_jobs(jobs: list[QueuedJob], ctx: ContextManager) -> list[dict]:
    extracted: list[dict] = []
    sem = asyncio.Semaphore(2)

    async def _extract_one(job: QueuedJob) -> list[dict]:
        async with sem:
            prompt = (
                "Extract ALL job/internship listings from this markdown. "
                "Return valid JSON matching the required schema. "
                "Be exhaustive — extract every single row/listing."
            )
            raw = await ctx.json_chat(prompt, INDEX_EXTRACT_SCHEMA, job.markdown, limit=20000)
            if isinstance(raw, dict) and "listings" in raw:
                return raw["listings"]
            return []

    tasks = [asyncio.create_task(_extract_one(j)) for j in jobs]
    results = await asyncio.gather(*tasks)
    for r in results:
        extracted.extend(r)

    return extracted
