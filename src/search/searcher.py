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

    _url_blacklist = (
        "indeed.com", "glassdoor.com", "ziprecruiter.com", "linkedin.com",
        "simplyhired.com", "remoteok.com", "remoterocketship.com",
        "dailyremote.com", "glassdoor.",
    )

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

    hit_lists = await asyncio.gather(
        *(_fetch_query(q) for q in queries[:8]), return_exceptions=True
    )

    results: list[dict[str, str]] = []
    for hl in hit_lists:
        if isinstance(hl, list):
            results.extend(hl)

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
