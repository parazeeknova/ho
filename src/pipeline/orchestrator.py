"""Pipeline: resume → search → async MQ → graph pipeline → verify → output."""

import asyncio
import os
import signal
import sys
from datetime import UTC, datetime

import httpx
from firecrawl import FirecrawlApp
from rich.console import Console

from src.llm.context import ContextManager
from src.matching.verifier import verify_jobs
from src.memory.pgvector_store import MemoryStore
from src.output.writer import write_md
from src.pipeline.graph import EMBED_URL, run_batch
from src.pipeline.queue import JobPipeline, QueuedJob
from src.rag.loader import load_resume
from src.search.searcher import (
    TARGET_POSITIONS_SCHEMA,
    extract_index_jobs,
    fetch_direct_json_feeds,
    map_company_careers,
    scrape_all,
    scrape_url_to_pipeline,
)

console = Console()

TARGET = 15
MAX_SCRAPE_WORKERS = 6
MATCH_CONCURRENCY = 4
VERIFY_CONCURRENCY = 4


def filter_recent(jobs: list[dict], max_days: int = 7) -> list[dict]:
    filtered = []
    now = datetime.now(UTC)
    for j in jobs:
        date_str = j.get("posted_date") or j.get("posted")
        if not date_str:
            filtered.append(j)
            continue
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            if (now - dt).days <= max_days:
                filtered.append(j)
        except ValueError, TypeError:
            filtered.append(j)
    return filtered


async def _scrape_index_links(
    app: FirecrawlApp, jobs: list[dict], max_workers: int = 4
) -> list[dict]:
    """Scrape apply-link URLs from extracted index jobs to get real JD markdown."""
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_workers)
    scraped: list[dict] = []

    _dead_page_texts = (
        "sorry, we couldn't find anything here",
        "job not found",
        "this position has been filled",
    )

    def _md_fallback(j: dict) -> str:
        return (
            f"Position: {j.get('role', '')} at {j.get('company', '')}. "
            f"Apply Link: {j.get('apply_link', '')}"
        )

    _image_hosts = ("i.imgur.com", "imgur.com/", ".png", ".jpg", ".jpeg", ".gif", ".webp")

    async def _scrape_one(j: dict) -> None:
        url = j.get("apply_link", "")
        if not url or not url.startswith("http"):
            return
        url_lower = url.lower()
        if any(h in url_lower for h in _image_hosts):
            return
        async with sem:
            try:
                result = await loop.run_in_executor(
                    None, lambda: app.scrape_url(url, formats=["markdown"])
                )
                md = getattr(result, "markdown", "") or ""
                md_lower = md.lower()
                if md and len(md) > 100 and not any(dead in md_lower for dead in _dead_page_texts):
                    scraped.append(
                        {
                            "markdown": md,
                            "url": url,
                            "title": j.get("role", ""),
                            "snippet": str(j),
                        }
                    )
                    return
            except Exception as e:
                print(f"    [dim]Failed to scrape {url[:40]}...: {e} -> Trying /extract API[/dim]")
            try:
                extract_result = await loop.run_in_executor(
                    None,
                    lambda: app.extract(
                        [url],
                        {
                            "prompt": (
                                "Extract job title, company name, required technical "
                                "skills, and full job description."
                            ),
                        },
                    ),
                )
                if extract_result and getattr(extract_result, "data", None):
                    md = str(extract_result.data)
                    if len(md) > 100:
                        scraped.append(
                            {
                                "markdown": md,
                                "url": url,
                                "title": j.get("role", ""),
                                "snippet": str(j),
                            }
                        )
                        return
            except Exception:
                pass
            print(f"    [dim]Using table metadata fallback for {url[:40]}...[/dim]")
            scraped.append(
                {
                    "markdown": _md_fallback(j),
                    "url": url,
                    "title": j.get("role", ""),
                    "snippet": str(j),
                }
            )

    tasks = [asyncio.create_task(_scrape_one(j)) for j in jobs]
    await asyncio.gather(*tasks)
    return scraped


async def _consumer(
    pipeline: JobPipeline,
    store: MemoryStore,
) -> tuple[list[dict], list[QueuedJob]]:
    matched: list[dict] = []
    index_queue: list[QueuedJob] = []
    web_buf: list[QueuedJob] = []

    while True:
        job = await pipeline.pop(timeout=2)
        if job is None:
            if pipeline.is_done:
                break
            continue

        if job.title.startswith("INDEX:"):
            if len(job.markdown) > 500:
                index_queue.append(job)
        else:
            web_buf.append(job)

        if len(web_buf) >= 5:
            batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
            scored = await run_batch(batch, store, concurrency=MATCH_CONCURRENCY)
            matched.extend(scored)
            matched.sort(key=lambda j: j.get("match_percent", 0), reverse=True)
            write_md(matched[:30], output_path="jobs.md")
            for _ in web_buf:
                await pipeline.task_done()
            web_buf.clear()

        console.print(f"  [{pipeline.log_status()}]")

    if web_buf:
        batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
        scored = await run_batch(batch, store, concurrency=MATCH_CONCURRENCY)
        matched.extend(scored)
        matched.sort(key=lambda j: j.get("match_percent", 0), reverse=True)
        write_md(matched[:30], output_path="jobs.md")
        for _ in web_buf:
            await pipeline.task_done()

    return matched, index_queue


async def _index_resume_in_pgvector(
    chunks: dict[str, str],
    store: MemoryStore,
) -> None:
    """Fetch embeddings from the embedding server and index resume chunks in pgvector."""
    embed_client = httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
    )
    try:
        records: list[dict[str, object]] = []
        for section, text in chunks.items():
            raw_lines = [ln.strip() for ln in text.split("\n")]
            lines = [ln for ln in raw_lines if ln and len(ln) > 10]
            for i in range(0, len(lines), 8):
                batch = lines[i : i + 8]
                resp = await embed_client.post(
                    f"{EMBED_URL}/v1/embeddings",
                    json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": batch},
                )
                resp.raise_for_status()
                data = resp.json()
                for item, content in zip(data["data"], batch, strict=True):
                    records.append(
                        {
                            "section": section,
                            "content": content,
                            "embedding": item["embedding"],
                        }
                    )
            if len(text) > 20:
                resp = await embed_client.post(
                    f"{EMBED_URL}/v1/embeddings",
                    json={"model": "Qwen/Qwen3-Embedding-0.6B", "input": [text[:500]]},
                )
                resp.raise_for_status()
                data = resp.json()
                records.append(
                    {
                        "section": section,
                        "content": text[:500],
                        "embedding": data["data"][0]["embedding"],
                    }
                )
        if records:
            await store.clear_embeddings()
            await store.index_resume_chunks(records)
        console.print(f"  [green]Indexed {len(records)} resume chunks in pgvector[/green]")
    finally:
        await embed_client.aclose()


async def _run_pipeline() -> None:
    ctx = ContextManager()
    continuous = os.getenv("OVERNIGHT_LOOP", "false").lower() == "true"

    def _cleanup(signum: int, frame: object) -> None:
        console.print("\n[yellow]Interrupted - flushing LLM context...[/yellow]")
        ctx._flush_sync()
        sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    await ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url="http://127.0.0.1:3002")

    console.rule("[bold cyan]PHASE 0: Initialise Agent Memory (pgvector)[/bold cyan]")
    store = await MemoryStore.create()
    console.print("  [green]Connected to agent-memory-db[/green]")

    console.rule("[bold cyan]PHASE 1: Load Resume + Index in pgvector[/bold cyan]")
    loop = asyncio.get_running_loop()

    existing_count = await store.chunk_count()
    chunks: dict[str, str] = {}
    full_text = ""
    reuse_resume = False

    if existing_count > 0:
        import questionary

        def _ask_reuse() -> str | None:
            return questionary.select(
                f"Found {existing_count} existing resume chunks in pgvector. What now?",
                choices=[
                    "Reuse existing resume (skip re-indexing)",
                    "Load new resume URL (re-index from scratch)",
                ],
            ).ask()

        choice = await loop.run_in_executor(None, _ask_reuse)
        if choice and choice.startswith("Reuse"):
            reuse_resume = True
            console.print(f"  [green]Reusing {existing_count} existing resume chunks[/green]")

    if not reuse_resume:

        def _load():
            return load_resume()

        full_text, chunks = await loop.run_in_executor(None, _load)
        await _index_resume_in_pgvector(chunks, store)

    positions: list[str]
    if reuse_resume and not full_text:
        positions = ["Software Engineer", "Backend Developer"]
        console.print(f"  [yellow]Target positions (cached):[/yellow] {', '.join(positions)}")
    else:
        raw_roles = await ctx.json_chat(
            "Based on this resume, identify the top 2-4 best-fitting entry-level / "
            "intern / new-grad / early-career (NOT senior/staff/lead/principal) "
            "job role domains (e.g. Backend Engineer, Frontend Engineer, Fullstack "
            "Developer, DevOps Engineer, ML Engineer, Data Engineer). "
            "Return valid JSON matching the required schema.\n\n" + full_text[:3000],
            schema=TARGET_POSITIONS_SCHEMA,
        )
        positions = raw_roles.get("roles", []) if isinstance(raw_roles, dict) else []
        if not positions:
            positions = ["Software Engineer", "Backend Developer"]
        console.print(f"  [yellow]Target positions:[/yellow] {', '.join(positions)}")

    sweep = 0
    while True:
        sweep += 1
        pipeline = JobPipeline()

        console.rule(f"[bold cyan]PHASE 2 (sweep {sweep}): Scrape + Graph Match[/bold cyan]")

        consumer_task = asyncio.create_task(_consumer(pipeline, store))

        await asyncio.gather(
            scrape_all(app, positions, ctx, pipeline, max_workers=MAX_SCRAPE_WORKERS),
            fetch_direct_json_feeds(positions, pipeline),
        )
        _map_domains = [
            "https://www.ycombinator.com/jobs",
            "https://jobs.lever.co",
            "https://boards.greenhouse.io",
            "https://jobs.ashbyhq.com",
        ]
        map_urls = await map_company_careers(app, _map_domains, keyword="software intern")
        for mu in map_urls:
            await scrape_url_to_pipeline(mu, app, pipeline)
        console.print(f"  [cyan]Map discovery: {len(map_urls)} career-page URLs[/cyan]")

        console.print("  [yellow]Producers done. Signalling stop...[/yellow]")
        pipeline.signal_done()

        matched_result, index_queue = await consumer_task

        if index_queue:
            console.rule(
                f"[bold cyan]PHASE 2b (sweep {sweep}): Extract Index Jobs + Graph Match[/bold cyan]"
            )
            index_jobs = await extract_index_jobs(index_queue, ctx)
            console.print(f"  [cyan]Extracted {len(index_jobs)} jobs from indexes[/cyan]")
            if index_jobs:
                console.print(
                    f"  [cyan]Scraping {len(index_jobs)} GitHub apply links for JD text...[/cyan]"
                )
                idx_batch = await _scrape_index_links(
                    app, index_jobs[:30], max_workers=MAX_SCRAPE_WORKERS
                )
                console.print(f"  [cyan]Scraped {len(idx_batch)} JDs from links[/cyan]")
                if idx_batch:
                    idx_scored = await run_batch(idx_batch, store, concurrency=MATCH_CONCURRENCY)
                    matched_result.extend(idx_scored)

        console.rule(f"[bold cyan]PHASE 3 (sweep {sweep}): Filter + Cross-Verify[/bold cyan]")
        scored = filter_recent(matched_result, max_days=7)
        scored.sort(key=lambda j: j["match_percent"], reverse=True)
        scored = scored[:TARGET]

        verified = await verify_jobs(app, scored[:TARGET], ctx, concurrency=VERIFY_CONCURRENCY)

        console.rule(f"[bold cyan]PHASE 4 (sweep {sweep}): Generate Output[/bold cyan]")
        write_md(verified[:40])
        await ctx.flush()

        console.print(f"  [cyan]{len(verified)} verified positions saved to jobs.md[/cyan]")
        console.print(f"[bold green]Sweep {sweep} complete[/bold green]")

        if not continuous:
            break

        console.print("\n[bold cyan]Sleeping for 10 minutes before next sweep...[/bold cyan]")
        await asyncio.sleep(600)

    await ctx.aclose()
    await store.close()


def run() -> None:
    asyncio.run(_run_pipeline())


if __name__ == "__main__":
    run()
