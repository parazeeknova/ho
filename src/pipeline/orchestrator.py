"""Pipeline: resume → search → async MQ → concurrent match → verify → output."""

import asyncio
import signal
import sys
from datetime import UTC, datetime

from firecrawl import FirecrawlApp
from rich.console import Console

from src.llm.context import REVALIDATE_SCHEMA, ContextManager
from src.matching.matcher import batch_match
from src.matching.verifier import verify_jobs
from src.output.writer import write_md
from src.pipeline.queue import JobPipeline, QueuedJob
from src.rag.engine import build_rag_from_chunks
from src.rag.loader import load_resume
from src.search.searcher import TARGET_POSITIONS_SCHEMA, extract_index_jobs, scrape_all

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


async def _revalidate_batch(
    jobs: list[dict],
    rag,
    ctx: ContextManager,
    concurrency: int = 4,
) -> list[dict]:
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)

    async def _revalidate_one(job: dict) -> dict | None:
        async with sem:
            skills = job.get("matching_skills", []) + job.get("missing_skills", [])
            query = " ".join(skills) if skills else str(job.get("role", ""))
            chunks = await loop.run_in_executor(None, rag.retrieve, query, 5)
            if not chunks or chunks[0][2] < 0.15:
                return None

            prompt = (
                "Cross-check this job extraction against the candidate's relevant "
                "resume snippets. Confirm or correct: role, company, match_percent, "
                "shortlist_probability. Set match_percent to 0 if clearly wrong.\n\n"
                f"Current extraction: {str(job)[:2000]}\n\n"
                "Relevant resume:\n"
                + "\n".join(f"[{c[0]}] {c[1]}" for c in chunks)
                + "\n\nReturn valid JSON matching the required schema."
            )

            result = await ctx.json_chat(prompt, REVALIDATE_SCHEMA)
            if isinstance(result, dict) and "match_percent" in result:
                result["match_percent"] = int(result["match_percent"])
                result["shortlist_probability"] = int(result.get("shortlist_probability", 0))
                result["_revalidated"] = True
                result["source_url"] = job.get("source_url", "")
                result["apply_link"] = job.get("apply_link", "")
                return result
            return job

    tasks = [asyncio.create_task(_revalidate_one(j)) for j in jobs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    validated: list[dict] = []
    for j, result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException):
            validated.append(j)
            continue
        if result is not None and result.get("match_percent", 0) >= 30:
            validated.append(result)
            pct = result.get("match_percent", "?")
            role = result.get("role", "?")
            company = result.get("company", "?")
            tag = "[reval]" if result.get("_revalidated") else "[kept]"
            console.print(f"  [green]{tag} {pct}% | {role} @ {company}[/green]")

    dropped = len(jobs) - len(validated)
    console.print(f"  [green]Kept {len(validated)}[/green], [red]dropped {dropped}[/red]")
    return validated


async def _consumer(
    pipeline: JobPipeline,
    rag,
    ctx: ContextManager,
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
            scored = await batch_match(batch, rag, ctx)
            matched.extend(scored)
            matched.sort(key=lambda j: j.get("match_percent", 0), reverse=True)
            write_md(matched[:30], output_path="jobs.md")
            for _ in web_buf:
                await pipeline.task_done()
            web_buf.clear()

        console.print(f"  [{pipeline.log_status()}]")

    if web_buf:
        batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
        scored = await batch_match(batch, rag, ctx)
        matched.extend(scored)
        matched.sort(key=lambda j: j.get("match_percent", 0), reverse=True)
        write_md(matched[:30], output_path="jobs.md")
        for _ in web_buf:
            await pipeline.task_done()

    return matched, index_queue


async def _run_pipeline() -> None:
    ctx = ContextManager()

    def _cleanup(signum: int, frame: object) -> None:
        console.print("\n[yellow]Interrupted - flushing LLM context...[/yellow]")
        ctx._flush_sync()
        sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    await ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url="http://127.0.0.1:3002")
    pipeline = JobPipeline()

    console.rule("[bold cyan]PHASE 1: Load Resume + Build RAG Index[/bold cyan]")
    loop = asyncio.get_running_loop()

    def _load_and_build():
        text, chunks = load_resume()
        rag = build_rag_from_chunks(chunks)
        return text, rag

    full_text, rag = await loop.run_in_executor(None, _load_and_build)
    console.print(f"  [green]Indexed {len(rag.doc_texts)} chunks[/green]")

    console.rule("[bold cyan]PHASE 2: Scrape + Concurrent Match[/bold cyan]")

    raw_roles = await ctx.json_chat(
        "Based on this resume, identify the top 2-4 best-fitting entry-level / "
        "intern / new-grad job role domains (e.g. Backend Engineer, Frontend "
        "Engineer, Fullstack Developer, DevOps Engineer, ML Engineer, Data "
        "Engineer). Return valid JSON matching the required schema.\n\n" + full_text[:3000],
        schema=TARGET_POSITIONS_SCHEMA,
    )
    positions: list[str] = raw_roles.get("roles", []) if isinstance(raw_roles, dict) else []
    if not positions:
        positions = ["Software Engineer", "Backend Developer"]
    console.print(f"  [yellow]Target positions:[/yellow] {', '.join(positions)}")

    consumer_task = asyncio.create_task(_consumer(pipeline, rag, ctx))

    await scrape_all(app, positions, ctx, pipeline, max_workers=MAX_SCRAPE_WORKERS)

    console.print("  [yellow]Producers done. Signalling stop...[/yellow]")
    pipeline.signal_done()

    matched_result, index_queue = await consumer_task

    if index_queue:
        console.rule("[bold cyan]PHASE 2b: Extract Index Jobs + Match[/bold cyan]")
        index_jobs = await extract_index_jobs(index_queue, ctx)
        console.print(f"  [cyan]Extracted {len(index_jobs)} jobs from indexes[/cyan]")
        if index_jobs:
            idx_batch = [
                {
                    "markdown": "",
                    "url": j.get("apply_link", ""),
                    "title": j.get("role", ""),
                    "snippet": str(j),
                }
                for j in index_jobs[:80]
            ]
            idx_scored = await batch_match(idx_batch, rag, ctx)
            matched_result.extend(idx_scored)

    console.rule("[bold cyan]PHASE 3: RAG Revalidation[/bold cyan]")
    validated = await _revalidate_batch(matched_result, rag, ctx)

    console.rule("[bold cyan]PHASE 4: Filter + Cross-Verify[/bold cyan]")
    scored = filter_recent(validated, max_days=7)
    scored.sort(key=lambda j: j["match_percent"], reverse=True)
    scored = scored[:TARGET]

    verified = await verify_jobs(app, scored[:TARGET], ctx, concurrency=VERIFY_CONCURRENCY)

    console.rule("[bold cyan]PHASE 5: Generate Output[/bold cyan]")
    write_md(verified)
    await ctx.flush()
    await ctx.aclose()

    console.print(f"\n  [dim]Queue: {pipeline.pending} items remaining[/dim]")
    console.print("\n[bold green]Pipeline complete[/bold green]")


def run() -> None:
    asyncio.run(_run_pipeline())


if __name__ == "__main__":
    run()
