"""Pipeline: resume → search → async MQ → concurrent match → verify → output."""

import asyncio
import os
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
        except (ValueError, TypeError):
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
            return  # image-based apply button — not a real JD page
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
            # ── /extract fallback for ATS pages ──
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
            # ── metadata fallback ──
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
    continuous = os.getenv("OVERNIGHT_LOOP", "false").lower() == "true"

    def _cleanup(signum: int, frame: object) -> None:
        console.print("\n[yellow]Interrupted - flushing LLM context...[/yellow]")
        ctx._flush_sync()
        sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    await ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url="http://127.0.0.1:3002")

    console.rule("[bold cyan]PHASE 1: Load Resume + Build RAG Index[/bold cyan]")
    loop = asyncio.get_running_loop()

    def _load_and_build():
        text, chunks = load_resume()
        rag = build_rag_from_chunks(chunks)
        return text, rag

    full_text, rag = await loop.run_in_executor(None, _load_and_build)
    console.print(f"  [green]Indexed {len(rag.doc_texts)} chunks[/green]")

    raw_roles = await ctx.json_chat(
        "Based on this resume, identify the top 2-4 best-fitting entry-level / "
        "intern / new-grad / early-career (NOT senior/staff/lead/principal) "
        "job role domains (e.g. Backend Engineer, Frontend Engineer, Fullstack "
        "Developer, DevOps Engineer, ML Engineer, Data Engineer). "
        "Return valid JSON matching the required schema.\n\n" + full_text[:3000],
        schema=TARGET_POSITIONS_SCHEMA,
    )
    positions: list[str] = raw_roles.get("roles", []) if isinstance(raw_roles, dict) else []
    if not positions:
        positions = ["Software Engineer", "Backend Developer"]
    console.print(f"  [yellow]Target positions:[/yellow] {', '.join(positions)}")

    sweep = 0
    global_verified: dict[str, dict] = {}
    while True:
        sweep += 1
        pipeline = JobPipeline()
        matched_result: list[dict] = []

        console.rule(f"[bold cyan]PHASE 2 (sweep {sweep}): Scrape + Concurrent Match[/bold cyan]")

        consumer_task = asyncio.create_task(_consumer(pipeline, rag, ctx))

        await asyncio.gather(
            scrape_all(app, positions, ctx, pipeline, max_workers=MAX_SCRAPE_WORKERS),
            fetch_direct_json_feeds(positions, pipeline),
        )
        # Discover YC / ATS career page listings via Firecrawl /map
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
                f"[bold cyan]PHASE 2b (sweep {sweep}): Extract Index Jobs + Match[/bold cyan]"
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
                    idx_scored = await batch_match(idx_batch, rag, ctx)
                    matched_result.extend(idx_scored)

        console.rule(f"[bold cyan]PHASE 3 (sweep {sweep}): RAG Revalidation[/bold cyan]")
        validated = await _revalidate_batch(matched_result, rag, ctx)

        console.rule(f"[bold cyan]PHASE 4 (sweep {sweep}): Filter + Cross-Verify[/bold cyan]")
        scored = filter_recent(validated, max_days=7)
        scored.sort(key=lambda j: j["match_percent"], reverse=True)
        scored = scored[:TARGET]

        verified = await verify_jobs(app, scored[:TARGET], ctx, concurrency=VERIFY_CONCURRENCY)

        console.rule(f"[bold cyan]PHASE 5 (sweep {sweep}): Generate Output[/bold cyan]")
        for j in verified:
            dedup_key = (
                j.get("apply_link")
                or j.get("source_url")
                or f"{j.get('role', '')}@{j.get('company', '')}"
            )
            if not dedup_key:
                continue
            existing = global_verified.get(dedup_key)
            if existing is None or j.get("match_percent", 0) > existing.get("match_percent", 0):
                global_verified[dedup_key] = j

        sorted_global = sorted(
            global_verified.values(),
            key=lambda x: x.get("match_percent", 0),
            reverse=True,
        )
        write_md(sorted_global[:40])
        await ctx.flush()

        console.print(
            f"  [cyan]Master Ledger: {len(global_verified)} unique verified"
            f" positions saved to jobs.md[/cyan]"
        )
        console.print(f"\n  [dim]Queue: {pipeline.pending} items remaining[/dim]")
        console.print(f"[bold green]Sweep {sweep} complete[/bold green]")

        if not continuous:
            break

        console.print(
            "\n[bold cyan]Sleeping for 10 minutes before next sweep...[/bold cyan]"
        )
        await asyncio.sleep(600)

    await ctx.aclose()


def run() -> None:
    asyncio.run(_run_pipeline())


if __name__ == "__main__":
    run()
