"""Pipeline: resume → search → async MQ → graph pipeline → verify → output."""

import asyncio
import gc
import os
import re
import signal
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta

import httpx
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from rich.console import Console

from src.agent.cleanup_agent import CleanupAgent
from src.agent.enrichment_agent import EnrichmentAgent
from src.agent.jobs_agent import JobsAgent
from src.agent.startup_agent import StartupAgent
from src.agent.telegram_agent import TelegramAgent, set_pipeline_state
from src.llm.context import ContextManager
from src.memory.pgvector_store import MemoryStore
from src.pipeline.graph import EMBED_URL, drain_retry_queue, run_batch
from src.pipeline.queue import JobPipeline, QueuedJob
from src.rag.github_linkedin_loader import enrich_candidate_chunks
from src.rag.loader import load_resume
from src.search.linkedin_guest import scrape_linkedin_guest_jobs
from src.search.searcher import (
    TARGET_POSITIONS_SCHEMA,
    extract_index_jobs,
    fetch_direct_json_feeds,
    harvest_and_save_domains,
    map_company_careers,
    scrape_all,
    scrape_url_to_pipeline,
)

console = Console()

TARGET = 15
MAX_SCRAPE_WORKERS = 18
MATCH_CONCURRENCY = 24
VERIFY_CONCURRENCY = 20


def filter_last_24_hours(jobs: list[dict]) -> list[dict]:
    """Drop any job with a posted_date provably older than 24 hours.

    Null / unparseable dates get checked for relative-date phrases
    (e.g. '3d ago', 'last week') before being admitted.
    """
    _old_date_rx = re.compile(
        r"(?:(?:(\d+)\s*(?:day|d)\b)|"
        r"(?:(\d+)\s*(?:week|w)\b)|"
        r"(?:(\d+)\s*(?:month|mo|m)\b)|"
        r"(?:(?:last|previous)\s+(?:week|month|year)))",
        re.IGNORECASE,
    )
    _recent_rx = re.compile(
        r"\b(?:today|now|just|hour|mins?|minutes?|seconds?|recently)\b",
        re.IGNORECASE,
    )
    _one_day_rx = re.compile(r"\b(?:1\s*(?:day|d)\s*ago)\b", re.IGNORECASE)

    filtered = []
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    for j in jobs:
        date_str = j.get("posted_date") or j.get("posted")
        if not date_str:
            filtered.append(j)
            continue
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            if dt >= cutoff:
                filtered.append(j)
        except ValueError, TypeError:
            d_lower = str(date_str).lower()
            if (
                _recent_rx.search(d_lower)
                or _one_day_rx.search(d_lower)
                or not _old_date_rx.search(d_lower)
            ):
                filtered.append(j)
    return filtered


async def _limited_scrape(
    item: dict,
    app: FirecrawlApp,
    pipeline: JobPipeline,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        await scrape_url_to_pipeline(item, app, pipeline)


async def _domain_producer(
    app: FirecrawlApp,
    pipeline: JobPipeline,
    store: MemoryStore,
    combined_domains: list[str],
    uncrawled_dynamic: list[str],
) -> None:
    """Map company career pages and push discovered job URLs into the pipeline."""
    console.print(
        f"  [bold lion]Aggressive Crawler: Mapping {len(combined_domains)} total domains "
        f"({len(uncrawled_dynamic)} dynamically discovered)[/bold lion]"
    )
    map_urls = await map_company_careers(app, combined_domains)
    console.print(f"  [cyan]Map discovery: {len(map_urls)} career-page URLs[/cyan]")

    max_map = 300
    if len(map_urls) > max_map:
        console.print(f"  [dim]Capping at {max_map} URLs (from {len(map_urls)})[/dim]")
        map_urls = map_urls[:max_map]

    scrape_sem = asyncio.Semaphore(MAX_SCRAPE_WORKERS)
    scrape_done = 0
    scrape_total = len(map_urls)
    scrape_lock = asyncio.Lock()

    async def _scrape_with_progress(item: dict) -> None:
        nonlocal scrape_done
        async with scrape_sem:
            await scrape_url_to_pipeline(item, app, pipeline)
        async with scrape_lock:
            scrape_done += 1
            if scrape_done % 20 == 0 or scrape_done == scrape_total:
                console.print(f"  Scraping map URLs... {scrape_done}/{scrape_total}")

    scrape_tasks = [_scrape_with_progress(mu) for mu in map_urls]
    await asyncio.gather(*scrape_tasks)

    if uncrawled_dynamic:
        await store.mark_domains_crawled(uncrawled_dynamic)


async def _scrape_index_links(
    app: FirecrawlApp, jobs: list[dict], max_workers: int = 12
) -> list[dict]:
    """Scrape apply-link URLs from extracted index jobs to get real JD markdown."""
    sem = asyncio.Semaphore(max_workers)
    scraped: list[dict] = []
    firecrawl_url = "http://127.0.0.1:3002"

    _dead_page_texts = (
        "sorry, we couldn't find anything here",
        "job not found",
        "this position has been filled",
    )

    _image_hosts = ("i.imgur.com", "imgur.com/", ".png", ".jpg", ".jpeg", ".gif", ".webp")

    async def _scrape_one(j: dict) -> None:
        url = j.get("apply_link", "")
        if not url or not url.startswith("http"):
            return
        url_lower = url.lower()
        if any(h in url_lower for h in _image_hosts):
            return

        role = j.get("role", "")
        company = j.get("company", "")
        location = j.get("location", "Remote")

        async with sem:
            for use_main_content in (True, False):
                try:
                    async with httpx.AsyncClient(timeout=20.0) as client:
                        payload: dict = {"url": url, "formats": ["markdown"]}
                        if use_main_content:
                            payload["onlyMainContent"] = True
                        resp = await client.post(f"{firecrawl_url}/v1/scrape", json=payload)
                        if resp.status_code == 200:
                            md = (resp.json().get("data") or {}).get("markdown", "") or ""
                            md_lower = md.lower()

                            # Prepend known metadata so the LLM always has context
                            md = (
                                f"Role: {role}\n"
                                f"Company: {company}\n"
                                f"Location: {location}\n"
                                f"Apply: {url}\n\n"
                                f"{md}"
                            )

                            if (
                                md
                                and len(md) > 200
                                and not any(dead in md_lower for dead in _dead_page_texts)
                            ):
                                scraped.append(
                                    {
                                        "markdown": md,
                                        "url": url,
                                        "title": role,
                                        "snippet": role,
                                    }
                                )
                                return
                            # If main-content gave too little, fall through to full-page scrape
                except Exception:
                    pass
            # Both attempts failed — skip

    tasks = [asyncio.create_task(_scrape_one(j)) for j in jobs]
    await asyncio.gather(*tasks)
    return scraped


async def _process_and_dispatch_batch(
    scored: list[dict],
    store: MemoryStore,
    ctx: ContextManager,
    app: FirecrawlApp,
) -> list[dict]:
    if not scored:
        return []

    scored = filter_last_24_hours(scored)

    enricher = EnrichmentAgent(store)
    enriched = await enricher.batch_enrich_and_rescore(scored, concurrency=VERIFY_CONCURRENCY)

    startup_agent = StartupAgent(ctx)
    startup_enriched = await startup_agent.batch_analyze_startups(
        enriched, concurrency=VERIFY_CONCURRENCY
    )

    high_match = [j for j in startup_enriched if int(j.get("match_percent", 0)) >= 70]
    if high_match:
        for j in high_match:
            company = str(j.get("company", ""))
            if not company or company in ("N/A", "Unknown"):
                continue
            try:
                posts = await startup_agent.mine_founder_posts(
                    company, roles=[str(j.get("role", ""))]
                )
                if posts:
                    j["founder_posts"] = posts
                    console.print(
                        f"  🚨 Founder hiring post for {company}: "
                        f"{posts[0].get('founder_name', '?')} "
                        f"({posts[0].get('post_url', '')[:60]}...)"
                    )
            except Exception:
                pass

    await JobsAgent(store=store).add_or_merge_jobs(startup_enriched)
    if ctx:
        await ctx.flush()

    cleanup_agent = CleanupAgent(store=store)
    clean_jobs = await cleanup_agent.clean_and_format_ledger()

    telegram_agent = TelegramAgent()
    if telegram_agent.is_configured:
        await telegram_agent.notify_verified_jobs(clean_jobs, store=store)

    return clean_jobs


async def _consumer(
    pipeline: JobPipeline,
    store: MemoryStore,
    app: FirecrawlApp,
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

        if len(web_buf) >= 6:
            batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
            scored = await run_batch(batch, store, concurrency=MATCH_CONCURRENCY)
            processed = await _process_and_dispatch_batch(scored, store, ctx, app)
            matched.extend(processed)
            for _ in web_buf:
                await pipeline.task_done()
            web_buf.clear()

        console.print(f"  [{pipeline.log_status()}]")

    if web_buf:
        batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
        scored = await run_batch(batch, store, concurrency=MATCH_CONCURRENCY)
        processed = await _process_and_dispatch_batch(scored, store, ctx, app)
        matched.extend(processed)
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
    telegram_agent = TelegramAgent()

    def _cleanup(signum: int, frame: object) -> None:
        console.print("\n[yellow]Interrupted - flushing LLM context...[/yellow]")
        ctx._flush_sync()
        sys.exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    await ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url="http://127.0.0.1:3002")

    await telegram_agent.start_polling()

    console.rule("[bold cyan]PHASE 0: Initialise Agent Memory (pgvector)[/bold cyan]")
    store = await MemoryStore.create()
    console.print("  [green]Connected to agent-memory-db[/green]")

    removed = await store.purge_fake_job_keys(["techco:backendengineer"])
    if removed:
        console.print(f"  [dim]Purged {removed} stale test entries[/dim]")

    console.rule("[bold cyan]PHASE 1: Load Resume + Index in pgvector[/bold cyan]")
    loop = asyncio.get_running_loop()

    existing_count = await store.chunk_count()
    chunks: dict[str, str] = {}
    full_text = ""
    reuse_resume = False

    if existing_count > 0:
        reuse_resume = True
        console.print(f"  [green]Reusing {existing_count} existing resume chunks[/green]")

    if not reuse_resume:

        def _load():
            return load_resume()

        full_text, chunks = await loop.run_in_executor(None, _load)
        chunks = enrich_candidate_chunks(chunks, full_text)
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

    set_pipeline_state(running=True, started_at=time.time(), phase="starting", sweep=0)
    if telegram_agent.is_configured:
        existing = await store.chunk_count()
        await telegram_agent.send_startup(existing)

    sweep = 0
    total_matched = 0
    while True:
        sweep += 1
        sweep_start = time.monotonic()

        try:
            set_pipeline_state(sweep=sweep, phase=f"sweep {sweep}: scraping")
            pipeline = JobPipeline()

            retry_jobs = drain_retry_queue()
            if retry_jobs:
                console.print(
                    f"  [yellow]Retrying {len(retry_jobs)} rate-limited jobs"
                    f" from previous sweep[/yellow]"
                )
                retry_scored = await run_batch(retry_jobs, store, concurrency=MATCH_CONCURRENCY)
                if retry_scored:
                    await _process_and_dispatch_batch(retry_scored, store, ctx, app)

            console.rule(f"[bold cyan]PHASE 2 (sweep {sweep}): Scrape + Graph Match[/bold cyan]")

            consumer_task = asyncio.create_task(_consumer(pipeline, store, app=app, ctx=ctx))

            # 100+ Curated ATS Platforms, Big Tech Portals, AI Startups, and Indian Unicorns
            _map_domains = [
                # ATS Roots & Global VC Boards
                "https://boards.greenhouse.io",
                "https://jobs.lever.co",
                "https://jobs.ashbyhq.com",
                "https://apply.workable.com",
                "https://jobs.smartrecruiters.com",
                "https://app.rippling.com/careers",
                "https://www.ycombinator.com/jobs",
                "https://wellfound.com/jobs",
                "https://jobs.sequoiacap.com",
                "https://jobs.a16z.com",
                # AI & Frontier Tech Labs
                "https://openai.com/careers",
                "https://jobs.ashbyhq.com/anthropic",
                "https://jobs.lever.co/cohere",
                "https://jobs.ashbyhq.com/mistral",
                "https://jobs.ashbyhq.com/scaleai",
                "https://apply.workable.com/huggingface",
                "https://jobs.ashbyhq.com/perplexity",
                "https://jobs.ashbyhq.com/character",
                "https://jobs.ashbyhq.com/anyscale",
                "https://jobs.lever.co/pinecone",
                "https://jobs.ashbyhq.com/weaviate",
                "https://jobs.ashbyhq.com/qdrant",
                "https://jobs.ashbyhq.com/wandb",
                "https://jobs.ashbyhq.com/replicate",
                "https://jobs.ashbyhq.com/together",
                # FAANG, Big Tech & Enterprise Cloud
                "https://careers.google.com",
                "https://careers.microsoft.com",
                "https://amazon.jobs/en",
                "https://jobs.apple.com",
                "https://metacareers.com",
                "https://jobs.netflix.com",
                "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
                "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site",
                "https://adobe.wd5.myworkdayjobs.com/external_experienced",
                "https://redhat.wd5.myworkdayjobs.com/en-US/jobs",
                "https://autodesk.wd1.myworkdayjobs.com/FAANG_Autodesk",
                "https://paypal.wd1.myworkdayjobs.com/paypal-careers",
                "https://crowdstrike.wd5.myworkdayjobs.com/crowdstrike",
                "https://paloaltonetworks.wd1.myworkdayjobs.com/paloaltonetworks",
                "https://jobs.ebayinc.com",
                "https://careers.oracle.com",
                "https://ibm.com/careers",
                "https://jobs.sap.com",
                "https://careers.servicenow.com",
                "https://atlassian.com/company/careers",
                "https://snowflake.com/careers",
                "https://databricks.com/company/careers",
                "https://mongodb.com/careers",
                "https://twilio.com/company/jobs",
                "https://elastic.co/about/careers",
                "https://palantir.com/careers",
                "https://cloudflare.com/careers",
                "https://okta.com/company/careers",
                "https://intuit.com/careers",
                "https://uber.com/us/en/careers",
                # Developer Tools, Fintech & Consumer Platforms
                "https://stripe.com/jobs",
                "https://block.xyz/careers",
                "https://coinbase.com/careers",
                "https://doordash.careers",
                "https://lyft.com/careers",
                "https://instacart.careers",
                "https://pinterest.careers",
                "https://snap.careers",
                "https://redditinc.com/careers",
                "https://roblox.careers",
                "https://lifeatspotify.com",
                "https://careers.duolingo.com",
                "https://careers.zoom.us",
                "https://slack.com/careers",
                "https://github.careers",
                "https://about.gitlab.com/jobs",
                "https://docker.com/careers",
                "https://hashicorp.com/careers",
                "https://confluent.io/careers",
                "https://zscaler.com/company/careers",
                "https://cisco.jobs",
                "https://qualcomm.com/company/careers",
                "https://jobs.intel.com",
                "https://jobs.amd.com",
                "https://arm.com/company/careers",
                # Top Indian Unicorns & Product Companies
                "https://careers.flipkart.com",
                "https://swiggy.com/careers",
                "https://zomato.com/careers",
                "https://jobs.lever.co/razorpay",
                "https://cred.club/careers",
                "https://meesho.com/careers",
                "https://phonepe.com/careers",
                "https://paytm.com/careers",
                "https://zepto.co.in/careers",
                "https://careers.olacabs.com",
                "https://inmobi.com/company/careers",
                "https://freshworks.com/company/careers",
                "https://zoho.com/careers",
                "https://jobs.lever.co/postman",
                "https://jobs.lever.co/browserstack",
                "https://jobs.lever.co/hasura",
                "https://unacademy.com/careers",
                "https://groww.in/careers",
                "https://careers.zerodha.com",
                "https://urbancompany.com/careers",
                "https://dream11.com/careers",
                "https://cars24.com/careers",
                "https://delhivery.com/careers",
                "https://nykaa.com/careers",
                "https://make-my-trip.com/careers",
            ]

            uncrawled_dynamic = await store.get_uncrawled_domains(limit=30)
            combined_domains = list(set(_map_domains + uncrawled_dynamic))

            domain_task = asyncio.create_task(
                _domain_producer(app, pipeline, store, combined_domains, uncrawled_dynamic)
            )

            await asyncio.gather(
                scrape_all(app, positions, ctx, pipeline, max_workers=MAX_SCRAPE_WORKERS),
                fetch_direct_json_feeds(positions, pipeline),
                *(
                    scrape_linkedin_guest_jobs(pos, location="India", pipeline=pipeline)
                    for pos in positions[:3]
                ),
            )

            await domain_task

            console.print("  [yellow]Producers done. Signalling stop...[/yellow]")
            pipeline.signal_done()

            matched_result, index_queue = await consumer_task

            if index_queue:
                console.rule(
                    f"[bold cyan]PHASE 2b (sweep {sweep}): Extract Index Jobs + Graph Match[/bold cyan]"  # noqa: E501
                )
                index_jobs = await extract_index_jobs(index_queue, ctx)
                console.print(f"  [cyan]Extracted {len(index_jobs)} jobs from indexes[/cyan]")

                # Harvest fresh ATS/career domains from apply links
                apply_links = [j.get("apply_link", "") for j in index_jobs if j.get("apply_link")]
                if apply_links:
                    new_domains = await harvest_and_save_domains(apply_links, store)
                    console.print(
                        f"  [dim]Harvested {new_domains} new career domains from apply links[/dim]"
                    )

                if index_jobs:
                    console.print(
                        f"  [cyan]Scraping {len(index_jobs)} GitHub apply links for JD text...[/cyan]"  # noqa: E501
                    )
                    idx_batch = await _scrape_index_links(
                        app, index_jobs, max_workers=MAX_SCRAPE_WORKERS
                    )
                    console.print(f"  [cyan]Scraped {len(idx_batch)} JDs from links[/cyan]")
                    if idx_batch:
                        idx_scored = await run_batch(
                            idx_batch, store, concurrency=MATCH_CONCURRENCY
                        )
                        await _process_and_dispatch_batch(idx_scored, store, ctx, app)

            cleanup_agent = CleanupAgent(store=store)
            clean_jobs = await cleanup_agent.clean_and_format_ledger()

            console.print(
                f"  [green]✓ {len(clean_jobs)} clean, verified undergrad positions "
                "stored & formatted in jobs.md[/green]"
            )

            if telegram_agent.is_configured:
                console.print(
                    "  📱 [bold yellow][TelegramAgent][/bold yellow] "
                    "Dispatching real-time notifications for verified jobs..."
                )
                await telegram_agent.notify_verified_jobs(clean_jobs, store=store)
            else:
                console.print(
                    "  📱 [dim][TelegramAgent] Telegram alerts skipped "
                    "(TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID unconfigured)[/dim]"
                )

            total_matched += len(clean_jobs)
            elapsed = time.monotonic() - sweep_start
            console.print(f"[bold green]Sweep {sweep} complete ({elapsed:.1f}s)[/bold green]")
            set_pipeline_state(matched_total=total_matched, phase="idle")
            if telegram_agent.is_configured:
                await telegram_agent.send_sweep_summary(sweep, len(clean_jobs), 0, elapsed)

            if os.environ.get("OVERNIGHT_LOOP", "false").lower() != "true":
                break

            console.print("\n[dim]Sleeping 30 minutes before next sweep...[/dim]")
            gc.collect()
            await asyncio.sleep(1800)

        except Exception as e:
            tb = traceback.format_exc()
            console.print(f"\n[red]Sweep {sweep} crashed:[/red]\n{tb}")
            set_pipeline_state(last_error=str(e), phase="crashed")
            if telegram_agent.is_configured:
                await telegram_agent.send_error(
                    f"Sweep {sweep} crashed: {e}",
                    dedup_key=f"sweep_crash_{sweep}",
                )
            gc.collect()
            console.print("  [yellow]Sleeping 15 minutes before next sweep...[/yellow]")
            await asyncio.sleep(900)

    await telegram_agent.stop_polling()
    set_pipeline_state(running=False, phase="shutdown")
    await ctx.aclose()
    await store.close()


def run() -> None:
    load_dotenv()
    asyncio.run(_run_pipeline())


if __name__ == "__main__":
    run()
