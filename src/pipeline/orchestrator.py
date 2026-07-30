"""Pipeline: resume → search → async MQ → graph pipeline → verify → output."""

import asyncio
import gc
import os
import re
import signal
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
from src.configuration import get_config
from src.connectors import discover_all
from src.graph.engine import WorkScheduler
from src.graph.entity import (
    EdgeType,
    FrontierEntry,
    GraphNode,
    NodeType,
    company_node,
    edge,
    make_founder_id,
    make_work_id,
)
from src.graph.event_bus import EventBus
from src.graph.frontier import CrawlFrontier
from src.graph.graph_store import GraphStore
from src.http_client import close_all as _close_http_clients
from src.llm.context import ContextManager
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore
from src.pipeline.graph import drain_retry_queue, run_batch
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
)

console = Console()
logger = get_logger("orchestrator")


def filter_recent_jobs(jobs: list[dict]) -> list[dict]:
    """Drop any job with a posted_date provably older than 14 days.

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
    _short_rx = re.compile(r"\b(?:(\d+)\s*(?:day|d)\s*ago)\b", re.IGNORECASE)

    filtered = []
    cutoff = datetime.now(UTC) - timedelta(days=14)
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
            if _recent_rx.search(d_lower) or _short_rx.search(d_lower):
                filtered.append(j)
            else:
                m = _old_date_rx.search(d_lower)
                if not m:
                    filtered.append(j)
    return filtered


async def map_new_domains(
    app: FirecrawlApp,
    store: MemoryStore,
    combined_domains: list[str],
    uncrawled_dynamic: list[str],
) -> list[dict[str, str]]:
    """Map company career pages and return discovered job URLs.

    Returns the raw map_urls list so the caller can feed them into
    scrape_all for unified concurrent scraping.
    """
    console.print(
        f"  [bold lion]Aggressive Crawler: Mapping {len(combined_domains)} total domains "
        f"({len(uncrawled_dynamic)} dynamically discovered)[/bold lion]"
    )
    map_urls = await map_company_careers(app, combined_domains)
    logger.info(f"Map discovery: {len(map_urls)} career-page URLs")

    if uncrawled_dynamic:
        await store.mark_domains_crawled(uncrawled_dynamic)

    return map_urls


async def _scrape_index_links(
    app: FirecrawlApp, jobs: list[dict], max_workers: int = 12
) -> list[dict]:
    """Scrape apply-link URLs from extracted index jobs to get real JD markdown."""
    sem = asyncio.Semaphore(max_workers)
    scraped: list[dict] = []
    cfg = get_config().firecrawl

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
                        resp = await client.post(f"{cfg.url}/v1/scrape", json=payload)
                        if resp.status_code == 200:
                            md = (resp.json().get("data") or {}).get("markdown", "") or ""
                            md_lower = md.lower()

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
                except Exception:
                    pass

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

    try:
        cfg = get_config().pipeline

        scored = filter_recent_jobs(scored)

        enricher = EnrichmentAgent(store)
        enriched = await enricher.batch_enrich_and_rescore(
            scored, concurrency=cfg.verify_concurrency
        )

        startup_agent = StartupAgent(ctx)
        startup_enriched = await startup_agent.batch_analyze_startups(
            enriched, concurrency=cfg.verify_concurrency
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
                        logger.info(
                            "Founder hiring post discovered",
                            entity=company,
                            extra={"founder": posts[0].get("founder_name", "?")},
                        )
                except Exception:
                    pass

        await JobsAgent(store=store).add_or_merge_jobs(startup_enriched)
        if ctx:
            await ctx.flush()

        cleanup_agent = CleanupAgent(store=store)
        clean_jobs = await cleanup_agent.clean_and_format_ledger()

        telegram_agent = TelegramAgent(ctx=ctx)
        if telegram_agent.is_configured:
            await telegram_agent.notify_verified_jobs(clean_jobs, store=store)

        return clean_jobs

    except Exception as e:
        logger.error(
            "Dispatch batch failed, returning un-enriched jobs",
            exception=str(e),
            extra={"job_count": len(scored)},
        )
        return filter_recent_jobs(scored)


async def _consumer(
    pipeline: JobPipeline,
    store: MemoryStore,
    app: FirecrawlApp,
    ctx: ContextManager,
) -> tuple[list[dict], list[QueuedJob]]:
    cfg = get_config().pipeline
    matched: list[dict] = []
    index_queue: list[QueuedJob] = []
    web_buf: list[QueuedJob] = []

    while True:
        try:
            job = await pipeline.pop(timeout=2)
        except asyncio.CancelledError:
            break
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
            scored = await run_batch(batch, store, concurrency=cfg.match_concurrency)
            processed = await _process_and_dispatch_batch(scored, store, ctx, app)
            matched.extend(processed)
            for _ in web_buf:
                await pipeline.task_done()
            web_buf.clear()

    if web_buf:
        batch = [{"markdown": j.markdown, "url": j.url, "title": j.title} for j in web_buf]
        scored = await run_batch(batch, store, concurrency=cfg.match_concurrency)
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
    cfg = get_config().embed
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
                    f"{cfg.url}/embeddings",
                    json={"model": cfg.model, "input": batch},
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
                    f"{cfg.url}/embeddings",
                    json={"model": cfg.model, "input": [text[:500]]},
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
        logger.info(f"Indexed {len(records)} resume chunks in pgvector")
    finally:
        await embed_client.aclose()


async def _run_pipeline() -> None:
    cfg = get_config()

    ctx = ContextManager()
    telegram_agent = TelegramAgent(ctx=ctx)

    shutdown_requested = asyncio.Event()

    def _cleanup(signum: int, frame: object) -> None:
        logger.info("Interrupted - flushing LLM context...", extra={"signal": signum})
        ctx._flush_sync()
        shutdown_requested.set()

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    await ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url=cfg.firecrawl.url)

    await telegram_agent.start_polling()

    console.rule("[bold cyan]PHASE 0: Initialise Agent Memory (pgvector)[/bold cyan]")
    store = await MemoryStore.create()
    logger.info("Connected to agent-memory-db")

    removed = await store.purge_fake_job_keys(["techco:backendengineer"])
    if removed:
        logger.info(f"Purged {removed} stale test entries")

    graph = await GraphStore.create()
    bus = EventBus()
    logger.info("Graph store initialised")

    engine_cfg = get_config().scheduler
    frontier = CrawlFrontier(max_size=engine_cfg.max_queue_size)
    engine = WorkScheduler(frontier, worker_count=3)

    bus.set_enqueue_callback(engine.enqueue_many)

    startup_agent = StartupAgent(ctx)

    async def _on_company_discovered(event):
        d = event.payload
        entries: list[FrontierEntry] = [
            FrontierEntry(
                id=make_work_id("founder_miner", event.node_id),
                agent="founder_miner",
                node_id=event.node_id,
                node_type=NodeType.COMPANY,
                priority=70,
                depth=1,
                payload={"company": d["name"]},
            )
        ]
        if d.get("url"):
            entries.append(
                FrontierEntry(
                    id=make_work_id("career_site_detector", event.node_id),
                    agent="career_site_detector",
                    node_id=event.node_id,
                    node_type=NodeType.COMPANY,
                    priority=60,
                    depth=1,
                    payload={"company": d["name"], "url": d["url"]},
                )
            )
        return entries

    async def _on_founder_discovered(event):
        d = event.payload
        return [
            FrontierEntry(
                id=make_work_id("founder_social_osint", event.node_id),
                agent="founder_social_osint",
                node_id=event.node_id,
                node_type=NodeType.FOUNDER,
                priority=50,
                depth=2,
                payload={"founder_name": d.get("name", ""), "company": d.get("company", "")},
            ),
            FrontierEntry(
                id=make_work_id("employee_discovery", event.node_id),
                agent="employee_discovery",
                node_id=event.node_id,
                node_type=NodeType.FOUNDER,
                priority=45,
                depth=2,
                payload={"company": d.get("company", "")},
            ),
        ]

    async def _on_career_site_discovered(event):
        url = event.payload.get("url", "")
        if any(
            a in url.lower()
            for a in ("greenhouse", "lever.co", "ashbyhq", "workable", "myworkdayjobs")
        ):
            return [
                FrontierEntry(
                    id=make_work_id("ats_crawler", event.node_id),
                    agent="ats_crawler",
                    node_id=event.node_id,
                    node_type=NodeType.CAREER_SITE,
                    priority=55,
                    depth=2,
                    payload={"company": event.payload.get("company", ""), "ats_url": url},
                )
            ]
        return []

    async def _founder_miner(entry: FrontierEntry) -> list[FrontierEntry]:
        cn = entry.payload.get("company", "")
        if not cn:
            return []
        enriched = await startup_agent.analyze_startup(
            {
                "role": "Startup Analysis",
                "company": cn,
                "match_percent": 50,
                "verdict": "WEAK_MATCH",
            }
        )
        founders = enriched.get("founders", [])
        node = await graph.get_node(entry.node_id)
        if node:
            node.data["founders"] = founders
            node.data["funding_stage"] = enriched.get("funding_stage", "")
            node, _ = await graph.upsert_node(node)
        results: list[FrontierEntry] = []
        for f in founders[:3]:
            if isinstance(f, dict) and f.get("name"):
                f_node = GraphNode(
                    id=make_founder_id(f["name"], cn),
                    node_type=NodeType.FOUNDER,
                    data={**f, "company": cn},
                )
                f_node, _ = await graph.upsert_node(f_node)
                _, _ = await graph.upsert_edge(edge(entry.node_id, EdgeType.FOUNDED_BY, f_node.id))
                await bus.fire(
                    bus.new_event(
                        "founder_discovered",
                        f_node.id,
                        NodeType.FOUNDER,
                        {"name": f["name"], "company": cn},
                    )
                )
        return results

    async def _stub(entry: FrontierEntry) -> list[FrontierEntry]:
        return []

    bus.subscribe("company_discovered", _on_company_discovered)
    bus.subscribe("founder_discovered", _on_founder_discovered)
    bus.subscribe("career_site_discovered", _on_career_site_discovered)

    engine.register_agent("founder_miner", _founder_miner)
    engine.register_agent("career_site_detector", _stub)
    engine.register_agent("founder_social_osint", _stub)
    engine.register_agent("employee_discovery", _stub)
    engine.register_agent("ats_crawler", _stub)

    engine.start(worker_count=3)
    logger.info("Graph expansion engine started")

    console.rule("[bold cyan]PHASE 1: Load Resume + Index in pgvector[/bold cyan]")
    loop = asyncio.get_running_loop()

    existing_count = await store.chunk_count()
    chunks: dict[str, str] = {}
    full_text = ""
    reuse_resume = False

    if existing_count > 0:
        reuse_resume = True
        logger.info(f"Reusing {existing_count} existing resume chunks")

    if not reuse_resume:

        def _load():
            return load_resume()

        full_text, chunks = await loop.run_in_executor(None, _load)
        chunks = await enrich_candidate_chunks(chunks, full_text)
        await _index_resume_in_pgvector(chunks, store)

    positions: list[str]
    if reuse_resume and not full_text:
        positions = ["Software Engineer", "Backend Developer"]
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
    logger.info(f"Target positions: {', '.join(positions)}")

    set_pipeline_state(running=True, started_at=time.time(), phase="starting", sweep=0)
    if telegram_agent.is_configured:
        existing = await store.chunk_count()
        await telegram_agent.send_startup(existing)

    sweep = 0
    total_matched = 0
    while True:
        if shutdown_requested.is_set():
            logger.info("Shutdown requested, breaking sweep loop")
            break

        sweep += 1
        sweep_start = time.monotonic()

        try:
            set_pipeline_state(sweep=sweep, phase=f"sweep {sweep}: scraping")
            pipeline = JobPipeline()

            retry_jobs = drain_retry_queue()
            if retry_jobs:
                logger.info(f"Retrying {len(retry_jobs)} rate-limited jobs from previous sweep")
                retry_scored = await run_batch(
                    retry_jobs, store, concurrency=cfg.pipeline.match_concurrency
                )
                if retry_scored:
                    await _process_and_dispatch_batch(retry_scored, store, ctx, app)

            console.rule(f"[bold cyan]PHASE 2 (sweep {sweep}): Scrape + Graph Match[/bold cyan]")

            consumer_task = asyncio.create_task(_consumer(pipeline, store, app=app, ctx=ctx))

            _map_domains = [
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

            uncrawled_dynamic = await store.get_uncrawled_domains(limit=150)
            combined_domains = list(set(_map_domains + uncrawled_dynamic))

            map_urls = await map_new_domains(app, store, combined_domains, uncrawled_dynamic)
            max_map = 300
            if len(map_urls) > max_map:
                map_urls = map_urls[:max_map]

            await asyncio.gather(
                scrape_all(
                    app,
                    positions,
                    ctx,
                    pipeline,
                    max_workers=cfg.pipeline.max_scrape_workers,
                    map_urls=map_urls,
                ),
                fetch_direct_json_feeds(positions, pipeline),
                *(
                    scrape_linkedin_guest_jobs(pos, location="India", pipeline=pipeline)
                    for pos in positions[:3]
                ),
            )

            pipeline.signal_done()

            matched_result, index_queue = await consumer_task

            if index_queue:
                console.rule(
                    f"[bold cyan]PHASE 2b (sweep {sweep}): "
                    "Extract Index Jobs + Graph Match[/bold cyan]"
                )
                index_jobs = await extract_index_jobs(index_queue, ctx)
                logger.info(f"Extracted {len(index_jobs)} jobs from indexes")

                apply_links = [j.get("apply_link", "") for j in index_jobs if j.get("apply_link")]
                if apply_links:
                    new_domains = await harvest_and_save_domains(apply_links, store)
                    logger.info(f"Harvested {new_domains} new career domains from apply links")

                if index_jobs:
                    logger.info(f"Scraping {len(index_jobs)} GitHub apply links for JD text...")
                    idx_batch = await _scrape_index_links(
                        app, index_jobs, max_workers=cfg.pipeline.max_scrape_workers
                    )
                    logger.info(f"Scraped {len(idx_batch)} JDs from links")
                    if idx_batch:
                        idx_scored = await run_batch(
                            idx_batch, store, concurrency=cfg.pipeline.match_concurrency
                        )
                        await _process_and_dispatch_batch(idx_scored, store, ctx, app)

            cleanup_agent = CleanupAgent(store=store)
            clean_jobs = await cleanup_agent.clean_and_format_ledger()

            logger.info(f"{len(clean_jobs)} clean, verified undergrad positions stored & formatted")

            if telegram_agent.is_configured:
                await telegram_agent.notify_verified_jobs(clean_jobs, store=store)

            total_matched += len(clean_jobs)
            elapsed = time.monotonic() - sweep_start
            logger.info(f"Sweep {sweep} complete ({elapsed:.1f}s, {len(clean_jobs)} matches)")
            set_pipeline_state(matched_total=total_matched, phase="idle")
            if telegram_agent.is_configured:
                await telegram_agent.send_sweep_summary(sweep, len(clean_jobs), 0, elapsed)

            if sweep % 3 == 0:
                set_pipeline_state(phase="graph maintenance")
                console.print(
                    "\n[bold cyan]PHASE 2c: Graph Maintenance (metrics + decay)[/bold cyan]"
                )
                try:
                    decayed = await graph.decay_stale_confidence(max_age_days=30)
                    logger.info(f"Confidence decay: {decayed} nodes decayed")
                    metrics = await graph.update_all_graph_metrics()
                    logger.info(
                        "Graph metrics updated",
                        extra={
                            "pagerank": metrics.get("pagerank_nodes", 0),
                            "components": metrics.get("wcc_components", 0),
                        },
                    )
                    generated = await engine.expansion_cycle(graph)
                    if generated:
                        logger.info(f"Expansion cycle: {generated} tasks enqueued")
                    entities = await discover_all()
                    if entities:
                        logger.info("Connector discovery: %d new companies", len(entities))
                        for entity in entities:
                            node = company_node(
                                entity.name,
                                description=entity.description,
                                source=entity.source,
                                url=entity.url or "",
                            )
                            node.confidence.score = entity.confidence
                            node.confidence.source_count = 1
                            node, node_events = await graph.upsert_node(node)
                            for evt in node_events:
                                await engine.process_mutation(evt, graph)
                            await bus.fire(
                                bus.new_event(
                                    "company_discovered",
                                    node.id,
                                    NodeType.COMPANY,
                                    {
                                        "name": entity.name,
                                        "url": entity.url,
                                        "source": entity.source,
                                        "description": entity.description,
                                    },
                                )
                            )

                    if telegram_agent.is_configured:
                        stealth = await graph.detect_stealth_hiring_signals(limit=8)
                        if stealth:
                            await telegram_agent.push_stealth_and_warm_intro_batch(stealth)
                except Exception as ge:
                    logger.exception("Graph maintenance skipped", exc=ge)

            if os.environ.get("OVERNIGHT_LOOP", "false").lower() != "true":
                break

            gc.collect()
            await asyncio.sleep(cfg.pipeline.sweep_interval)

        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
            if "consumer_task" in locals() and not consumer_task.done():
                consumer_task.cancel()
            break
        except Exception as e:
            if "consumer_task" in locals() and not consumer_task.done():
                consumer_task.cancel()
            tb = traceback.format_exc()
            logger.exception(f"Sweep {sweep} crashed", exc=e)
            set_pipeline_state(last_error=str(e), phase="crashed")
            if telegram_agent.is_configured:
                await telegram_agent.send_error(
                    f"Sweep {sweep} crashed: {e}",
                    dedup_key=f"sweep_crash_{sweep}",
                )
            gc.collect()
            await asyncio.sleep(cfg.pipeline.sweep_interval)

    await engine.shutdown(drain=False)

    await telegram_agent.stop_polling()
    set_pipeline_state(running=False, phase="shutdown")
    await bus.shutdown(timeout=5.0)
    await ctx.aclose()
    await _close_http_clients()
    await graph.close()
    await store.close()
    logger.info("Pipeline shutdown complete")


def run() -> None:
    load_dotenv()
    # Validate config on startup
    cfg = get_config()
    problems = cfg.validate()
    if problems:
        for p in problems:
            logger.warning(f"Config problem: {p}")
    asyncio.run(_run_pipeline())


if __name__ == "__main__":
    run()
