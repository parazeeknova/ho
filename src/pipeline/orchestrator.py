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
from src.connectors import discover_all
from src.graph.engine import WorkScheduler
from src.graph.entity import (
    EdgeType,
    FrontierEntry,
    GraphNode,
    NodeType,
    company_node,
    compute_centrality,
    edge,
    make_founder_id,
    make_work_id,
)
from src.graph.event_bus import EventBus
from src.graph.frontier import CrawlFrontier
from src.graph.graph_store import GraphStore
from src.llm.context import ContextManager
from src.memory.pgvector_store import MemoryStore
from src.pipeline.graph import EMBED_URL, drain_retry_queue, run_batch
from src.pipeline.queue import JobPipeline, QueuedJob
from src.rag.github_linkedin_loader import enrich_candidate_chunks
from src.rag.loader import load_resume
from src.search.linkedin_guest import scrape_linkedin_guest_jobs
from src.search.searcher import (
    TARGET_POSITIONS_SCHEMA,
    extract_career_domain,
    extract_index_jobs,
    fetch_direct_json_feeds,
    harvest_and_save_domains,
    map_company_careers,
    scrape_all,
    scrape_url_to_pipeline,
)
from src.search.startup_discoverer import discover_startups

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

    telegram_agent = TelegramAgent(ctx=ctx)
    if telegram_agent.is_configured:
        await telegram_agent.notify_verified_jobs(clean_jobs, store=store)

    return clean_jobs


async def _enrich_discovered_startups(
    startups: list[dict[str, str]],
    store: MemoryStore,
    ctx: ContextManager,
    telegram_agent: TelegramAgent,
) -> int:
    """Enrich startup discoveries with founder/funding data, store in ledger,
    and queue careers domains for next-sweep scraping. Returns count stored."""
    if not startups:
        return 0

    startup_agent = StartupAgent(ctx)
    stored = 0

    for s in startups:
        company_name = s.get("company", "").strip()
        if not company_name or company_name in ("N/A", "Unknown"):
            continue

        job_entry = {
            "role": "[Startup Discovery]",
            "company": company_name,
            "company_description": s.get("description", ""),
            "apply_link": s.get("url", ""),
            "url": s.get("url", ""),
            "source": s.get("source", "discovered"),
            "location": "Remote",
            "match_percent": 50,
            "verdict": "WEAK_MATCH",
            "shortlist_probability": 40,
        }

        try:
            enriched = await startup_agent.analyze_startup(job_entry)
        except Exception:
            enriched = job_entry

        await JobsAgent(store=store).add_or_merge_jobs([enriched])

        # Harvest career domains for next sweep
        url = s.get("url", "")
        if url and url.startswith("http"):
            try:
                domain = extract_career_domain(url)
                if domain:
                    await store.add_discovered_domain(domain, url)
            except Exception:
                pass

        stored += 1

    await ctx.flush()
    cleanup_agent = CleanupAgent(store=store)
    await cleanup_agent.clean_and_format_ledger()

    if telegram_agent.is_configured and stored > 0:
        sample = ", ".join(s["company"] for s in startups[:5] if s.get("company"))
        await telegram_agent._send_raw(
            f"🏢 <b>Startup Discovery</b>\n\nDiscovered {stored} startups: {sample}..."
        )

    return stored


async def _expand_company_graph(
    graph: GraphStore,
    bus: EventBus,
    store: MemoryStore,
    ctx: ContextManager,
    telegram_agent: TelegramAgent,
) -> tuple[int, int]:
    """Graph expansion round with fire-and-forget event bus, centrality."""
    startup_agent = StartupAgent(ctx)
    frontier = CrawlFrontier(max_size=300)
    engine = WorkScheduler(frontier, worker_count=3)

    # Wire event bus -> frontier: handler results auto-enqueue
    bus.set_enqueue_callback(engine.enqueue_many)

    async def _on_company_discovered(event):
        d = event.payload
        entries = [
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

    bus.subscribe("company_discovered", _on_company_discovered)
    bus.subscribe("founder_discovered", _on_founder_discovered)
    bus.subscribe("career_site_discovered", _on_career_site_discovered)

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
            await graph.upsert_node(node)
        results: list[FrontierEntry] = []
        for f in founders[:3]:
            if isinstance(f, dict) and f.get("name"):
                f_node = GraphNode(
                    id=make_founder_id(f["name"], cn),
                    node_type=NodeType.FOUNDER,
                    data={**f, "company": cn},
                )
                await graph.upsert_node(f_node)
                await graph.upsert_edge(edge(entry.node_id, EdgeType.FOUNDED_BY, f_node.id))
                # Fire-and-forget: spawns handler task, returns immediately
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

    engine.register_agent("founder_miner", _founder_miner)
    engine.register_agent("career_site_detector", _stub)
    engine.register_agent("founder_social_osint", _stub)
    engine.register_agent("employee_discovery", _stub)
    engine.register_agent("ats_crawler", _stub)

    engine.start(worker_count=3)

    # Discovery phase

    entities = await discover_all()
    if not entities:
        await engine.shutdown(drain=False)
        return 0, 0

    console.print(f"  🏢 Connectors: {len(entities)} companies discovered")
    nodes_created = 0
    events_fired = 0

    for entity in entities:
        node = company_node(
            entity.name, description=entity.description, source=entity.source, url=entity.url or ""
        )
        node.confidence.score = entity.confidence
        node.confidence.source_count = 1
        await graph.upsert_node(node)
        nodes_created += 1

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
        events_fired += 1

    # Centrality

    all_nodes = await graph.get_nodes_by_type(NodeType.COMPANY, limit=200)
    all_edges = await graph.get_all_edges(limit=500)
    node_map = {n.id: n for n in all_nodes}
    edge_dicts = [{"source": e.source_id, "target": e.target_id} for e in all_edges]
    centrality = compute_centrality(list(node_map.keys()), edge_dicts)
    for nid, rank in centrality.items():
        node = node_map.get(nid)
        if node and rank > 0.01:
            node.data["pagerank"] = rank
            await graph.upsert_node(node)

    await asyncio.sleep(3)

    # Graph-driven work generation
    # New edges trigger discovery: FOUNDED_BY -> check for missing founder profiles
    for e in all_edges:
        target = await graph.get_node(e.target_id)
        source = await graph.get_node(e.source_id)
        if target and source and e.edge_type == EdgeType.FOUNDED_BY:
            founder_name = target.data.get("name", "")
            if founder_name:
                existing = await graph.get_nodes_by_type(NodeType.FOUNDER, limit=1)
                if not any(n.data.get("linkedin_url") for n in existing):
                    await engine.enqueue(
                        FrontierEntry(
                            id=make_work_id("founder_social_osint", target.id),
                            agent="founder_social_osint",
                            node_id=target.id,
                            node_type=NodeType.FOUNDER,
                            priority=40,
                            depth=2,
                            payload={
                                "founder_name": founder_name,
                                "company": source.data.get("name", ""),
                            },
                        )
                    )

    # Store

    jobs_agent = JobsAgent(store=store)
    for entity in entities[:20]:
        await jobs_agent.add_or_merge_jobs(
            [
                {
                    "role": "[Graph Discovery]",
                    "company": entity.name,
                    "company_description": entity.description,
                    "apply_link": entity.url or "",
                    "source": entity.source,
                    "location": "Remote",
                    "match_percent": 40,
                    "verdict": "WEAK_MATCH",
                    "shortlist_probability": 30,
                }
            ]
        )

    m = await engine.get_metrics()
    console.print(
        f"  Scheduler: {m.completed_work} done, {m.pending_work} pending"
        f" | Centrality: {len(centrality)} nodes"
    )
    await engine.shutdown(drain=True)
    return nodes_created, events_fired


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
    telegram_agent = TelegramAgent(ctx=ctx)

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

    graph = await GraphStore.create()
    bus = EventBus()
    console.print("  [green]Graph store initialised[/green]")

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

            startup_discovery_task = asyncio.create_task(discover_startups(positions))

            await asyncio.gather(
                scrape_all(app, positions, ctx, pipeline, max_workers=MAX_SCRAPE_WORKERS),
                fetch_direct_json_feeds(positions, pipeline),
                *(
                    scrape_linkedin_guest_jobs(pos, location="India", pipeline=pipeline)
                    for pos in positions[:3]
                ),
            )

            await domain_task

            discovered = await startup_discovery_task
            if discovered:
                console.print(f"  🏢 Enriching {len(discovered)} discovered startups...")
                stored = await _enrich_discovered_startups(discovered, store, ctx, telegram_agent)
                console.print(f"  🏢 Stored {stored} startup discoveries in ledger")

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

            set_pipeline_state(phase="graph expansion")
            console.print("\n[bold cyan]PHASE 3: Company Graph Expansion[/bold cyan]")
            try:
                nodes, events = await _expand_company_graph(graph, bus, store, ctx, telegram_agent)
                console.print(f"  🏢 Graph: {nodes} nodes, {events} events")
            except Exception as ge:
                console.print(f"  [yellow]Graph expansion skipped: {ge}[/yellow]")

            console.print("\n[dim]Sleeping 5 minutes before next sweep...[/dim]")
            gc.collect()
            await asyncio.sleep(300)

        except Exception as e:
            if "consumer_task" in locals() and not consumer_task.done():
                consumer_task.cancel()
            tb = traceback.format_exc()
            console.print(f"\n[red]Sweep {sweep} crashed:[/red]\n{tb}")
            set_pipeline_state(last_error=str(e), phase="crashed")
            if telegram_agent.is_configured:
                await telegram_agent.send_error(
                    f"Sweep {sweep} crashed: {e}",
                    dedup_key=f"sweep_crash_{sweep}",
                )
            gc.collect()
            console.print("  [yellow]Sleeping 5 minutes before next sweep...[/yellow]")
            await asyncio.sleep(300)

    await telegram_agent.stop_polling()
    set_pipeline_state(running=False, phase="shutdown")
    await ctx.aclose()
    await graph.close()
    await store.close()


def run() -> None:
    load_dotenv()
    asyncio.run(_run_pipeline())


if __name__ == "__main__":
    run()
