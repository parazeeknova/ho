"""RadarOrchestrator: source-first incremental job radar pipeline.

Replaces the legacy orchestrator as the primary ingestion path.
Keeps the legacy pipeline behind LEGACY_PIPELINE=false-by-default flag.
"""

from __future__ import annotations

import asyncio
import gc
import os
import signal
import time
from typing import Any

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from rich.console import Console

from src.agent.startup_agent import StartupAgent
from src.agent.telegram_agent import TelegramAgent, set_pipeline_state
from src.configuration import get_config
from src.graph.engine import WorkScheduler
from src.graph.entity import (
    EdgeType,
    FrontierEntry,
    GraphNode,
    NodeType,
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
from src.radar.agents import (
    ats_crawler,
    career_site_detector,
    employee_discovery_agent,
    founder_social_agent,
)
from src.radar.extractors import extract_github_index_markdown
from src.radar.gates import run_gates
from src.radar.models import JobCandidate, JobObservation
from src.radar.outreach import generate_outreach_card
from src.radar.queue import enqueue_candidate, get_queue_status, process_queue
from src.radar.sources import (
    get_source_health,
    load_checkpoints,
    persist_checkpoints,
    record_failure,
    record_success,
    register_source,
)
from src.rag.loader import load_resume
from src.search.searcher import (
    GITHUB_INDEXES,
)

console = Console()
logger = get_logger("radar_orchestrator")


async def _job_processor_handler(entry: FrontierEntry) -> list[FrontierEntry]:
    """Process a single job observation through the radar pipeline."""
    url = entry.payload.get("observation_url", "")
    source = entry.payload.get("source", "unknown")
    company = entry.payload.get("company", "")

    if not url or not url.startswith("http"):
        return []

    try:
        cfg = get_config().firecrawl
        import httpx

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{cfg.url}/v1/scrape",
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            if resp.status_code != 200:
                record_failure(source)
                return []
            md = (resp.json().get("data") or {}).get("markdown", "") or ""
            if not md or len(md) < 100:
                record_failure(source)
                return []

        obs = JobObservation(
            url=url,
            source=source,
            raw_markdown=md,
            title=company or url,
            snippet=f"{company} — {url[:80]}",
        )

        known: set[str] = set()
        last_seen: dict[str, float] = {}
        candidate, rejections = await run_gates(obs, known, last_seen)

        if candidate is None:
            for _gate, reason, _desc in rejections:
                logger.info("Radar gate rejected", gate=_gate, reason=reason.value, url=url)
            return []

        candidate.extra["raw_markdown"] = md
        await enqueue_candidate(candidate, priority=40)
        return []

    except Exception as e:
        logger.warning("Job processor failed", url=url, exception=str(e))
        record_failure(source)
        return []


async def _outreach_handler(entry: FrontierEntry) -> list[FrontierEntry]:
    """Generate cold-outreach cards for founders with sufficient signal."""
    company = entry.payload.get("company", "")
    founder_name = entry.payload.get("founder_name", "")
    linkedin_url = entry.payload.get("linkedin", "")

    candidate = JobCandidate(
        canonical_id=f"{company}:{founder_name}:outreach",
        source="outreach_generator",
        direct_apply_url=linkedin_url,
        normalized_company=company,
        normalized_role="",
        normalized_location="Remote",
        founders=[{"name": founder_name, "linkedin_url": linkedin_url}],
    )

    card = generate_outreach_card(candidate)
    if card:
        telegram_agent = TelegramAgent()
        if telegram_agent.is_configured:
            await telegram_agent.send_categorized_alert(
                "outreach",
                {
                    "role": f"Outreach to {founder_name}",
                    "company": company,
                    "apply_link": linkedin_url,
                    "founders": candidate.founders,
                    "funding_stage": candidate.funding_stage,
                },
                dedup_key=f"outreach:{company}:{founder_name}",
            )
    return []


async def _scrape_indexes(app: FirecrawlApp) -> list[JobObservation]:
    """Fetch GitHub indexes and extract direct apply URLs locally."""
    all_observations: list[JobObservation] = []
    import httpx

    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx_url in GITHUB_INDEXES:
            try:
                resp = await client.get(idx_url)
                if resp.status_code == 200:
                    md = resp.text
                    obs = extract_github_index_markdown(md, idx_url)
                    all_observations.extend(obs)
                    record_success(f"github:{idx_url.split('/')[-1]}", len(obs), len(obs))
            except Exception:
                record_failure(f"github:{idx_url.split('/')[-1]}")
    return all_observations


async def _fetch_and_gate_observations(
    observations: list[JobObservation],
) -> list[JobCandidate]:
    """Fetch posting content and run deterministic gates."""
    passed: list[JobCandidate] = []
    cfg = get_config().firecrawl
    import httpx

    known: set[str] = set()
    last_seen: dict[str, float] = {}

    sem = asyncio.Semaphore(8)

    async def _process_one(obs: JobObservation) -> None:
        async with sem:
            try:
                if obs.raw_markdown and len(obs.raw_markdown) > 100:
                    md = obs.raw_markdown
                else:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.post(
                            f"{cfg.url}/v1/scrape",
                            json={"url": obs.url, "formats": ["markdown"], "onlyMainContent": True},
                        )
                        if resp.status_code != 200:
                            return
                        md = (resp.json().get("data") or {}).get("markdown", "") or ""
                        if not md or len(md) < 100:
                            return

                obs.raw_markdown = md
                candidate, rejections = await run_gates(obs, known, last_seen)
                if candidate is not None:
                    candidate.extra["raw_markdown"] = md
                    passed.append(candidate)
                else:
                    for _gate, _reason, _desc in rejections:
                        known.add(obs.canonical_url_hash())
            except Exception:
                pass

    tasks = [asyncio.create_task(_process_one(o)) for o in observations]
    await asyncio.gather(*tasks, return_exceptions=True)
    return passed


async def _run_radar_pipeline() -> None:
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

    console.rule("[bold cyan]RADAR PHASE 0: Initialise Memory + Graph[/bold cyan]")
    store = await MemoryStore.create()
    removed = await store.purge_fake_job_keys(["techco:backendengineer"])
    if removed:
        logger.info(f"Purged {removed} stale test entries")

    graph = await GraphStore.create()
    bus = EventBus()

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
                    payload={
                        "company": event.payload.get("company", ""),
                        "ats_url": url,
                        "ats_type": "ats_board",
                    },
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

    bus.subscribe("company_discovered", _on_company_discovered)
    bus.subscribe("founder_discovered", _on_founder_discovered)
    bus.subscribe("career_site_discovered", _on_career_site_discovered)

    engine.register_agent("founder_miner", _founder_miner)
    engine.register_agent("career_site_detector", career_site_detector)
    engine.register_agent("founder_social_osint", founder_social_agent)
    engine.register_agent("employee_discovery", employee_discovery_agent)
    engine.register_agent("ats_crawler", ats_crawler)
    engine.register_agent("job_processor", _job_processor_handler)
    engine.register_agent("outreach_generator", _outreach_handler)

    engine.start(worker_count=3)
    logger.info("Radar graph expansion engine started")

    console.rule("[bold cyan]RADAR PHASE 1: Load Resume[/bold cyan]")
    loop = asyncio.get_running_loop()
    existing_count = await store.chunk_count()

    full_text = ""
    if existing_count > 0:
        logger.info(f"Reusing {existing_count} existing resume chunks")
    else:

        def _load():
            return load_resume()

        full_text, chunks = await loop.run_in_executor(None, _load)
        from src.pipeline.orchestrator import _index_resume_in_pgvector

        await _index_resume_in_pgvector(chunks, store)

    candidate_persona = cfg.candidate.persona
    await load_checkpoints(store)

    register_source("greenhouse_boards", "ats_board", initial_quality=0.8)
    register_source("lever_boards", "ats_board", initial_quality=0.8)
    register_source("ashby_boards", "ats_board", initial_quality=0.8)
    register_source("workable_boards", "ats_board", initial_quality=0.7)
    register_source("smartrecruiters_boards", "ats_board", initial_quality=0.7)
    for idx_url in GITHUB_INDEXES:
        register_source(f"github:{idx_url.split('/')[-1]}", "github_index", initial_quality=0.6)

    set_pipeline_state(
        running=True,
        started_at=time.time(),
        phase="starting",
        sweep=0,
        rejected_total=0,
        matched_total=0,
    )
    if telegram_agent.is_configured:
        await telegram_agent.send_startup(existing_count)

    sweep = 0

    while True:
        if shutdown_requested.is_set():
            break

        sweep += 1
        sweep_start = time.monotonic()
        set_pipeline_state(
            sweep=sweep,
            phase=f"sweep {sweep}: scraping",
            sweep_started_at=time.time(),
        )

        try:
            console.rule(
                f"[bold cyan]RADAR PHASE 2 (sweep {sweep}): Source Polling + Gating[/bold cyan]",
            )

            all_observations: list[JobObservation] = []

            idx_obs = await _scrape_indexes(app)
            all_observations.extend(idx_obs)
            logger.info(f"GitHub indexes: {len(idx_obs)} observations")

            ats_sources = [
                sid
                for sid, cp in get_source_health().items()
                if cp["type"] == "ats_board" and cp["active"]
            ]

            map_urls = []
            for ats_url in ats_sources[:3]:
                try:
                    resp_map = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda u=ats_url: app.map_url(u),
                    )
                    if isinstance(resp_map, list):
                        map_urls.extend(resp_map)
                except Exception:
                    pass

            for url in map_urls[:100]:
                if isinstance(url, dict):
                    link = url.get("url", "")
                elif isinstance(url, str):
                    link = url
                else:
                    continue
                if link.startswith("http"):
                    all_observations.append(
                        JobObservation(url=link, source="ats_map", title="", snippet="")
                    )

            candidates = await _fetch_and_gate_observations(all_observations)
            gated_count = len(candidates)
            rejected_count = len(all_observations) - gated_count
            logger.info(f"Gating: {gated_count} passed, {rejected_count} rejected")

            console.rule(f"[bold cyan]RADAR PHASE 3 (sweep {sweep}): LLM Matching[/bold cyan]")
            resume_ctx = full_text[:3000] if full_text else cfg.candidate.persona
            for c in candidates:
                await enqueue_candidate(c, priority=50)

            matched = await process_queue(
                ctx,
                resume_ctx,
                candidate_persona,
                store,
                max_candidates=cfg.radar.max_candidates_per_sweep,
            )
            logger.info(f"LLM queue: {len(matched)} matched")

            emergency = [c for c in matched if c.is_urgent and c.is_accepted]
            startup_signals = [c for c in matched if c.is_accepted and c.funding_stage]
            eligible = [c for c in matched if c.is_accepted and not c.is_urgent]
            review = [c for c in matched if c.freshness_lane.name.lower() == "review"]

            if telegram_agent.is_configured:
                for c in emergency:
                    await telegram_agent.send_categorized_alert(
                        "urgent", _candidate_to_job_card(c), dedup_key=c.canonical_id
                    )
                for c in startup_signals[:10]:
                    if c not in emergency:
                        await telegram_agent.send_categorized_alert(
                            "startup_signal", _candidate_to_job_card(c), dedup_key=c.canonical_id
                        )
                dig = [_candidate_to_job_card(c) for c in eligible[:5]]
                await telegram_agent.send_category_digest("eligible", dig)
                drv = [_candidate_to_job_card(c) for c in review[:5]]
                await telegram_agent.send_category_digest("review", drv)

            queue_s = get_queue_status()
            set_pipeline_state(
                matched_total=len([c for c in matched if c.is_accepted]),
                rejected_total=len([c for c in matched if c.is_rejected or c.is_near_miss]),
                phase="idle",
                llm_queue=queue_s,
                source_health=get_source_health(),
                sweep_interval=cfg.pipeline.sweep_interval,
            )

            await persist_checkpoints(store)

            elapsed = time.monotonic() - sweep_start
            if telegram_agent.is_configured:
                await telegram_agent.send_sweep_summary(
                    sweep,
                    len([c for c in matched if c.is_accepted]),
                    len(all_observations),
                    elapsed,
                )

            gc.collect()
            if os.environ.get("OVERNIGHT_LOOP", "true").lower() != "true":
                break
            await asyncio.sleep(cfg.pipeline.sweep_interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Radar sweep {sweep} crashed", exc=e)
            set_pipeline_state(last_error=str(e), phase="crashed")
            await asyncio.sleep(cfg.pipeline.sweep_interval)

    await engine.shutdown(drain=False)
    await telegram_agent.stop_polling()
    set_pipeline_state(running=False, phase="shutdown")
    await bus.shutdown(timeout=5.0)
    await ctx.aclose()
    await _close_http_clients()
    await graph.close()
    await store.close()
    logger.info("Radar pipeline shutdown complete")


def _candidate_to_job_card(candidate: JobCandidate) -> dict[str, Any]:
    return {
        "role": candidate.normalized_role,
        "company": candidate.normalized_company,
        "match_percent": candidate.match_percent,
        "shortlist_probability": candidate.shortlist_probability,
        "salary": candidate.salary.raw if candidate.salary else None,
        "location": candidate.normalized_location,
        "apply_link": candidate.direct_apply_url,
        "jd_summary": candidate.jd_summary,
        "company_description": candidate.company_description,
        "founders": candidate.founders,
        "funding_stage": candidate.funding_stage,
        "funding_info": candidate.funding_info,
        "osint_signals": candidate.osint_signals,
    }


def run() -> None:
    load_dotenv()
    cfg = get_config()
    problems = cfg.validate()
    if problems:
        for p in problems:
            logger.warning(f"Config problem: {p}")
    asyncio.run(_run_radar_pipeline())


if __name__ == "__main__":
    run()
