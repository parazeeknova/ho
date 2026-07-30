"""RadarOrchestrator: source-first incremental job radar pipeline."""

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
from src.radar.boards import get_company_boards
from src.radar.extractors import extract_github_index_markdown
from src.radar.gates import run_gates
from src.radar.models import (
    FreshnessLane,
    JobCandidate,
    JobObservation,
)
from src.radar.outreach import generate_outreach_card
from src.radar.queue import enqueue_candidate, get_queue_status, process_queue
from src.radar.sources import (
    diff_snapshots,
    get_source_health,
    load_checkpoints,
    persist_checkpoints,
    record_failure,
    record_success,
    register_source,
    should_poll,
)
from src.rag.loader import load_resume
from src.search.searcher import GITHUB_INDEXES

console = Console()
logger = get_logger("radar_orchestrator")

# ── Source registry (loaded from boards.py) ──────────────────────────
_COMPANY_BOARDS = get_company_boards()


def _make_posting_id(obs: JobObservation) -> str:
    return obs.canonical_url_hash()


# ── Source polling ───────────────────────────────────────────────────


async def _scrape_indexes() -> list[JobObservation]:
    import httpx

    all_obs: list[JobObservation] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx_url in GITHUB_INDEXES:
            source_id = f"github:{idx_url.rsplit('/', 1)[-1]}"
            if not should_poll(source_id):
                continue
            try:
                resp = await client.get(idx_url)
                if resp.status_code == 200:
                    obs = extract_github_index_markdown(resp.text, idx_url)
                    for o in obs:
                        o.source = f"github_index:{idx_url.rsplit('/', 1)[-1]}"
                    all_obs.extend(obs)
                    record_success(source_id, len(obs), len(obs))
                else:
                    record_failure(source_id)
            except Exception:
                record_failure(source_id)
    return all_obs


async def _poll_company_board(
    board: dict[str, str],
    app: FirecrawlApp,
) -> list[JobObservation]:
    source_id = board["id"]
    if not should_poll(source_id):
        return []

    direct_urls: list[str] = []
    try:
        resp = app.map_url(board["url"])
        if isinstance(resp, list):
            for item in resp:
                url = item if isinstance(item, str) else item.get("url", "")
                if url and url.startswith("http"):
                    direct_urls.append(url)
        elif isinstance(resp, dict):
            for link in resp.get("links", []) or []:
                if isinstance(link, str) and link.startswith("http"):
                    direct_urls.append(link)
    except Exception as e:
        record_failure(source_id)
        logger.warning("Board poll failed", source=source_id, exception=str(e))
        return []

    if not direct_urls:
        record_success(source_id, 0, 0)
        return []

    state = diff_snapshots(source_id, direct_urls)
    new_urls = state.new_urls

    observations: list[JobObservation] = []
    for url in new_urls:
        obs = JobObservation(
            url=url,
            source=source_id,
            title="",
            snippet="",
            extra={"is_snapshot_delta": True},
        )
        observations.append(obs)

    record_success(source_id, len(new_urls), len(new_urls))
    return observations


# ── Posting fetch + gates ────────────────────────────────────────────


async def _fetch_postings_and_gate(
    observations: list[JobObservation],
    store: MemoryStore,
) -> tuple[list[JobCandidate], dict[str, Any]]:
    import time as _time

    import httpx

    cfg = get_config().firecrawl
    passed: list[JobCandidate] = []
    rejected_count = 0
    gate_stats: dict[str, int] = {}

    known_hashes: set[str] = set()
    last_seen: dict[str, float] = {}
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch("SELECT url_hash, last_seen FROM job_observations")
            for r in rows:
                known_hashes.add(r["url_hash"])
                if r["last_seen"]:
                    last_seen[r["url_hash"]] = float(r["last_seen"])
    except Exception:
        pass

    sem = asyncio.Semaphore(6)

    async def _process_one(obs: JobObservation) -> None:
        nonlocal rejected_count
        async with sem:
            try:
                if not obs.raw_markdown or len(obs.raw_markdown) < 100:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        resp = await client.post(
                            f"{cfg.url}/v1/scrape",
                            json={
                                "url": obs.url,
                                "formats": ["markdown"],
                                "onlyMainContent": True,
                            },
                        )
                        if resp.status_code != 200:
                            return
                        md = (resp.json().get("data") or {}).get("markdown", "") or ""
                        if not md or len(md) < 100:
                            return
                        obs.raw_markdown = md

                now_ts = _time.time()
                obs.observed_at = now_ts

                posting_id = _make_posting_id(obs)
                await _persist_observation(store, obs, posting_id)

                candidate, rejections = await run_gates(obs, known_hashes, last_seen)

                if candidate is not None:
                    candidate.extra["raw_markdown"] = obs.raw_markdown
                    candidate.extra["version"] = 1
                    candidate.extra["posting_id"] = posting_id
                    candidate.canonical_id = posting_id
                    passed.append(candidate)
                else:
                    rejected_count += 1
                    for _g, reason, _desc in rejections:
                        key = reason.value
                        gate_stats[key] = gate_stats.get(key, 0) + 1
                    await _persist_rejected_observation(
                        store,
                        obs,
                        posting_id,
                        rejections,
                    )

            except Exception:
                pass

    tasks = [asyncio.create_task(_process_one(o)) for o in observations]
    await asyncio.gather(*tasks, return_exceptions=True)
    return passed, {"rejected": rejected_count, "gate_stats": gate_stats}


async def _persist_observation(
    store: MemoryStore,
    obs: JobObservation,
    posting_id: str,
) -> None:
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO job_observations (url_hash, url, source, title, snippet,
                    first_seen, last_seen, freshness_lane, direct_posting_verified, raw_json)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                ON CONFLICT (url_hash) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen
                """,
                posting_id,
                obs.url,
                obs.source,
                obs.title or "",
                obs.snippet or "",
                obs.observed_at,
                obs.observed_at,
                "review",
                not obs.source.startswith("github_index:"),
                "{}",
            )
    except Exception:
        pass


async def _persist_rejected_observation(
    store: MemoryStore,
    obs: JobObservation,
    posting_id: str,
    rejections: list[tuple[str, Any, str]],
) -> None:
    try:
        reason = rejections[0][1] if rejections else None
        detail = rejections[0][2] if rejections else ""
        reason_str = reason.value if reason else "unknown"
        async with store._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO radar_candidates (canonical_id, source, direct_apply_url,
                    normalized_company, normalized_role, normalized_location,
                    eligibility, rejection_reason, rejection_detail,
                    freshness_lane, source_confidence, first_seen, last_seen)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$12)
                ON CONFLICT (canonical_id) DO NOTHING
                """,
                f"rejected:{posting_id}",
                obs.source,
                obs.url,
                obs.title or "unknown",
                "",
                "Remote",
                "rejected",
                reason_str,
                detail[:200],
                "review",
                0.3,
                obs.observed_at,
            )
    except Exception:
        pass


# ── Queue ranking ────────────────────────────────────────────────────


def _rank_for_queue(candidates: list[JobCandidate]) -> list[JobCandidate]:
    urgent: list[JobCandidate] = []
    high_fit: list[JobCandidate] = []
    review: list[JobCandidate] = []

    for c in candidates:
        if c.is_urgent:
            urgent.append(c)
        elif c.role_family and c.role_family.value not in ("unknown",):
            high_fit.append(c)
        else:
            review.append(c)

    urgent.sort(key=lambda c: c.source_confidence, reverse=True)
    high_fit.sort(key=lambda c: c.source_confidence, reverse=True)
    return urgent + high_fit + review


# ── Post-LLM enrichment ──────────────────────────────────────────────


async def _enrich_high_fit(
    candidates: list[JobCandidate],
    startup_agent: StartupAgent,
    store: MemoryStore,
) -> None:
    for c in candidates:
        if not c.is_accepted or not (c.is_urgent or c.match_percent >= 60):
            continue
        try:
            enriched = await startup_agent.analyze_startup(
                {
                    "role": c.normalized_role,
                    "company": c.normalized_company,
                    "match_percent": c.match_percent,
                    "verdict": c.verdict,
                }
            )
            c.founders = enriched.get("founders", [])
            c.funding_stage = enriched.get("funding_stage", "")
            c.funding_info = enriched.get("funding_info", {})
            c.founder_socials = enriched.get("founder_socials", [])
            c.company_news = enriched.get("company_news", "")
            c.osint_signals = enriched.get("osint_signals", [])
            await _persist_candidate_full(store, c)
        except Exception:
            pass


async def _persist_candidate_full(
    store: MemoryStore,
    candidate: JobCandidate,
) -> None:
    try:
        data: dict[str, Any] = {
            "canonical_id": candidate.canonical_id,
            "source": candidate.source,
            "direct_apply_url": candidate.direct_apply_url,
            "normalized_company": candidate.normalized_company,
            "normalized_role": candidate.normalized_role,
            "normalized_location": candidate.normalized_location,
            "freshness_lane": candidate.freshness_lane.name.lower(),
            "source_confidence": candidate.source_confidence,
            "eligibility": candidate.eligibility.name.lower(),
            "rejection_reason": (
                candidate.rejection_reason.value if candidate.rejection_reason else ""
            ),
            "role_family": candidate.role_family.value,
            "salary_amount": candidate.salary.amount if candidate.salary else None,
            "salary_currency": candidate.salary.currency if candidate.salary else "",
            "salary_period": candidate.salary.period if candidate.salary else "",
            "salary_raw": candidate.salary.raw if candidate.salary else "",
            "posted_date": candidate.posted_date or "",
            "first_seen": candidate.first_seen,
            "last_seen": candidate.last_seen,
            "matching_skills": candidate.matching_skills,
            "missing_skills": candidate.missing_skills,
            "match_percent": candidate.match_percent,
            "shortlist_probability": candidate.shortlist_probability,
            "verdict": candidate.verdict,
            "jd_summary": candidate.jd_summary,
            "company_description": candidate.company_description,
            "role_summary": candidate.role_summary,
            "is_remote": candidate.is_remote,
            "founders": candidate.founders,
            "funding_stage": candidate.funding_stage,
            "funding_info": candidate.funding_info,
            "founder_socials": candidate.founder_socials,
            "company_news": candidate.company_news,
            "osint_signals": candidate.osint_signals,
            "extra": candidate.extra,
        }
        await store.upsert_radar_candidate(data)
    except Exception:
        pass


# ── Telegram notification ────────────────────────────────────────────


async def _notify_telegram(
    telegram_agent: TelegramAgent,
    matched: list[JobCandidate],
    store: MemoryStore,
) -> None:
    if not telegram_agent.is_configured:
        return

    notified_keys: set[str] = set()
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch("SELECT dedup_key FROM telegram_notified_jobs")
            notified_keys.update(r["dedup_key"] for r in rows)
    except Exception:
        pass

    def _mk_key(c: JobCandidate) -> str:
        return c.extra.get("posting_id", c.canonical_id)

    urgent = [
        c for c in matched if c.is_urgent and c.is_accepted and _mk_key(c) not in notified_keys
    ]
    startup_sig = [
        c
        for c in matched
        if c.is_accepted and c.funding_stage and _mk_key(c) not in notified_keys and c not in urgent
    ]
    eligible = [
        c
        for c in matched
        if c.is_accepted
        and not c.is_urgent
        and _mk_key(c) not in notified_keys
        and c not in startup_sig
    ]
    review = [
        c
        for c in matched
        if c.freshness_lane == FreshnessLane.REVIEW and _mk_key(c) not in notified_keys
    ]

    async def _notify(
        category: str,
        candidates: list[JobCandidate],
    ) -> None:
        for c in candidates:
            key = _mk_key(c)
            card = _candidate_to_job_card(c)
            try:
                ok = await telegram_agent.send_categorized_alert(
                    category,
                    card,
                    dedup_key=key,
                )
                if ok:
                    notified_keys.add(key)
                    async with store._pool.acquire() as conn:
                        await conn.execute(
                            "INSERT INTO telegram_notified_jobs "
                            "(dedup_key, role, company) "
                            "VALUES ($1,$2,$3) "
                            "ON CONFLICT (dedup_key) DO NOTHING",
                            key,
                            c.normalized_role,
                            c.normalized_company,
                        )
            except Exception:
                pass

    await _notify("urgent", urgent)
    await _notify("startup_signal", startup_sig[:10])

    if eligible:
        dig = [_candidate_to_job_card(c) for c in eligible[:5]]
        await telegram_agent.send_category_digest("eligible", dig)
    if review:
        drv = [_candidate_to_job_card(c) for c in review[:5]]
        await telegram_agent.send_category_digest("review", drv)

    # Startup-signal digest (funding, launch, hiring signals)
    startup_digest = []
    for c in matched:
        if c.is_accepted and (c.funding_stage or c.extra.get("launch_signals")):
            card = _candidate_to_job_card(c)
            card["signal_type"] = "funding" if c.funding_stage else "launch"
            startup_digest.append(card)
    if startup_digest and not any(
        s in notified_keys for s in [d["company"] for d in startup_digest]
    ):
        await telegram_agent.send_category_digest(
            "startup_signal",
            startup_digest[:5],
        )


async def _process_outreach_cards(
    candidates: list[JobCandidate],
    telegram_agent: TelegramAgent,
    store: MemoryStore,
) -> None:
    if not telegram_agent.is_configured:
        return

    notified: set[str] = set()
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch("SELECT dedup_key FROM telegram_notified_jobs")
            notified.update(r["dedup_key"] for r in rows)
    except Exception:
        pass

    for c in candidates:
        if not c.is_accepted or (not c.founders and not c.funding_stage):
            continue
        card = generate_outreach_card(c)
        if card is None or card.confidence < 0.4:
            continue
        dedup = f"outreach:{c.canonical_id}"
        if dedup in notified:
            continue
        try:
            ok = await telegram_agent.send_categorized_alert(
                "outreach",
                _candidate_to_job_card(c),
                dedup_key=dedup,
            )
            if ok:
                async with store._pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO telegram_notified_jobs "
                        "(dedup_key, role, company) "
                        "VALUES ($1,$2,$3) "
                        "ON CONFLICT (dedup_key) DO NOTHING",
                        dedup,
                        f"Outreach to {c.normalized_company}",
                        c.normalized_company,
                    )
        except Exception:
            pass


# ── Company event dispatch for graph expansion ───────────────────────


async def _dispatch_company_events(
    candidates: list[JobCandidate],
    graph: GraphStore,
    bus: EventBus,
) -> None:
    from src.graph.entity import company_node as _cn

    seen_companies: set[str] = set()
    for c in candidates:
        if not c.is_accepted and not c.is_near_miss:
            continue
        company = c.normalized_company.lower().strip()
        if not company or company in ("unknown", "n/a", ""):
            continue
        if company in seen_companies:
            continue
        seen_companies.add(company)
        try:
            node = await graph.get_node(company)
            if node is None:
                node = _cn(company, source="radar")
                node, _ = await graph.upsert_node(node)
            await bus.fire(
                bus.new_event(
                    "company_discovered",
                    node.id,
                    NodeType.COMPANY,
                    {"name": c.normalized_company, "url": c.direct_apply_url},
                )
            )
        except Exception:
            pass


# ── Graph event handlers ────────────────────────────────────────────


async def _founder_miner(
    entry: FrontierEntry,
    graph: GraphStore,
    bus: EventBus,
    startup_agent: StartupAgent,
) -> list[FrontierEntry]:
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
            _, _ = await graph.upsert_edge(
                edge(entry.node_id, EdgeType.FOUNDED_BY, f_node.id),
            )
            await bus.fire(
                bus.new_event(
                    "founder_discovered",
                    f_node.id,
                    NodeType.FOUNDER,
                    {"name": f["name"], "company": cn},
                )
            )
    return results


async def _outreach_handler(entry: FrontierEntry) -> list[FrontierEntry]:
    company = entry.payload.get("company", "")
    founder_name = entry.payload.get("founder_name", "")
    linkedin_url = entry.payload.get("linkedin", "")
    verified_posts = entry.payload.get("verified_posts", [])

    if not company or not (founder_name or linkedin_url):
        return []

    candidate = JobCandidate(
        canonical_id=f"outreach:{company}:{founder_name}",
        source="outreach_generator",
        direct_apply_url=linkedin_url,
        normalized_company=company,
        normalized_role="",
        normalized_location="Remote",
        founders=[{"name": founder_name, "linkedin_url": linkedin_url}],
        funding_stage=entry.payload.get("funding_stage", ""),
        extra={
            "verified_posts": verified_posts,
            "hiring_signals": verified_posts,
        },
    )
    card = generate_outreach_card(candidate)
    if card and card.confidence >= 0.4:
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


# ── Main pipeline ────────────────────────────────────────────────────


async def _run_radar_pipeline() -> None:
    cfg = get_config()

    ctx = ContextManager()
    telegram_agent = TelegramAgent(ctx=ctx)

    shutdown_requested = asyncio.Event()

    def _cleanup(signum: int, frame: object) -> None:
        logger.info("Interrupted", extra={"signal": signum})
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

    # Register sources
    for board in _COMPANY_BOARDS:
        register_source(board["id"], "ats_board", initial_quality=0.6)
    for idx_url in GITHUB_INDEXES:
        register_source(
            f"github:{idx_url.rsplit('/', 1)[-1]}",
            "github_index",
            initial_quality=0.3,
        )

    # Graph event subscriptions
    async def _sub_company(event):
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
            ),
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

    async def _sub_founder(event):
        d = event.payload
        return [
            FrontierEntry(
                id=make_work_id("founder_social_osint", event.node_id),
                agent="founder_social_osint",
                node_id=event.node_id,
                node_type=NodeType.FOUNDER,
                priority=50,
                depth=2,
                payload={
                    "founder_name": d.get("name", ""),
                    "company": d.get("company", ""),
                },
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

    async def _sub_career(event):
        url = event.payload.get("url", "")
        ats_domains = (
            "greenhouse",
            "lever.co",
            "ashbyhq",
            "workable",
            "myworkdayjobs",
        )
        if any(a in url.lower() for a in ats_domains):
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

    bus.subscribe("company_discovered", _sub_company)
    bus.subscribe("founder_discovered", _sub_founder)
    bus.subscribe("career_site_discovered", _sub_career)

    engine.register_agent(
        "founder_miner",
        lambda e: _founder_miner(e, graph, bus, startup_agent),
    )
    engine.register_agent("career_site_detector", career_site_detector)
    engine.register_agent("founder_social_osint", founder_social_agent)
    engine.register_agent("employee_discovery", employee_discovery_agent)
    engine.register_agent("ats_crawler", _ats_crawler_ingest)
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

            # GitHub indexes (discovery only, labelled as index-derived)
            idx_obs = await _scrape_indexes()
            all_observations.extend(idx_obs)
            logger.info(f"GitHub indexes: {len(idx_obs)} observations")

            # Company-specific ATS boards
            for board in _COMPANY_BOARDS:
                if not should_poll(board["id"]):
                    continue
                board_obs = await _poll_company_board(board, app)
                all_observations.extend(board_obs)
                if board_obs:
                    logger.info(
                        f"Board {board['id']}: {len(board_obs)} new URLs",
                    )

            # Fetch + gate
            candidates, gate_stats = await _fetch_postings_and_gate(
                all_observations,
                store,
            )
            set_pipeline_state(
                scraped=len(all_observations),
                gated=len(candidates),
            )
            logger.info(
                f"Gating: {len(candidates)} passed, {gate_stats['rejected']} rejected",
            )

            # LLM matching
            console.rule(
                f"[bold cyan]RADAR PHASE 3 (sweep {sweep}): LLM Matching[/bold cyan]",
            )
            resume_ctx = full_text[:3000] if full_text else candidate_persona

            ranked = _rank_for_queue(candidates)
            for c in ranked:
                prio = 80 if c.is_urgent else 60 if c.role_family.value != "unknown" else 40
                await enqueue_candidate(c, priority=prio)

            matched = await process_queue(
                ctx,
                resume_ctx,
                candidate_persona,
                store,
                max_candidates=cfg.radar.max_candidates_per_sweep,
            )
            logger.info(f"LLM queue: {len(matched)} matched")

            await _enrich_high_fit(matched, startup_agent, store)
            await _dispatch_company_events(matched, graph, bus)
            await _notify_telegram(telegram_agent, matched, store)
            await _process_outreach_cards(matched, telegram_agent, store)

            # Sweep counters
            accepted = len([c for c in matched if c.is_accepted])
            near_miss = len([c for c in matched if c.is_near_miss])
            rejected_llm = len([c for c in matched if c.is_rejected])
            queue_s = get_queue_status()

            set_pipeline_state(
                matched_total=accepted,
                rejected_total=rejected_llm + near_miss + gate_stats["rejected"],
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
                    accepted,
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


# ── ATS crawler ingest: routes ats_crawler output to posting-ingestion ──


async def _ats_crawler_ingest(entry: FrontierEntry) -> list[FrontierEntry]:
    return await ats_crawler(entry)


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
