"""RadarOrchestrator: source-first, globally-scoped, high-pay underdog job radar."""

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
from src.radar.discovery import (
    _extract_domain as _discovery_domain,
)
from src.radar.discovery import (
    _resolve_company_domain,
    detect_ats_for_company,
    discover_from_hackernews,
    discover_from_searxng,
    discover_from_vc_portfolios,
    discover_from_wellfound,
    discover_from_yc,
)
from src.radar.extractors import extract_github_index_markdown
from src.radar.gates import run_gates
from src.radar.models import JobCandidate, JobObservation
from src.radar.outreach import generate_outreach_card
from src.radar.queue import enqueue_candidate, get_queue_status, process_queue
from src.radar.scoring import compute_underdog_score
from src.radar.signals import extract_signals
from src.radar.sources import (
    diff_snapshots,
    get_checkpoint,
    get_source_health,
    load_active_sources,
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

_SEED_BOARDS = [
    ("openai:greenhouse", "https://boards.greenhouse.io/openai", "greenhouse"),
    ("anthropic:ashby", "https://jobs.ashbyhq.com/anthropic", "ashby"),
    ("stripe:greenhouse", "https://boards.greenhouse.io/stripe", "greenhouse"),
    ("airbnb:greenhouse", "https://boards.greenhouse.io/airbnb", "greenhouse"),
    ("ycombinator:jobs", "https://www.ycombinator.com/jobs", "careers_page"),
    ("wellfound:jobs", "https://wellfound.com/jobs", "careers_page"),
]

_SCHEDULER_ERRORS: dict[str, int] = {}
_PIPELINE_METRICS: dict[str, Any] = {
    "dropped_postings": 0,
    "failed_fetches": 0,
    "failed_source_persistence": 0,
    "unknown_agents": 0,
    "job_processor_invocations": 0,
    "job_processor_success": 0,
    "job_processor_failures": 0,
}

_DISCOVERY_METRICS: dict[str, int] = {}


def _posting_id(obs: JobObservation) -> str:
    return obs.canonical_url_hash()


# ── Source persistence ───────────────────────────────────────────────


def _save_source_checkpoint(source_id: str, board_url: str, origin: str = "discovery") -> None:
    cp = register_source(source_id, "ats_board", initial_quality=0.5)
    cp.board_url = board_url
    cp.discovery_origin = origin


async def _persist_discovered_sources(
    store: MemoryStore,
    discovered: list[dict[str, Any]],
) -> int:
    """Persist discovered boards with their URLs. Returns count added."""
    count = 0
    for c in discovered:
        website = c.get("website", "")
        source_id = f"discovered:{c.get('name', '').lower().replace(' ', '-')[:60]}"
        try:
            register_source(source_id, "ats_board", initial_quality=0.5)
            cp = get_checkpoint(source_id)
            cp.board_url = c.get("ats_url", website)
            cp.company_name = c.get("name", "")
            cp.discovery_origin = c.get("source", "")
            cp.active = True
            count += 1
        except Exception:
            _PIPELINE_METRICS["failed_source_persistence"] += 1
    return count


# ── Company discovery ────────────────────────────────────────────────


async def _discover_new_companies() -> list[dict[str, Any]]:
    logger.info("Starting company discovery sweep")
    _DISCOVERY_METRICS["sweeps"] = _DISCOVERY_METRICS.get("sweeps", 0) + 1
    results: list[dict[str, Any]] = []
    adapters = [
        ("yc", discover_from_yc, 30),
        ("vc", discover_from_vc_portfolios, 40),
        ("wellfound", discover_from_wellfound, 30),
        ("hn", discover_from_hackernews, 30),
    ]
    searx_kinds = ["hiring", "funding", "launch"]

    for adp_name, func, limit in adapters:
        try:
            companies = await func(limit)
            for c in companies:
                c["discovered_from"] = adp_name
            results.extend(companies)
            logger.info(f"Discovery {adp_name}: {len(companies)} companies")
        except Exception as e:
            logger.warning(f"Discovery adapter {adp_name} failed", exception=str(e))
            _DISCOVERY_METRICS[f"failed_{adp_name}"] += 1

    for kind in searx_kinds:
        try:
            companies = await discover_from_searxng(kind)
            for c in companies:
                c["discovered_from"] = f"searxng_{kind}"
            results.extend(companies)
            logger.info(f"Discovery searxng/{kind}: {len(companies)} companies")
        except Exception as e:
            logger.warning(f"Discovery searxng/{kind} failed", exception=str(e))

    # Deduplicate by domain
    seen_domains: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in results:
        domain = _discovery_domain(c.get("website", ""))
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            deduped.append(c)
        elif not domain and c["name"] not in {d["name"] for d in deduped}:
            deduped.append(c)

    new_sources = 0
    for c in deduped:
        website = c.get("website", "")
        if (
            (not website or not website.startswith("http"))
            and c.get("name")
            and (domain := await _resolve_company_domain(c["name"]))
        ):
            website = f"https://{domain}"
            c["website"] = website
        if not website or not website.startswith("http"):
            _DISCOVERY_METRICS["no_domain"] = _DISCOVERY_METRICS.get("no_domain", 0) + 1
            continue
        ats_url = await detect_ats_for_company(website)
        if ats_url:
            c["ats_url"] = ats_url
            new_sources += 1
            _DISCOVERY_METRICS["ats_verified"] = _DISCOVERY_METRICS.get("ats_verified", 0) + 1

    _DISCOVERY_METRICS["companies_found"] = _DISCOVERY_METRICS.get("companies_found", 0) + len(
        deduped
    )
    _DISCOVERY_METRICS["domain_resolved"] = _DISCOVERY_METRICS.get("domain_resolved", 0) + len(
        [c for c in deduped if c.get("website") and c["website"].startswith("http")]
    )
    _DISCOVERY_METRICS["sources_added"] = _DISCOVERY_METRICS.get("sources_added", 0) + new_sources

    logger.info(f"Discovery: {len(deduped)} companies, {new_sources} ATS detected")
    return deduped


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


async def _poll_board(board: dict[str, str], app: FirecrawlApp) -> list[JobObservation]:
    source_id = board["id"]
    board_url = board["url"]
    if not should_poll(source_id):
        return []
    if not board_url or not board_url.startswith("http"):
        return []

    direct_urls: list[str] = []
    try:
        resp = app.map_url(board_url)
        if isinstance(resp, list):
            for item in resp:
                url = item if isinstance(item, str) else item.get("url", "")
                if url and url.startswith("http"):
                    direct_urls.append(url)
        elif isinstance(resp, dict):
            for link in resp.get("links", []) or []:
                if isinstance(link, str) and link.startswith("http"):
                    direct_urls.append(link)
    except Exception:
        record_failure(source_id)
        return []

    if not direct_urls:
        record_success(source_id, 0, 0)
        return []

    had_prior = bool(get_checkpoint(source_id).last_snapshot_hash)
    state = diff_snapshots(source_id, direct_urls)
    new_urls = state.new_urls

    observations: list[JobObservation] = []
    for url in new_urls:
        observations.append(
            JobObservation(
                url=url,
                source=source_id,
                title="",
                snippet="",
                extra={
                    "is_snapshot_delta": had_prior,
                    "official_source": True,
                    "source_type": "ats_board",
                },
                source_freshness_evidence=None,
            )
        )
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
                            json={"url": obs.url, "formats": ["markdown"], "onlyMainContent": True},
                        )
                        if resp.status_code != 200:
                            _PIPELINE_METRICS["failed_fetches"] += 1
                            return
                        md = (resp.json().get("data") or {}).get("markdown", "") or ""
                        if not md or len(md) < 100:
                            _PIPELINE_METRICS["dropped_postings"] += 1
                            return
                        obs.raw_markdown = md

                now_ts = _time.time()
                obs.observed_at = now_ts
                pid = _posting_id(obs)

                candidate, rejections = await run_gates(obs, known_hashes, last_seen)
                if candidate is not None:
                    candidate.extra["raw_markdown"] = obs.raw_markdown
                    candidate.extra["version"] = 1
                    candidate.extra["posting_id"] = pid
                    candidate.canonical_id = pid

                    # Pre-LLM signal extraction
                    sigs = extract_signals(obs.raw_markdown, obs.title)
                    if sigs["salary"]:
                        candidate.salary = sigs["salary"]
                        candidate.salary_annual_usd = sigs["salary_annual_usd"]
                    candidate.sponsors_visa = sigs["sponsors_visa"]
                    candidate.is_remote = sigs["is_remote"]

                    if obs.extra.get("is_snapshot_delta"):
                        candidate.extra["is_snapshot_delta"] = True
                    passed.append(candidate)
                else:
                    rejected_count += 1
                    for _g, reason, _desc in rejections:
                        gate_stats[reason.value] = gate_stats.get(reason.value, 0) + 1

                # Persist observation after gating (so gate decisions use DB state)
                await _persist_observation(store, obs, pid)

                # Persist rejected observations too
                if candidate is None:
                    await _persist_rejected(store, obs, pid, rejections)

            except Exception:
                _PIPELINE_METRICS["failed_fetches"] += 1

    tasks = [asyncio.create_task(_process_one(o)) for o in observations]
    await asyncio.gather(*tasks, return_exceptions=True)
    return passed, {"rejected": rejected_count, "gate_stats": gate_stats}


async def _persist_observation(store: MemoryStore, obs: JobObservation, posting_id: str) -> None:
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO job_observations (url_hash, url, source, title, snippet,
                    first_seen, last_seen, freshness_lane, direct_posting_verified)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (url_hash) DO UPDATE SET last_seen = EXCLUDED.last_seen""",
                posting_id,
                obs.url,
                obs.source,
                obs.title or "",
                obs.snippet or "",
                obs.observed_at,
                obs.observed_at,
                "review",
                not obs.source.startswith("github_index:"),
            )
    except Exception:
        pass


async def _persist_rejected(
    store: MemoryStore,
    obs: JobObservation,
    pid: str,
    rejections: list[tuple[str, Any, str]],
) -> None:
    try:
        reason = rejections[0][1] if rejections else None
        detail = rejections[0][2] if rejections else ""
        reason_str = reason.value if reason else "unknown"
        async with store._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO radar_candidates (canonical_id, source, direct_apply_url,
                    normalized_company, eligibility, rejection_reason, rejection_detail,
                    freshness_lane, first_seen, last_seen, role_family)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9,'unknown')
                ON CONFLICT (canonical_id) DO NOTHING""",
                f"rejected:{pid}",
                obs.source,
                obs.url,
                obs.title or "unknown",
                "rejected",
                reason_str,
                detail[:200],
                "review",
                obs.observed_at,
            )
    except Exception:
        pass


# ── Queue ranking ────────────────────────────────────────────────────


def _rank_for_queue(candidates: list[JobCandidate]) -> list[JobCandidate]:
    urgent_high: list[JobCandidate] = []
    urgent: list[JobCandidate] = []
    sponsor: list[JobCandidate] = []
    rest: list[JobCandidate] = []

    for c in candidates:
        c.underdog_score = compute_underdog_score(c)
        c.extra["group_key"] = _group_key(c)

        sal = c.salary_annual_usd or 0
        if c.is_urgent and sal >= 60000:
            urgent_high.append(c)
        elif c.is_urgent:
            urgent.append(c)
        elif c.sponsors_visa:
            sponsor.append(c)
        else:
            rest.append(c)

    def _sort(x: JobCandidate) -> float:
        return _sort_key(x)

    return (
        sorted(urgent_high, key=_sort, reverse=True)
        + sorted(urgent, key=_sort, reverse=True)
        + sorted(sponsor, key=_sort, reverse=True)
        + sorted(rest, key=_sort, reverse=True)
    )


def _sort_key(c: JobCandidate) -> float:
    sal = c.salary_annual_usd or 0
    return (
        (sal / 10000) * 0.4
        + c.match_percent * 0.3
        + c.underdog_score * 0.2
        + c.source_confidence * 0.1
    )


def _group_key(c: JobCandidate) -> str:
    from src.radar.models import make_canonical_id

    return make_canonical_id(c.normalized_company, c.normalized_role, c.normalized_location)


# ── Post-LLM enrichment ──────────────────────────────────────────────


async def _enrich_high_fit(
    candidates: list[JobCandidate], sa: StartupAgent, store: MemoryStore
) -> None:
    for c in candidates:
        if not c.is_accepted or not (c.is_urgent or c.match_percent >= 60):
            continue
        try:
            enriched = await sa.analyze_startup(
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
            c.underdog_score = compute_underdog_score(c)
            await _persist_full(store, c)
        except Exception:
            pass


async def _persist_full(store: MemoryStore, c: JobCandidate) -> None:
    try:
        data: dict[str, Any] = {
            "canonical_id": c.canonical_id,
            "source": c.source,
            "direct_apply_url": c.direct_apply_url,
            "normalized_company": c.normalized_company,
            "normalized_role": c.normalized_role,
            "normalized_location": c.normalized_location,
            "freshness_lane": c.freshness_lane.name.lower(),
            "source_confidence": c.source_confidence,
            "eligibility": c.eligibility.name.lower(),
            "rejection_reason": c.rejection_reason.value if c.rejection_reason else "",
            "role_family": c.role_family.value,
            "salary_amount": c.salary.amount if c.salary else None,
            "salary_currency": c.salary.currency if c.salary else "",
            "salary_period": c.salary.period if c.salary else "",
            "salary_raw": c.salary.raw if c.salary else "",
            "posted_date": c.posted_date or "",
            "first_seen": c.first_seen,
            "last_seen": c.last_seen,
            "matching_skills": c.matching_skills,
            "missing_skills": c.missing_skills,
            "match_percent": c.match_percent,
            "shortlist_probability": c.shortlist_probability,
            "verdict": c.verdict,
            "jd_summary": c.jd_summary,
            "company_description": c.company_description,
            "role_summary": c.role_summary,
            "is_remote": c.is_remote,
            "founders": c.founders,
            "funding_stage": c.funding_stage,
            "funding_info": c.funding_info,
            "founder_socials": c.founder_socials,
            "company_news": c.company_news,
            "osint_signals": c.osint_signals,
            "extra": c.extra,
        }
        await store.upsert_radar_candidate(data)
    except Exception:
        pass


# ── Telegram ─────────────────────────────────────────────────────────


async def _notify_telegram(
    ta: TelegramAgent, matched: list[JobCandidate], store: MemoryStore
) -> None:
    if not ta.is_configured:
        return
    notified: set[str] = set()
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch("SELECT dedup_key FROM telegram_notified_jobs")
            notified.update(r["dedup_key"] for r in rows)
    except Exception:
        pass

    def _pid(c: JobCandidate) -> str:
        return c.extra.get("posting_id", c.canonical_id)

    urgent = [
        c
        for c in matched
        if c.is_urgent
        and c.is_accepted
        and c.salary_annual_usd
        and c.salary_annual_usd >= 60000
        and _pid(c) not in notified
    ]
    underdog = [
        c
        for c in matched
        if c.is_accepted and c.underdog_score >= 0.6 and _pid(c) not in notified and c not in urgent
    ]
    sponsor = [
        c
        for c in matched
        if c.is_accepted
        and c.sponsors_visa
        and _pid(c) not in notified
        and c not in urgent
        and c not in underdog
    ]
    startup = [
        c
        for c in matched
        if c.is_accepted
        and c.funding_stage
        and _pid(c) not in notified
        and c not in urgent
        and c not in underdog
    ]

    async def _notify(cat: str, candidates: list[JobCandidate]) -> None:
        for c in candidates:
            key = _pid(c)
            try:
                ok = await ta.send_categorized_alert(cat, _card(c), dedup_key=key)
                if ok:
                    notified.add(key)
                    async with store._pool.acquire() as conn:
                        await conn.execute(
                            "INSERT INTO telegram_notified_jobs (dedup_key, role, company) "
                            "VALUES ($1,$2,$3) ON CONFLICT (dedup_key) DO NOTHING",
                            key,
                            c.normalized_role,
                            c.normalized_company,
                        )
            except Exception:
                pass

    await _notify("urgent", urgent)
    await _notify("outreach", underdog[:15])
    await _notify("eligible", sponsor[:10])
    await _notify("startup_signal", startup[:10])


def _card(c: JobCandidate) -> dict[str, Any]:
    return {
        "role": c.normalized_role,
        "company": c.normalized_company,
        "match_percent": c.match_percent,
        "shortlist_probability": c.shortlist_probability,
        "salary": c.salary.raw if c.salary else None,
        "salary_annual_usd": c.salary_annual_usd,
        "location": c.normalized_location,
        "apply_link": c.direct_apply_url,
        "jd_summary": c.jd_summary,
        "company_description": c.company_description,
        "founders": c.founders,
        "funding_stage": c.funding_stage,
        "funding_info": c.funding_info,
        "osint_signals": c.osint_signals,
        "sponsors_visa": c.sponsors_visa,
        "underdog_score": round(c.underdog_score, 2),
    }


# ── Graph handlers ───────────────────────────────────────────────────


async def _dispatch_company_events(
    candidates: list[JobCandidate], graph: GraphStore, bus: EventBus
) -> None:
    from src.graph.entity import company_node as _cn

    seen: set[str] = set()
    for c in candidates:
        if not c.is_accepted and not c.is_near_miss:
            continue
        company = c.normalized_company.lower().strip()
        if not company or company in ("unknown", "n/a", "") or company in seen:
            continue
        seen.add(company)
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


async def _founder_miner(
    entry: FrontierEntry, graph: GraphStore, bus: EventBus, sa: StartupAgent
) -> list[FrontierEntry]:
    cn = entry.payload.get("company", "")
    if not cn:
        return []
    enriched = await sa.analyze_startup(
        {"role": "Startup Analysis", "company": cn, "match_percent": 50, "verdict": "WEAK_MATCH"}
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
            fn = GraphNode(
                id=make_founder_id(f["name"], cn),
                node_type=NodeType.FOUNDER,
                data={**f, "company": cn},
            )
            fn, _ = await graph.upsert_node(fn)
            _, _ = await graph.upsert_edge(edge(entry.node_id, EdgeType.FOUNDED_BY, fn.id))
            await bus.fire(
                bus.new_event(
                    "founder_discovered",
                    fn.id,
                    NodeType.FOUNDER,
                    {"name": f["name"], "company": cn},
                )
            )
    return results


async def _outreach_handler(entry: FrontierEntry) -> list[FrontierEntry]:
    company, founder_name, linkedin_url = (
        entry.payload.get("company", ""),
        entry.payload.get("founder_name", ""),
        entry.payload.get("linkedin", ""),
    )
    verified_posts = entry.payload.get("verified_posts", [])
    if not company or not (founder_name or linkedin_url):
        return []
    c = JobCandidate(
        canonical_id=f"outreach:{company}:{founder_name}",
        source="outreach_generator",
        direct_apply_url=linkedin_url,
        normalized_company=company,
        normalized_role="",
        normalized_location="Remote",
        founders=[{"name": founder_name, "linkedin_url": linkedin_url}],
        funding_stage=entry.payload.get("funding_stage", ""),
        extra={"verified_posts": verified_posts, "hiring_signals": verified_posts},
    )
    card = generate_outreach_card(c)
    if card and card.confidence >= 0.4:
        ta = TelegramAgent()
        if ta.is_configured:
            await ta.send_categorized_alert(
                "outreach",
                {
                    "role": f"Outreach to {founder_name}",
                    "company": company,
                    "apply_link": linkedin_url,
                    "founders": c.founders,
                    "funding_stage": c.funding_stage,
                },
                dedup_key=f"outreach:{company}:{founder_name}",
            )
    return []


# ── job_processor ────────────────────────────────────────────────────


async def _job_processor(entry: FrontierEntry) -> list[FrontierEntry]:
    """Process ats_crawler output: fetch posting, load DB state, gate, enqueue."""
    import httpx

    _PIPELINE_METRICS["job_processor_invocations"] += 1
    url = entry.payload.get("observation_url", "")
    source = entry.payload.get("source", "unknown")
    company = entry.payload.get("company", "")
    if not url or not url.startswith("http"):
        logger.warning("job_processor: invalid URL", source=source, entry_id=entry.id)
        _PIPELINE_METRICS["dropped_postings"] += 1
        return []

    cfg = get_config().firecrawl
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{cfg.url}/v1/scrape",
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            if resp.status_code != 200:
                _PIPELINE_METRICS["job_processor_failures"] += 1
                return []
            md = (resp.json().get("data") or {}).get("markdown", "") or ""
            if not md or len(md) < 100:
                _PIPELINE_METRICS["dropped_postings"] += 1
                return []
    except Exception:
        _PIPELINE_METRICS["job_processor_failures"] += 1
        return []

    obs = JobObservation(
        url=url,
        source=source,
        raw_markdown=md,
        title=company or "",
        snippet=f"{company} — {url[:80]}",
    )
    pid = _posting_id(obs)

    store = await MemoryStore.create()
    try:
        # Load existing observations (NOT seed with own hash — that causes false duplicate)
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

        # Check if already seen
        if pid in known_hashes:
            logger.debug("job_processor: duplicate URL skipped", url=url)
            return []

        obs.observed_at = time.time()
        candidate, rejections = await run_gates(obs, known_hashes, last_seen)

        if candidate is not None:
            candidate.extra["raw_markdown"] = md
            candidate.extra["version"] = 1
            candidate.extra["posting_id"] = pid
            candidate.canonical_id = pid

            sigs = extract_signals(md, obs.title)
            if sigs["salary"]:
                candidate.salary = sigs["salary"]
                candidate.salary_annual_usd = sigs["salary_annual_usd"]
            candidate.sponsors_visa = sigs["sponsors_visa"]
            candidate.is_remote = sigs["is_remote"]

            await _persist_observation(store, obs, pid)
            await enqueue_candidate(candidate, priority=40)
            _PIPELINE_METRICS["job_processor_success"] += 1
        else:
            await _persist_observation(store, obs, pid)
            await _persist_rejected(store, obs, pid, rejections)
            _PIPELINE_METRICS["dropped_postings"] += 1
    finally:
        await store.close()

    return []


# ── Main pipeline ────────────────────────────────────────────────────


async def _run_radar_pipeline() -> None:
    cfg = get_config()
    ctx = ContextManager()
    ta = TelegramAgent(ctx=ctx)
    shutdown_requested = asyncio.Event()

    def _cleanup(sig, _frame):
        ctx._flush_sync()
        shutdown_requested.set()

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    await ctx.flush()
    app = FirecrawlApp(api_key="sk-no-auth", api_url=cfg.firecrawl.url)
    await ta.start_polling()

    store = await MemoryStore.create()
    await store.purge_fake_job_keys(["techco:backendengineer"])
    graph = await GraphStore.create()
    bus = EventBus()

    ecfg = get_config().scheduler
    frontier = CrawlFrontier(max_size=ecfg.max_queue_size)
    engine = WorkScheduler(frontier, worker_count=3)
    bus.set_enqueue_callback(engine.enqueue_many)
    sa = StartupAgent(ctx)
    await load_checkpoints(store)

    for id_, _url, _adapter in _SEED_BOARDS:
        cp = register_source(id_, "ats_board", initial_quality=0.6)
        cp.board_url = _url
    for idx_url in GITHUB_INDEXES:
        register_source(f"github:{idx_url.rsplit('/', 1)[-1]}", "github_index", initial_quality=0.3)

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

    async def _sub_career(event):
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

    bus.subscribe("company_discovered", _sub_company)
    bus.subscribe("founder_discovered", _sub_founder)
    bus.subscribe("career_site_discovered", _sub_career)

    engine.register_agent("founder_miner", lambda e: _founder_miner(e, graph, bus, sa))
    engine.register_agent("career_site_detector", career_site_detector)
    engine.register_agent("founder_social_osint", founder_social_agent)
    engine.register_agent("employee_discovery", employee_discovery_agent)
    engine.register_agent("ats_crawler", ats_crawler)
    engine.register_agent("job_processor", _job_processor)
    engine.register_agent("outreach_generator", _outreach_handler)
    engine.start(worker_count=3)

    console.rule("[bold cyan]RADAR PHASE 1: Load Resume[/bold cyan]")
    loop = asyncio.get_running_loop()
    existing_count = await store.chunk_count()
    full_text = ""
    if existing_count > 0:
        logger.info(f"Reusing {existing_count} existing resume chunks")
    else:
        from src.pipeline.orchestrator import _index_resume_in_pgvector

        full_text, chunks = await loop.run_in_executor(None, load_resume)
        await _index_resume_in_pgvector(chunks, store)

    candidate_persona = cfg.candidate.persona
    set_pipeline_state(
        running=True,
        started_at=time.time(),
        phase="starting",
        sweep=0,
        rejected_total=0,
        matched_total=0,
    )
    if ta.is_configured:
        await ta.send_startup(existing_count)

    sweep = 0
    last_discovery = 0.0
    while True:
        if shutdown_requested.is_set():
            break
        sweep += 1
        sweep_start = time.monotonic()
        set_pipeline_state(
            sweep=sweep, phase=f"sweep {sweep}: scraping", sweep_started_at=time.time()
        )

        try:
            console.rule(
                f"[bold cyan]RADAR PHASE 2 (sweep {sweep}): Source Polling + Gating[/bold cyan]"
            )

            # Dynamic discovery + ATS detection
            if sweep == 1 or time.monotonic() - last_discovery > cfg.radar.poll_low_freq_seconds:
                last_discovery = time.monotonic()
                discovered = await _discover_new_companies()
                await _persist_discovered_sources(store, discovered)
                await persist_checkpoints(store)  # Persist new sources immediately

            all_obs: list[JobObservation] = []
            all_obs.extend(await _scrape_indexes())

            # Load active sources (seeds + dynamically discovered, all with URLs)
            active_sources = await load_active_sources(store)
            # Also poll seed boards
            for id_, url, _adapter in _SEED_BOARDS:
                if should_poll(id_) and not any(s["id"] == id_ for s in active_sources):
                    active_sources.append({"id": id_, "url": url})

            for board in active_sources:
                board_obs = await _poll_board(board, app)
                all_obs.extend(board_obs)

            logger.info(
                f"Sources: {len(all_obs)} observations across {len(active_sources)} active sources"
            )

            candidates, gate_stats = await _fetch_postings_and_gate(all_obs, store)
            set_pipeline_state(scraped=len(all_obs), gated=len(candidates))
            logger.info(f"Gating: {len(candidates)} passed, {gate_stats['rejected']} rejected")

            console.rule(f"[bold cyan]RADAR PHASE 3 (sweep {sweep}): LLM Matching[/bold cyan]")
            resume_ctx = full_text[:3000] if full_text else candidate_persona

            ranked = _rank_for_queue(candidates)
            for c in ranked:
                sal = c.salary_annual_usd or 0
                if c.is_urgent and sal >= 60000:
                    prio = 90
                elif c.is_urgent:
                    prio = 80
                elif c.sponsors_visa:
                    prio = 70
                else:
                    prio = 50
                await enqueue_candidate(c, priority=prio)

            matched = await process_queue(
                ctx,
                resume_ctx,
                candidate_persona,
                store,
                max_candidates=cfg.radar.max_candidates_per_sweep,
            )
            logger.info(f"LLM queue: {len(matched)} matched")

            await _enrich_high_fit(matched, sa, store)
            await _dispatch_company_events(matched, graph, bus)
            await _notify_telegram(ta, matched, store)

            accepted = len([c for c in matched if c.is_accepted])
            rejected_llm = len([c for c in matched if c.is_rejected])
            near_miss = len([c for c in matched if c.is_near_miss])

            set_pipeline_state(
                matched_total=accepted,
                rejected_total=rejected_llm + near_miss + gate_stats["rejected"],
                phase="idle",
                llm_queue=get_queue_status(),
                source_health=get_source_health(),
                scheduler_errors=dict(_SCHEDULER_ERRORS),
                pipeline_metrics=dict(_PIPELINE_METRICS),
                discovery_metrics=dict(_DISCOVERY_METRICS),
                sweep_interval=cfg.pipeline.sweep_interval,
            )

            await persist_checkpoints(store)
            elapsed = time.monotonic() - sweep_start
            if ta.is_configured:
                await ta.send_sweep_summary(sweep, accepted, len(all_obs), elapsed)

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
    await ta.stop_polling()
    set_pipeline_state(running=False, phase="shutdown")
    await bus.shutdown(timeout=5.0)
    await ctx.aclose()
    await _close_http_clients()
    await graph.close()
    await store.close()


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
