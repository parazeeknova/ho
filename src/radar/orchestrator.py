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
    detect_ats_for_company,
    discover_from_searxng,
    discover_from_yc,
)
from src.radar.extractors import extract_github_index_markdown
from src.radar.gates import run_gates
from src.radar.models import (
    JobCandidate,
    JobObservation,
)
from src.radar.outreach import generate_outreach_card
from src.radar.queue import enqueue_candidate, get_queue_status, process_queue
from src.radar.scoring import compute_underdog_score
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

_SEED_BOARDS = [
    ("openai:greenhouse", "https://boards.greenhouse.io/openai", "greenhouse"),
    ("anthropic:ashby", "https://jobs.ashbyhq.com/anthropic", "ashby"),
    ("stripe:greenhouse", "https://boards.greenhouse.io/stripe", "greenhouse"),
    ("airbnb:greenhouse", "https://boards.greenhouse.io/airbnb", "greenhouse"),
    ("ycombinator:jobs", "https://www.ycombinator.com/jobs", "careers_page"),
    ("wellfound:jobs", "https://wellfound.com/jobs", "careers_page"),
]


def _make_posting_id(obs: JobObservation) -> str:
    return obs.canonical_url_hash()


def _board_entry(id_: str, url: str, adapter: str) -> dict[str, str]:
    return {"id": id_, "url": url, "adapter": adapter}


async def _load_persisted_sources(store: MemoryStore) -> list[dict[str, str]]:
    """Load all persisted company boards from Postgres plus seed boards."""
    boards = [_board_entry(id_, url, adapter) for id_, url, adapter in _SEED_BOARDS]
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source_id, source_type, active FROM source_checkpoints "
                "WHERE source_type = 'ats_board' AND active = TRUE"
            )
            seen = {b["id"] for b in boards}
            for r in rows:
                sid = r["source_id"]
                if sid not in seen:
                    url = await _load_source_url(conn, sid)
                    if url:
                        seen.add(sid)
                        boards.append(_board_entry(sid, url, "ats"))
    except Exception:
        pass
    return boards


async def _load_source_url(conn, source_id: str) -> str:
    try:
        row = await conn.fetchrow(
            "SELECT url FROM job_observations WHERE source = $1 LIMIT 1",
            source_id,
        )
        if row:
            return row["url"].rsplit("/", 1)[0]
    except Exception:
        pass
    return ""


async def _persist_discovered_source(
    store: MemoryStore,
    source_id: str,
    board_url: str,
) -> None:
    try:
        register_source(source_id, "ats_board", initial_quality=0.5)
        async with store._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO source_checkpoints
                    (source_id, source_type, last_polled, last_snapshot_hash,
                     last_snapshot_count, quality_score, active)
                VALUES ($1, 'ats_board', 0, '', 0, 0.5, TRUE)
                ON CONFLICT (source_id) DO UPDATE SET active = TRUE
                """,
                source_id,
            )
    except Exception:
        pass


# ── Company discovery ────────────────────────────────────────────────


async def _discover_new_companies(
    store: MemoryStore,
) -> list[dict[str, Any]]:
    """Discover new companies and their ATS pages."""
    logger.info("Starting company discovery sweep")
    discovered: list[dict[str, Any]] = []

    yc = await discover_from_yc(limit=30)
    for c in yc:
        c["discovered_from"] = "yc"
    discovered.extend(yc)
    logger.info(f"YC discovery: {len(yc)} companies")

    searx = await discover_from_searxng("hiring")
    for c in searx:
        c["discovered_from"] = "searxng"
    discovered.extend(searx)
    logger.info(f"SearXNG discovery: {len(searx)} companies")

    new_sources = 0
    for c in discovered:
        website = c.get("website", "")
        if not website or not website.startswith("http"):
            continue
        name = c.get("name", "")
        if not name:
            continue
        source_id = f"discovered:{name.lower().replace(' ', '-')}"

        ats_url = await detect_ats_for_company(website)
        if ats_url:
            await _persist_discovered_source(store, source_id, ats_url)
            new_sources += 1
            c["ats_url"] = ats_url
            logger.debug("New source registered", source=source_id, ats_url=ats_url)

    logger.info(f"Discovery: {len(discovered)} companies, {new_sources} new ATS sources")
    return discovered


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
    except Exception:
        record_failure(source_id)
        return []

    if not direct_urls:
        record_success(source_id, 0, 0)
        return []

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
                extra={"is_snapshot_delta": True},
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
                    if obs.extra.get("is_snapshot_delta"):
                        candidate.extra["is_snapshot_delta"] = True
                    passed.append(candidate)
                else:
                    rejected_count += 1
                    for _g, reason, _desc in rejections:
                        gate_stats[reason.value] = gate_stats.get(reason.value, 0) + 1
                    await _persist_rejected(store, obs, posting_id, rejections)
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
                    first_seen, last_seen, freshness_lane, direct_posting_verified)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (url_hash) DO UPDATE SET last_seen = EXCLUDED.last_seen
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
            )
    except Exception:
        pass


async def _persist_rejected(
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
                    normalized_company, eligibility, rejection_reason, rejection_detail,
                    freshness_lane, first_seen, last_seen, role_family)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9,'unknown')
                ON CONFLICT (canonical_id) DO NOTHING
                """,
                f"rejected:{posting_id}",
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


# ── Queue ranking + eligibility scoring ─────────────────────────────


def _rank_for_queue(candidates: list[JobCandidate]) -> list[JobCandidate]:
    urgent_high: list[JobCandidate] = []
    urgent: list[JobCandidate] = []
    sponsor: list[JobCandidate] = []
    rest: list[JobCandidate] = []

    for c in candidates:
        c.underdog_score = compute_underdog_score(c)
        c.extra["group_key"] = _group_key(c)
        if c.salary:
            c.salary_annual_usd = _compute_annual_usd(c.salary)

        # Detect sponsor/relocation/remote evidence from scraped markdown
        md = c.extra.get("raw_markdown", "").lower()
        if any(
            kw in md
            for kw in (
                "sponsor",
                "relocation",
                "e-verify",
                "global remote",
                "work from anywhere",
                "visa transfer",
            )
        ):
            c.sponsors_visa = True

        if c.is_urgent and c.salary_annual_usd and c.salary_annual_usd >= 60000:
            urgent_high.append(c)
        elif c.is_urgent:
            urgent.append(c)
        elif c.sponsors_visa:
            sponsor.append(c)
        else:
            rest.append(c)

    urgent_high.sort(key=_sort_key, reverse=True)
    urgent.sort(key=_sort_key, reverse=True)
    sponsor.sort(key=_sort_key, reverse=True)
    rest.sort(key=_sort_key, reverse=True)
    return urgent_high + urgent + sponsor + rest


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


def _compute_annual_usd(salary) -> float | None:
    try:
        rate = {"hour": 2000, "month": 12, "year": 1}.get(salary.period)
        if rate is None:
            return None
        fx = {"USD": 1.0, "INR": 1.0 / 86, "EUR": 1.1, "GBP": 1.25}.get(salary.currency, 1.0)
        return salary.amount * rate * fx
    except Exception:
        return None


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
            c.underdog_score = compute_underdog_score(c)
            await _persist_full(store, c)
        except Exception:
            pass


async def _persist_full(store: MemoryStore, candidate: JobCandidate) -> None:
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
            "rejection_reason": candidate.rejection_reason.value
            if candidate.rejection_reason
            else "",
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

    def _pid(c: JobCandidate) -> str:
        return c.extra.get("posting_id", c.canonical_id)

    urgent_high = [
        c
        for c in matched
        if c.is_urgent
        and c.is_accepted
        and c.salary_annual_usd
        and c.salary_annual_usd >= 60000
        and _pid(c) not in notified_keys
    ]
    underdog = [
        c
        for c in matched
        if c.is_accepted
        and c.underdog_score >= 0.6
        and _pid(c) not in notified_keys
        and c not in urgent_high
    ]
    sponsor = [
        c
        for c in matched
        if c.is_accepted
        and c.sponsors_visa
        and _pid(c) not in notified_keys
        and c not in urgent_high
        and c not in underdog
    ]
    startup_sig = [
        c
        for c in matched
        if c.is_accepted
        and c.funding_stage
        and _pid(c) not in notified_keys
        and c not in urgent_high
        and c not in underdog
    ]

    async def _notify(category: str, candidates: list[JobCandidate]) -> None:
        for c in candidates:
            key = _pid(c)
            try:
                ok = await telegram_agent.send_categorized_alert(
                    category,
                    _candidate_to_job_card(c),
                    dedup_key=key,
                )
                if ok:
                    notified_keys.add(key)
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

    await _notify("urgent", urgent_high)
    await _notify("outreach", underdog[:15])
    await _notify("eligible", sponsor[:10])
    await _notify("startup_signal", startup_sig[:10])


def _candidate_to_job_card(candidate: JobCandidate) -> dict[str, Any]:
    return {
        "role": candidate.normalized_role,
        "company": candidate.normalized_company,
        "match_percent": candidate.match_percent,
        "shortlist_probability": candidate.shortlist_probability,
        "salary": candidate.salary.raw if candidate.salary else None,
        "salary_annual_usd": candidate.salary_annual_usd,
        "location": candidate.normalized_location,
        "apply_link": candidate.direct_apply_url,
        "jd_summary": candidate.jd_summary,
        "company_description": candidate.company_description,
        "founders": candidate.founders,
        "funding_stage": candidate.funding_stage,
        "funding_info": candidate.funding_info,
        "osint_signals": candidate.osint_signals,
        "sponsors_visa": candidate.sponsors_visa,
        "underdog_score": round(candidate.underdog_score, 2),
    }


# ── Company events + outreach ────────────────────────────────────────


async def _dispatch_company_events(
    candidates: list[JobCandidate],
    graph: GraphStore,
    bus: EventBus,
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
        extra={"verified_posts": verified_posts, "hiring_signals": verified_posts},
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


# ── Job processor: registered handler for ats_crawler output ─────────


async def _job_processor(entry: FrontierEntry) -> list[FrontierEntry]:
    """Registered handler: process ats_crawler FrontierEntry → posting ingestion."""
    import httpx

    url = entry.payload.get("observation_url", "")
    source = entry.payload.get("source", "unknown")
    company = entry.payload.get("company", "")
    if not url or not url.startswith("http"):
        logger.warning("job_processor: invalid URL dropped", source=source, entry_id=entry.id)
        return []

    cfg = get_config().firecrawl
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{cfg.url}/v1/scrape",
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            if resp.status_code != 200:
                return []
            md = (resp.json().get("data") or {}).get("markdown", "") or ""
            if not md or len(md) < 100:
                return []

        obs = JobObservation(
            url=url,
            source=source,
            raw_markdown=md,
            title=company or "",
            snippet=f"{company} — {url[:80]}",
        )
        posting_id = _make_posting_id(obs)

        store = await MemoryStore.create()
        try:
            await _persist_observation(store, obs, posting_id)

            import time as _time

            now_ts = _time.time()
            obs.observed_at = now_ts

            known: set[str] = {posting_id}
            last_seen: dict[str, float] = {posting_id: now_ts}
            candidate, rejections = await run_gates(obs, known, last_seen)

            if candidate is not None:
                candidate.extra["raw_markdown"] = md
                candidate.extra["version"] = 1
                candidate.extra["posting_id"] = posting_id
                candidate.canonical_id = posting_id
                await enqueue_candidate(candidate, priority=40)
            else:
                await _persist_rejected(store, obs, posting_id, rejections)
        finally:
            await store.close()
    except Exception as e:
        logger.warning("job_processor failed", url=url, exception=str(e))
    return []


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
    await store.purge_fake_job_keys(["techco:backendengineer"])

    graph = await GraphStore.create()
    bus = EventBus()

    engine_cfg = get_config().scheduler
    frontier = CrawlFrontier(max_size=engine_cfg.max_queue_size)
    engine = WorkScheduler(frontier, worker_count=3)
    bus.set_enqueue_callback(engine.enqueue_many)

    startup_agent = StartupAgent(ctx)
    await load_checkpoints(store)

    # Register sources
    for id_, _url, _adapter in _SEED_BOARDS:
        register_source(id_, "ats_board", initial_quality=0.6)
    for idx_url in GITHUB_INDEXES:
        register_source(f"github:{idx_url.rsplit('/', 1)[-1]}", "github_index", initial_quality=0.3)

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

    # Register ALL agent handlers including job_processor
    engine.register_agent(
        "founder_miner",
        lambda e: _founder_miner(e, graph, bus, startup_agent),
    )
    engine.register_agent("career_site_detector", career_site_detector)
    engine.register_agent("founder_social_osint", founder_social_agent)
    engine.register_agent("employee_discovery", employee_discovery_agent)
    engine.register_agent("ats_crawler", ats_crawler)
    engine.register_agent("job_processor", _job_processor)
    engine.register_agent("outreach_generator", _outreach_handler)

    engine.start(worker_count=3)
    logger.info("Radar graph expansion engine started (7 agents registered)")

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
    last_discovery = 0.0
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

            # Company discovery (every 5 sweeps)
            if sweep == 1 or time.monotonic() - last_discovery > cfg.radar.poll_low_freq_seconds:
                last_discovery = time.monotonic()
                await _discover_new_companies(store)

            all_observations: list[JobObservation] = []

            idx_obs = await _scrape_indexes()
            all_observations.extend(idx_obs)

            boards = await _load_persisted_sources(store)
            for board in boards:
                if not should_poll(board["id"]):
                    continue
                board_obs = await _poll_company_board(board, app)
                all_observations.extend(board_obs)

            logger.info(
                f"Sources: {len(idx_obs)} from indexes, total {len(all_observations)} observations"
            )

            candidates, gate_stats = await _fetch_postings_and_gate(all_observations, store)
            set_pipeline_state(scraped=len(all_observations), gated=len(candidates))
            logger.info(f"Gating: {len(candidates)} passed, {gate_stats['rejected']} rejected")

            console.rule(
                f"[bold cyan]RADAR PHASE 3 (sweep {sweep}): LLM Matching[/bold cyan]",
            )
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

            await _enrich_high_fit(matched, startup_agent, store)
            await _dispatch_company_events(matched, graph, bus)
            await _notify_telegram(telegram_agent, matched, store)

            accepted = len([c for c in matched if c.is_accepted])
            near_miss = len([c for c in matched if c.is_near_miss])
            rejected_llm = len([c for c in matched if c.is_rejected])

            set_pipeline_state(
                matched_total=accepted,
                rejected_total=rejected_llm + near_miss + gate_stats["rejected"],
                phase="idle",
                llm_queue=get_queue_status(),
                source_health=get_source_health(),
                sweep_interval=cfg.pipeline.sweep_interval,
            )

            await persist_checkpoints(store)

            elapsed = time.monotonic() - sweep_start
            if telegram_agent.is_configured:
                await telegram_agent.send_sweep_summary(
                    sweep, accepted, len(all_observations), elapsed
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
