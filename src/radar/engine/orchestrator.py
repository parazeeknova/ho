"""RadarOrchestrator: source-first, globally-scoped, high-pay underdog job radar."""

from __future__ import annotations

import asyncio
import gc
import json
import os
import signal
import time
from datetime import date, datetime
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
    make_company_id,
    make_founder_id,
    make_work_id,
)
from src.graph.event_bus import EventBus
from src.graph.frontier import CrawlFrontier
from src.graph.graph_store import GraphStore
from src.http_client import close_all as _close_http_clients
from src.http_client import get_client
from src.llm.context import ContextManager
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore
from src.radar.core.extractors import GITHUB_INDEXES
from src.radar.core.gates import run_gates
from src.radar.core.models import EligibilityState, JobCandidate, JobObservation
from src.radar.core.queue import enqueue_candidate, get_queue_status, process_queue
from src.radar.core.scoring import compute_underdog_score, rank_score
from src.radar.core.signals import extract_signals
from src.radar.engine.outreach import generate_outreach_card
from src.radar.sources.agents import (
    ats_crawler,
    career_site_detector,
    employee_discovery_agent,
    founder_social_agent,
)
from src.radar.sources.board_registry import REGISTERED_BOARDS, get_discovery_index_sources
from src.radar.sources.discovery import (
    _extract_domain as _discovery_domain,
)
from src.radar.sources.discovery import (
    _resolve_company_domain,
    detect_ats_for_company,
    discover_from_azure,
    discover_from_betalist,
    discover_from_dealroom,
    discover_from_hackernews,
    discover_from_remoteok,
    discover_from_vc_portfolios,
    discover_from_weworkremotely,
    discover_from_yc,
    is_aggregator_domain,
)
from src.radar.sources.sources import (
    diff_snapshots,
    get_all_checkpoints,
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
from src.rag.loader import index_resume_in_pgvector, load_resume

console = Console()
logger = get_logger("radar_orchestrator")

_SEED_BOARDS = REGISTERED_BOARDS
_DISCOVERY_INDEX_SOURCES = get_discovery_index_sources()

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


def _hash_board_url(board_url: str) -> str:
    import hashlib

    return hashlib.sha256(board_url.encode()).hexdigest()[:8]


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
        board_url = c.get("ats_url", website) or website
        name_slug = c.get("name", "").lower().replace(" ", "-")[:40]
        board_hash = _hash_board_url(board_url)
        source_id = f"discovered:{name_slug}:{board_hash}"
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

    cfg = get_config()
    discovery_source = cfg.radar.discovery_source if cfg else "all"

    # Discovery disabled entirely (relic retired, no local adapters wanted).
    if discovery_source == "none":
        logger.info("Discovery disabled (DISCOVERY_SOURCE=none)")
        return results

    # Azure-relic-only discovery: the relic's company index is the single
    # source of truth for new companies. No local adapters, no search crawler.
    if discovery_source == "azure":
        try:
            azure_companies = await discover_from_azure(limit=6000)
            for c in azure_companies:
                c.setdefault("discovered_from", "azure")
            results.extend(azure_companies)
            logger.info(f"Azure discovery: {len(azure_companies)} companies")
        except Exception as e:
            logger.warning("Azure discovery failed", exception=str(e))
            _DISCOVERY_METRICS["failed_azure"] = _DISCOVERY_METRICS.get("failed_azure", 0) + 1
        if not results:
            logger.warning("Azure discovery returned nothing; no local adapters as fallback")
        _DISCOVERY_METRICS["companies_found"] = _DISCOVERY_METRICS.get("companies_found", 0) + len(
            results
        )
        # Skip companies already registered as sources in a prior sweep so a
        # full 6k blob doesn't re-persist known boards every discovery pass.
        known_slugs: set[str] = set()
        for sid in get_all_checkpoints():
            if sid.startswith("discovered:"):
                parts = sid.split(":", 2)
                if len(parts) >= 2:
                    known_slugs.add(parts[1])
        fresh: list[dict[str, Any]] = []
        skipped = 0
        for c in results:
            name_slug = c.get("name", "").lower().replace(" ", "-")[:40]
            if name_slug in known_slugs:
                skipped += 1
                continue
            fresh.append(c)
        # Cap new sources per discovery pass so a fresh 6k blob doesn't flood
        # the poller with thousands of boards at once. Each sweep will add up
        # to AZURE_DISCOVERY_BATCH new boards; subsequent sweeps continue.
        batch_cap = int(os.environ.get("AZURE_DISCOVERY_BATCH", "300"))
        fresh = fresh[:batch_cap]
        results = fresh
        logger.info(
            f"Azure discovery: {len(results)} new companies ({skipped} already registered)"
        )
        _DISCOVERY_METRICS["sources_added"] = _DISCOVERY_METRICS.get("sources_added", 0) + len(
            results
        )
        logger.info(f"Discovery: {len(results)} companies from Azure relic")
        return results

    adapters = [
        ("dealroom", discover_from_dealroom, 50),
        ("yc", discover_from_yc, 30),
        ("vc", discover_from_vc_portfolios, 40),
        ("hn", discover_from_hackernews, 30),
        ("remoteok", discover_from_remoteok, 30),
        ("weworkremotely", discover_from_weworkremotely, 30),
        ("betalist", discover_from_betalist, 30),
    ]

    # Search crawler for direct job + signal discovery
    from src.radar.sources.crawler import run_search_discovery

    try:
        search_results = await run_search_discovery(max_total_results=150)
        for c in search_results:
            c["discovered_from"] = c.get("source", "search")
        results.extend(search_results)
        logger.info(f"Search crawler: {len(search_results)} results")
    except Exception as e:
        logger.warning("Search crawler failed", exception=str(e))

    for adp_name, func, limit in adapters:
        try:
            t0 = time.monotonic()
            companies = await func(limit)
            elapsed = time.monotonic() - t0
            for c in companies:
                c["discovered_from"] = adp_name
            results.extend(companies)
            logger.info(f"Discovery {adp_name}: {len(companies)} companies ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"Discovery adapter {adp_name} failed", exception=str(e))
            _DISCOVERY_METRICS[f"failed_{adp_name}"] += 1

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

    # Build set of already-known company slugs from prior discovery sweeps
    known_slugs: set[str] = set()
    for sid in get_all_checkpoints():
        if sid.startswith("discovered:"):
            # source_id format: "discovered:{name_slug}:{board_hash}"
            parts = sid.split(":", 2)
            if len(parts) >= 2:
                known_slugs.add(parts[1])
    if known_slugs:
        logger.info(
            f"Discovery cache: {len(known_slugs)} companies already registered as active sources"
        )

    new_sources = 0
    skipped_known = 0
    total = len(deduped)
    logger.info(f"Resolving domains and detecting ATS for {total} companies...")

    sem = asyncio.Semaphore(10)

    async def _probe_company(c: dict[str, Any]) -> None:
        nonlocal new_sources, skipped_known
        async with sem:
            # Skip companies already persisted as sources in a prior sweep
            name_slug = c.get("name", "").lower().replace(" ", "-")[:40]
            if name_slug in known_slugs:
                skipped_known += 1
                if skipped_known <= 3:
                    logger.info(
                        f"Already registered, reusing cached source: {c.get('name', name_slug)}"
                    )
                elif skipped_known == 4:
                    logger.info("Reusing cached sources: ... (suppressing further logs)")
                return

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
                return
            # Validate: reject aggregator/news/social/VC domains
            if is_aggregator_domain(
                website.replace("https://", "").replace("http://", "").split("/")[0]
            ):
                _DISCOVERY_METRICS["aggregator_rejected"] = (
                    _DISCOVERY_METRICS.get("aggregator_rejected", 0) + 1
                )
                return
            ats_url = await detect_ats_for_company(website)
            if ats_url:
                c["ats_url"] = ats_url
                new_sources += 1
                _DISCOVERY_METRICS["ats_verified"] = _DISCOVERY_METRICS.get("ats_verified", 0) + 1
                logger.info(f"ATS detected for {c.get('name', website)}: {ats_url}")

    done = 0

    async def _tracked(c: dict[str, Any]) -> None:
        nonlocal done
        await _probe_company(c)
        done += 1
        if done % 5 == 0:
            logger.info(
                f"Domain/ATS progress: {done}/{total} "
                f"(ATS found: {new_sources}, reused cached: {skipped_known})"
            )

    await asyncio.gather(*[_tracked(c) for c in deduped])

    _DISCOVERY_METRICS["companies_found"] = _DISCOVERY_METRICS.get("companies_found", 0) + len(
        deduped
    )
    _DISCOVERY_METRICS["domain_resolved"] = _DISCOVERY_METRICS.get("domain_resolved", 0) + len(
        [c for c in deduped if c.get("website") and c["website"].startswith("http")]
    )
    _DISCOVERY_METRICS["sources_added"] = _DISCOVERY_METRICS.get("sources_added", 0) + new_sources

    logger.info(
        f"Discovery: {len(deduped)} companies, {new_sources} new ATS, "
        f"{skipped_known} already known (skipped)"
    )
    return deduped


# ── Source polling ───────────────────────────────────────────────────


async def _scrape_indexes() -> list[JobObservation]:
    try:
        from src.radar.sources.github_poller import poll_all_github_indexes_etag

        return await poll_all_github_indexes_etag()
    except Exception as e:
        logger.warning(f"GitHub ETag index poller fallback failed: {e}")
        return []


async def _poll_board(board: dict[str, str], app: FirecrawlApp) -> list[JobObservation]:
    source_id = board["id"]
    board_url = board["url"]
    source_type = board.get("source_type", "discovery_index")
    is_official = source_type == "official_ats"
    if not should_poll(source_id):
        return []
    if not board_url or not board_url.startswith("http"):
        return []

    logger.info(f"Polling board: {source_id}")
    t0 = time.monotonic()

    # Layer 1: High-Speed Direct ATS API Interceptor
    # (Greenhouse, Lever, Ashby, Workable, SmartRecruiters)
    try:
        from src.radar.sources.ats_interceptor import intercept_ats_board

        ats_obs = await intercept_ats_board(board_url, source_id)
        if ats_obs is not None:
            record_success(source_id, len(ats_obs), len(ats_obs))
            return ats_obs
    except Exception as e:
        logger.debug(f"ATS interceptor fallback for {board_url}: {e}")

    # Tier 2: expensive full-board map. Only re-map when the last snapshot is
    # stale (snapshot_ttl_hours) or no snapshot exists yet (first discovery).
    ats_cfg = get_config().ats
    cp = get_checkpoint(source_id)
    if cp.last_snapshot_hash and cp.last_polled:
        snapshot_age_hours = (time.time() - cp.last_polled) / 3600.0
        if snapshot_age_hours < ats_cfg.snapshot_ttl_hours:
            logger.debug(
                f"Board {source_id} snapshot {snapshot_age_hours:.1f}h old "
                f"(TTL {ats_cfg.snapshot_ttl_hours}h), skipping re-map"
            )
            return []

    direct_urls: list[str] = []
    try:
        resp = await asyncio.wait_for(
            asyncio.to_thread(app.map_url, board_url),
            timeout=120.0,
        )
        if isinstance(resp, list):
            for item in resp:
                url = item if isinstance(item, str) else item.get("url", "")
                if url and url.startswith("http"):
                    direct_urls.append(url)
        elif isinstance(resp, dict):
            for link in resp.get("links", []) or []:
                if isinstance(link, str) and link.startswith("http"):
                    direct_urls.append(link)
    except Exception as exc:
        logger.warning(f"Firecrawl map_url failed for {board_url}: {exc}")
        record_failure(source_id)
        return []

    map_elapsed = time.monotonic() - t0
    if map_elapsed > 5.0:
        logger.info(f"Board {source_id} map took {map_elapsed:.1f}s")

    if not direct_urls:
        record_success(source_id, 0, 0)
        return []

    had_prior = bool(get_checkpoint(source_id).last_snapshot_hash)
    state = diff_snapshots(source_id, direct_urls)
    new_urls = state.new_urls
    elapsed = time.monotonic() - t0
    logger.info(
        f"Board {source_id}: {len(direct_urls)} URLs mapped, {len(new_urls)} new ({elapsed:.1f}s)",
    )

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
                    "official_source": is_official,
                    "source_type": source_type,
                },
                source_freshness_evidence=None,
            )
        )
    record_success(source_id, len(new_urls), len(new_urls))
    return observations


# ── Posting fetch + gates ────────────────────────────────────────────


def _json_to_markdown(raw_json: Any) -> str:
    """Flatten an ATS item JSON into readable job markdown.

    The crawl worker stores the raw API item; the matcher needs a human-
    readable description. Pulls the platform-specific long-form fields
    (descriptionPlain, description, content, summary, ...) plus core
    attributes, recursing shallowly.
    """
    if not isinstance(raw_json, dict):
        return ""
    parts: list[str] = []
    for key in ("title", "name", "text"):
        val = raw_json.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
            break
    location = raw_json.get("location")
    if isinstance(location, dict) and location.get("name"):
        parts.append(f"Location: {location['name']}")
    elif isinstance(location, str) and location:
        parts.append(f"Location: {location}")
    for key in ("city", "country", "team", "department", "employmentType", "category"):
        val = raw_json.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key.title()}: {val.strip()}")
    for key in (
        "descriptionPlain",
        "description",
        "content",
        "summary",
        "overview",
        "jobAd",
        "descriptionHtml",
    ):
        val = raw_json.get(key)
        if isinstance(val, str) and len(val) > 40:
            parts.append(val.strip())
            break
    if "lists" in raw_json and isinstance(raw_json["lists"], list):
        for item in raw_json["lists"]:
            if isinstance(item, dict):
                head = item.get("text", "")
                content = item.get("content", "")
                if head:
                    parts.append(f"## {head}")
                if isinstance(content, str) and content:
                    parts.append(content)
    text = "\n\n".join(p for p in parts if p)
    if len(text) > 20:
        import html as _html

        return _html.unescape(text)
    return ""


async def _load_ungated_observations(store: MemoryStore, limit: int = 400) -> list[JobObservation]:
    """Pull never-gated observations ordered by LEARNED gate-pass probability.

    Rows in job_observations without a matching radar_candidates row were
    ingested (Azure dumps, historic corpus) but never gated. The sweep's live
    source polling only re-surfaces a few thousand postings, so this drains the
    stored corpus a batch at a time into the gate + matcher.

    Ordering is learned, not hand-maintained: for each keyword in the title we
    look up how often that keyword historically passed the gate (from
    radar_candidates eligibility), then sort by that learned pass-rate, then by
    resume affinity, then freshness. This self-adapts as the corpus and gate
    evolve, so the sweep budget never collapses after the obvious junior roles
    are drained.
    """
    try:
        scores = await store.learned_title_scores()
        score_sql = "(0.0)"
        # Only drain titles that contain at least one keyword with a known,
        # non-trivial historical pass-rate. This filters out the long tail of
        # non-technical / obscure titles that the gate would reject anyway,
        # so the LLM budget is spent on roles that actually have a chance.
        known_kw_filter = "TRUE"
        if scores:
            # Build a SQL CASE scoring a title by its known keyword pass-rates.
            # Only alphanumeric keywords are safe to embed; punctuation tokens
            # (ci/cd, ui/ux, founder's, data/robot) are skipped to avoid syntax
            # errors from escaped quotes/slashes.
            known = sorted(scores.items(), key=lambda kv: -kv[1])[:150]
            arms = " ".join(
                f"WHEN lower(o.title) LIKE '%{kw}%' THEN {sc:.3f}"
                for kw, sc in known
                if kw and kw.replace("-", "").isalnum()
            )
            if arms:
                score_sql = f"(CASE {arms} ELSE 0.05 END)"
                # Hard inclusion: only drain titles that contain a strong,
                # alphanumeric keyword with a decent historical pass-rate.
                strong = [kw for kw, sc in known if sc >= 0.5 and kw.replace("-", "").isalnum()]
                if strong:
                    # Use ILIKE OR-chain to avoid regex-escaping pitfalls.
                    conds = " OR ".join(
                        f"lower(o.title) LIKE '%{kw}%'" for kw in strong[:60]
                    )
                    known_kw_filter = f"({conds})"

        async with store._pool.acquire() as conn:
            centroid: list[float] | None = None
            try:
                import numpy as np

                vecs = await conn.fetch(
                    "SELECT embedding FROM resume_embeddings WHERE embedding IS NOT NULL"
                )
                if vecs:
                    acc = np.zeros(1024, dtype=np.float32)
                    for v in vecs:
                        acc += np.asarray(v["embedding"].to_list(), dtype=np.float32)
                    acc /= len(vecs)
                    centroid = acc.tolist()
            except Exception:
                centroid = None

            # Known-bad seniority keywords must still sink to the bottom even if
            # a learned token is present (a "Senior" role is never a junior win).
            hard_block = (
                "CASE WHEN lower(o.title) ~ "
                "'senior|staff|principal|lead|head|director|vp\\b|chief|architect' "
                "THEN 1 ELSE 0 END"
            )
            # Freshness boost: recent postings (<3 days) rank above old ones so
            # the budget isn't spent on stale listings the gate would reject as
            # source_stale.
            fresh_bonus = (
                "CASE WHEN o.last_seen > extract(epoch from now()) - 3*86400 "
                "THEN 0 ELSE 1 END"
            )

            if centroid is not None:
                import numpy as np

                q = np.asarray(centroid, dtype=np.float32)
                qn = np.linalg.norm(q)
                if qn > 0:
                    q = q / qn
                rows = await conn.fetch(
                    f"""
                    SELECT o.url, o.source, o.title, o.snippet, o.last_seen, o.raw_json,
                           (CASE WHEN e.embedding IS NULL THEN NULL
                                 ELSE 1 - (e.embedding <=> $2::vector) END) AS affinity
                    FROM job_observations o
                    LEFT JOIN radar_candidates r ON r.direct_apply_url = o.url
                    LEFT JOIN obs_embeddings e ON e.url_hash = md5(o.url)
                    WHERE r.canonical_id IS NULL
                      AND {known_kw_filter}
                    ORDER BY {hard_block} ASC,
                             {score_sql} DESC,
                             {fresh_bonus} ASC,
                             affinity DESC NULLS LAST,
                             o.last_seen DESC
                    LIMIT $1
                    """,
                    limit,
                    q.tolist(),
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT o.url, o.source, o.title, o.snippet, o.last_seen, o.raw_json,
                           NULL AS affinity
                    FROM job_observations o
                    LEFT JOIN radar_candidates r ON r.direct_apply_url = o.url
                    WHERE r.canonical_id IS NULL
                      AND {known_kw_filter}
                    ORDER BY {hard_block} ASC,
                             {score_sql} DESC,
                             {fresh_bonus} ASC,
                             o.last_seen DESC
                    LIMIT $1
                    """,
                    limit,
                )
    except Exception as exc:
        logger.warning(f"Learned drain failed ({exc}); falling back to freshness")
        try:
            async with store._pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT o.url, o.source, o.title, o.snippet, o.last_seen, o.raw_json
                    FROM job_observations o
                    LEFT JOIN radar_candidates r ON r.direct_apply_url = o.url
                    WHERE r.canonical_id IS NULL
                    ORDER BY o.last_seen DESC
                    LIMIT $1
                    """,
                    limit,
                )
        except Exception:
            return []
    from src.radar.core.gates import prefilter_observation

    out: list[JobObservation] = []
    for r in rows:
        if not r["url"] or not r["url"].startswith("http"):
            continue
        raw_json = r["raw_json"]
        if isinstance(raw_json, str) and raw_json:
            try:
                raw_json = json.loads(raw_json)
            except Exception:
                raw_json = None
        obs = JobObservation(
            url=r["url"],
            source=r["source"] or "corpus",
            title=r["title"] or "",
            snippet=r["snippet"] or "",
            raw_markdown=_json_to_markdown(raw_json),
            observed_at=float(r["last_seen"] or 0),
        )
        # Cheap gate pre-check: skip anything the title/url gates would reject,
        # so the sweep budget never lands on roles doomed to rejection.
        if not prefilter_observation(obs, set(), {}):
            continue
        out.append(obs)
        if len(out) >= limit:
            break
    return out


async def _fetch_postings_and_gate(
    observations: list[JobObservation],
    store: MemoryStore,
) -> tuple[list[JobCandidate], dict[str, Any]]:
    import time as _time

    cfg = get_config().firecrawl
    passed: list[JobCandidate] = []
    rejected_count = 0
    gate_stats: dict[str, int] = {}
    gate_start = _time.monotonic()

    known_hashes: set[str] = set()
    last_seen: dict[str, float] = {}
    try:
        async with store._pool.acquire() as conn:
            # Already-gated postings live in radar_candidates (canonical_id
            # for passed, "rejected:<pid>" for rejected). Observations that
            # exist but were NEVER gated must still go through the gates -
            # loading known_hashes from job_observations (the whole corpus)
            # would skip everything and starve the matcher.
            rows = await conn.fetch("SELECT canonical_id FROM radar_candidates")
            for r in rows:
                cid = r["canonical_id"] or ""
                known_hashes.add(cid.removeprefix("rejected:"))
            rows = await conn.fetch("SELECT url_hash, last_seen FROM job_observations")
            for r in rows:
                if r["last_seen"]:
                    last_seen[r["url_hash"]] = float(r["last_seen"])
    except Exception:
        pass

    sem = asyncio.Semaphore(8)
    processed_count = 0
    total = len(observations)
    if total > 0:
        logger.info(f"Fetching and gating {total} postings...")

    async def _process_one(obs: JobObservation) -> None:
        nonlocal rejected_count, processed_count
        async with sem:
            try:
                pid = _posting_id(obs)
                if pid in known_hashes:
                    # Already indexed before: URL_DUPLICATE gate would reject this
                    # regardless, so skip the expensive scrape entirely and just
                    # refresh last_seen.
                    obs.observed_at = _time.time()
                    await _persist_observation(store, obs, pid)
                    processed_count += 1
                    return

                if not obs.raw_markdown or len(obs.raw_markdown) < 100:
                    client = await get_client("orchestrator", timeout=15.0)
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

                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(
                        f"Posting fetch/gate: {processed_count}/{total} (passed: {len(passed)})",
                    )

            except Exception:
                _PIPELINE_METRICS["failed_fetches"] += 1

    tasks = [asyncio.create_task(_process_one(o)) for o in observations]
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = _time.monotonic() - gate_start
    logger.info(
        f"Gating complete: {len(passed)} passed, {rejected_count} rejected in {elapsed:.1f}s"
    )
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
        return rank_score(x)

    return (
        sorted(urgent_high, key=_sort, reverse=True)
        + sorted(urgent, key=_sort, reverse=True)
        + sorted(sponsor, key=_sort, reverse=True)
        + sorted(rest, key=_sort, reverse=True)
    )


def _group_key(c: JobCandidate) -> str:
    from src.radar.core.models import make_canonical_id

    return make_canonical_id(c.normalized_company, c.normalized_role, c.normalized_location)


# ── Post-LLM enrichment ──────────────────────────────────────────────


def _as_list(val: Any) -> list[Any]:
    """Coerce JSON-string-typed enrichment values back to real lists."""
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return []
    return val if isinstance(val, list) else []


async def _enrich_high_fit(
    candidates: list[JobCandidate], sa: StartupAgent, store: MemoryStore, graph: GraphStore
) -> None:
    accepted = [c for c in candidates if c.is_accepted]
    if not accepted:
        return
    logger.info(f"Enriching {len(accepted)} accepted candidates with OSINT & founder details...")

    jobs: list[dict[str, Any]] = []
    by_candidate: dict[int, JobCandidate] = {}
    for c in accepted:
        by_candidate[len(jobs)] = c
        jobs.append(
            {
                "role": c.normalized_role,
                "company": c.normalized_company,
                "match_percent": c.match_percent,
                "verdict": c.verdict,
                "source": c.source,
                "apply_link": c.direct_apply_url,
                "jd_summary": c.jd_summary,
                "company_description": c.company_description,
                "role_summary": c.role_summary,
            }
        )

    enriched_all = await sa.batch_analyze_startups(jobs, concurrency=8)

    for idx, c in by_candidate.items():
        enriched = enriched_all[idx] if idx < len(enriched_all) else {}
        c.founders = _as_list(enriched.get("founders"))
        c.funding_stage = enriched.get("funding_stage", "")
        c.funding_info = enriched.get("funding_info", {}) or {}
        c.founder_socials = _as_list(enriched.get("founder_socials"))
        c.company_news = enriched.get("company_news", "")
        c.osint_signals = _as_list(enriched.get("osint_signals"))
        c.underdog_score = compute_underdog_score(c)

        # Inject graph structural insights (predictive link prediction)
        if c.funding_stage or enriched.get("founders"):
            try:
                graph_insights = await graph.generate_graph_insights_for_llm(
                    make_company_id(c.normalized_company)
                )
                if graph_insights:
                    existing = list(c.osint_signals or [])
                    existing.append(graph_insights)
                    c.osint_signals = existing
            except Exception:
                pass

        await _persist_full(store, c)

    for idx, c in by_candidate.items():
        logger.info(f"Enriching {c.normalized_company}: {idx + 1}/{len(accepted)}")


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
) -> int:
    """Send Telegram alerts for accepted candidates. Returns actual send count."""
    if not ta.is_configured:
        return 0
    notified: set[str] = set()
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch("SELECT dedup_key FROM telegram_notified_jobs")
            notified.update(r["dedup_key"] for r in rows)
    except Exception:
        pass

    def _pid(c: JobCandidate) -> str:
        return c.extra.get("posting_id", c.canonical_id)

    sent_count = 0

    urgent = [
        c
        for c in matched
        if c.is_urgent
        and c.is_accepted
        and c.salary_annual_usd
        and c.salary_annual_usd >= 60000
        and _pid(c) not in notified
    ]
    urgent_pids = {_pid(c) for c in urgent}

    underdog = [
        c
        for c in matched
        if c.is_accepted
        and c.underdog_score >= 0.6
        and _pid(c) not in notified
        and _pid(c) not in urgent_pids
    ]
    underdog_pids = {_pid(c) for c in underdog}

    sponsor = [
        c
        for c in matched
        if c.is_accepted
        and c.sponsors_visa
        and _pid(c) not in notified
        and _pid(c) not in urgent_pids
        and _pid(c) not in underdog_pids
    ]
    sponsor_pids = {_pid(c) for c in sponsor}

    startup = [
        c
        for c in matched
        if c.is_accepted
        and c.funding_stage
        and _pid(c) not in notified
        and _pid(c) not in urgent_pids
        and _pid(c) not in underdog_pids
        and _pid(c) not in sponsor_pids
    ]
    startup_pids = {_pid(c) for c in startup}

    # Catch-all: any accepted candidate not in specialized buckets
    categorized_pids = urgent_pids | underdog_pids | sponsor_pids | startup_pids
    general = [
        c
        for c in matched
        if c.is_accepted and _pid(c) not in notified and _pid(c) not in categorized_pids
    ]

    async def _notify(cat: str, candidates: list[JobCandidate]) -> int:
        count = 0
        for c in candidates:
            link = c.direct_apply_url or c.extra.get("source_url") or c.extra.get("ats_url") or ""
            if not link or not str(link).startswith("http"):
                logger.warning(
                    f"Telegram skip (no valid link): {c.normalized_company} / {c.normalized_role}"
                )
                continue
            key = _pid(c)
            try:
                if c.salary is None and not c.salary_annual_usd:
                    from src.radar.sources.salary_lookup import estimate_salary

                    est, est_source = await estimate_salary(
                        c.normalized_company, c.normalized_role, store
                    )
                    if est is not None:
                        c.salary = est
                        c.salary_annual_usd = est.annual_usd_equivalent
                        c.extra["salary_estimated"] = True
                        if est_source:
                            c.extra["salary_source"] = est_source
                ok = await ta.send_categorized_alert(cat, _card(c), dedup_key=key)
                if ok:
                    count += 1
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
        return count

    sent_count += await _notify("urgent", urgent)
    sent_count += await _notify("outreach", underdog[:15])
    sent_count += await _notify("eligible", sponsor[:10])
    sent_count += await _notify("startup_signal", startup[:10])
    sent_count += await _notify("general_accepted", general[:20])
    logger.info(
        f"Telegram delivery: {sent_count} sent "
        f"(urgent={len(urgent)}, underdog={len(underdog)}, "
        f"sponsor={len(sponsor)}, startup={len(startup)}, general={len(general)})"
    )
    return sent_count


async def _check_and_notify_stealth_signals(
    ta: TelegramAgent, graph: GraphStore, store: MemoryStore
) -> int:
    """Detect and notify high-centrality stealth startups before public job postings exist."""
    if not ta.is_configured:
        return 0
    signals = await graph.detect_stealth_hiring_signals(limit=5)
    if not signals:
        return 0
    sent = 0
    for s in signals:
        cname = s.get("company_name", "Unknown")
        if not cname or cname in ("Unknown", "N/A", ""):
            continue
        key = f"stealth:{cname.lower().strip()}"
        job_card = {
            "role": "Founding / Early Engineer (Stealth Signal)",
            "company": cname,
            "match_percent": 85,
            "shortlist_probability": 80,
            "location": "Remote / Onsite",
            "apply_link": s.get("url") or f"https://www.google.com/search?q={cname}+startup",
            "funding_stage": s.get("funding_stage") or "Seed / Venture Backed",
            "company_description": (
                f"High graph-centrality stealth company (PageRank: {s.get('pagerank', 0.0)}). "
                "No public postings on ATS boards yet — prime target for cold outreach!"
            ),
            "osint_signals": [
                f"PageRank Score: {s.get('pagerank', 0.0)}",
                "Pre-posting stealth hiring signal detected",
            ],
        }
        try:
            ok = await ta.send_categorized_alert("startup_signal", job_card, dedup_key=key)
            if ok:
                sent += 1
        except Exception:
            pass
    if sent > 0:
        logger.info(f"Dispatched {sent} stealth startup hiring signals to Telegram")
    return sent


def _card(c: JobCandidate) -> dict[str, Any]:
    link = c.direct_apply_url or c.extra.get("source_url") or c.extra.get("ats_url") or ""
    return {
        "role": c.normalized_role,
        "company": c.normalized_company,
        "match_percent": c.match_percent,
        "shortlist_probability": c.shortlist_probability,
        "salary": c.salary.raw if c.salary else None,
        "salary_annual_usd": c.salary_annual_usd,
        "salary_estimated": bool(c.extra.get("salary_estimated")),
        "salary_source": c.extra.get("salary_source", ""),
        "location": c.normalized_location,
        "apply_link": link,
        "jd_summary": c.jd_summary,
        "company_description": c.company_description,
        "role_summary": c.role_summary,
        "founders": c.founders,
        "funding_stage": c.funding_stage,
        "funding_info": c.funding_info,
        "osint_signals": c.osint_signals,
        "sponsors_visa": c.sponsors_visa,
        "is_remote": c.is_remote,
        "underdog_score": round(c.underdog_score, 2),
        "matching_skills": c.matching_skills,
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
            logger.info(f"Dispatched company '{c.normalized_company}' to graph event bus")
            await bus.fire(
                bus.new_event(
                    "company_discovered",
                    node.id,
                    NodeType.COMPANY,
                    {"name": c.normalized_company, "url": c.direct_apply_url},
                )
            )
        except Exception as e:
            logger.debug(f"Failed dispatching company event for {company}: {e}")


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
        node.data["founders"] = _json_safe(founders)
        node.data["funding_stage"] = enriched.get("funding_stage", "")
        node.data = _json_safe(node.data)
        node, _ = await graph.upsert_node(node)
    logger.info(f"Founder miner: resolved {len(founders)} founders for '{cn}'")
    results: list[FrontierEntry] = []
    for f in founders[:3]:
        if isinstance(f, dict) and f.get("name"):
            fn = GraphNode(
                id=make_founder_id(f["name"], cn),
                node_type=NodeType.FOUNDER,
                data=_json_safe({**f, "company": cn}),
            )
            fn, _ = await graph.upsert_node(fn)
            _, _ = await graph.upsert_edge(edge(entry.node_id, EdgeType.FOUNDED_BY, fn.id))
            logger.info(f"Graph edge: founder '{f['name']}' -FOUNDED_BY-> company '{cn}'")
            await bus.fire(
                bus.new_event(
                    "founder_discovered",
                    fn.id,
                    NodeType.FOUNDER,
                    _json_safe(
                        {
                            "name": f["name"],
                            "company": cn,
                            "title": f.get("title", ""),
                            "email": f.get("email"),
                            "linkedin_url": f.get("linkedin_url"),
                            "github_url": f.get("github_url"),
                            "funding_stage": enriched.get("funding_stage", ""),
                            "funding_info": enriched.get("funding_info"),
                            "osint_signals": enriched.get("osint_signals", []),
                            "company_news": enriched.get("company_news", ""),
                            "match_percent": enriched.get("match_percent", 0),
                            "shortlist_probability": enriched.get("shortlist_probability", 0),
                            "careers_url": (node.data.get("careers_url", "") if node else ""),
                            "website": (node.data.get("website", "") if node else ""),
                        }
                    ),
                )
            )
    return results


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON values (datetime, date, sets) to JSON-safe forms."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    return obj


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
            link = linkedin_url or (c.founders[0].get("email") if c.founders else "")
            await ta.send_categorized_alert(
                "outreach",
                {
                    "role": f"Outreach to {founder_name}",
                    "company": company,
                    "apply_link": link,
                    "source_url": (
                        entry.payload.get("website") or entry.payload.get("careers_url") or link
                    ),
                    "company_description": entry.payload.get("company_news", ""),
                    "match_percent": entry.payload.get("match_percent", 0),
                    "shortlist_probability": entry.payload.get("shortlist_probability", 0),
                    "funding_stage": c.funding_stage,
                    "osint_signals": entry.payload.get("osint_signals", []),
                    "founders": c.founders,
                },
                dedup_key=f"outreach:{company}:{founder_name}",
            )
    return []


# ── job_processor ────────────────────────────────────────────────────


async def _job_processor(entry: FrontierEntry) -> list[FrontierEntry]:
    """Process ats_crawler output: fetch posting, load DB state, gate, enqueue."""

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
        client = await get_client("orchestrator", timeout=20.0)
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
        # Already-gated postings live in radar_candidates; the observation
        # corpus itself must not be treated as "already seen" or nothing
        # from the Azure dump would ever be gated.
        known_hashes: set[str] = set()
        last_seen: dict[str, float] = {}
        try:
            async with store._pool.acquire() as conn:
                rows = await conn.fetch("SELECT canonical_id FROM radar_candidates")
                for r in rows:
                    cid = r["canonical_id"] or ""
                    known_hashes.add(cid.removeprefix("rejected:"))
                rows = await conn.fetch("SELECT url_hash, last_seen FROM job_observations")
                for r in rows:
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
            await enqueue_candidate(candidate, priority=40, store=store)
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
    if not os.getenv("HO_WORKER_ONLY"):
        await ta.start_polling()

    store = await MemoryStore.create()
    await store.purge_fake_job_keys(["techco:backendengineer"])
    graph = await GraphStore.create()
    bus = EventBus()

    ecfg = get_config().scheduler
    frontier = CrawlFrontier(max_size=ecfg.max_queue_size)
    engine = WorkScheduler(frontier, worker_count=8)
    bus.set_enqueue_callback(engine.enqueue_many)
    sa = StartupAgent(ctx, store=store)
    await load_checkpoints(store)

    for id_, _url, _source_type in _SEED_BOARDS:
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
        entries = [
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
        # Cold-DM card: the founder is a real outreach target once we have
        # an email or profile URL (triangulated emails flow in from the
        # startup OSINT cache; LinkedIn only when search finds a real page).
        email = d.get("email")
        linkedin = d.get("linkedin_url")
        if email or linkedin:
            try:
                ta = TelegramAgent(ctx=ctx)
                if ta.is_configured:
                    link = linkedin or (f"mailto:{email}" if email else "")
                    await ta.send_categorized_alert(
                        "outreach",
                        {
                            "role": f"Founder: {d.get('name', '?')}",
                            "company": d.get("company", ""),
                            "apply_link": link,
                            "source_url": d.get("website") or d.get("careers_url") or link,
                            "company_description": d.get("company_news", ""),
                            "match_percent": d.get("match_percent", 0),
                            "shortlist_probability": d.get("shortlist_probability", 0),
                            "funding_stage": d.get("funding_stage", ""),
                            "funding_info": d.get("funding_info"),
                            "osint_signals": d.get("osint_signals", []),
                            "founders": [
                                {
                                    "name": d.get("name", "?"),
                                    "title": d.get("title", ""),
                                    "email": email,
                                    "linkedin_url": linkedin,
                                    "github_url": d.get("github_url"),
                                }
                            ],
                        },
                        dedup_key=f"founder:{event.node_id}",
                    )
            except Exception as exc:
                logger.warning(f"Founder alert failed: {exc}")
        return entries

    async def _sub_career(event):
        url = event.payload.get("url", "")
        if any(
            a in url.lower()
            for a in (
                "greenhouse",
                "lever.co",
                "ashbyhq",
                "workable",
                "myworkdayjobs",
                "smartrecruiters",
                "rippling",
                "teamtailor",
                "recruitee",
                "comeet",
                "jobscore",
                "jazzhr",
            )
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

    is_worker = bool(os.getenv("HO_WORKER_ONLY"))

    if not is_worker and ta.is_configured:
        asyncio.create_task(ta.start_polling())

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
    engine.start(worker_count=8)

    loop = asyncio.get_running_loop()
    existing_count = await store.chunk_count()
    full_text = ""
    if existing_count > 0:
        if not is_worker:
            logger.info(f"Reusing {existing_count} existing resume chunks")
    else:
        full_text, chunks = await loop.run_in_executor(None, load_resume)
        await index_resume_in_pgvector(chunks, store)

    candidate_persona = cfg.candidate.persona

    if not is_worker:
        console.rule("[bold cyan]RADAR PHASE 1: Load Resume[/bold cyan]")
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

        # Re-deliver any previously accepted candidates that were never
        # notified to Telegram (e.g., if DNS was down during the first sweep)
        try:
            async with store._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT canonical_id, normalized_role, normalized_company, "
                    "direct_apply_url, normalized_location, match_percent, shortlist_probability, "
                    "verdict, funding_stage, salary_amount, salary_currency, salary_period, "
                    "salary_raw, jd_summary, company_description, role_summary, "
                    "is_remote, founders, funding_info, "
                    "founder_socials, company_news, osint_signals, extra "
                    "FROM radar_candidates "
                    "WHERE eligibility = 'accepted' "
                    "AND canonical_id NOT IN (SELECT dedup_key FROM telegram_notified_jobs) "
                    "LIMIT 50",
                )
            if rows:
                from src.radar.core.models import JobCandidate, NormalizedSalary

                pending: list[JobCandidate] = []
                for r in rows:
                    c = JobCandidate(
                        canonical_id=r["canonical_id"],
                        normalized_role=r["normalized_role"],
                        normalized_company=r["normalized_company"],
                        normalized_location=r.get("normalized_location") or "Remote",
                        source="radar",
                        direct_apply_url=r.get("direct_apply_url", "") or "",
                        match_percent=r["match_percent"],
                        shortlist_probability=r.get("shortlist_probability") or 0,
                        verdict=r["verdict"],
                        funding_stage=r.get("funding_stage", ""),
                    )
                    salary_amount = r.get("salary_amount")
                    salary_raw = r.get("salary_raw") or ""
                    if salary_amount or salary_raw:
                        c.salary = NormalizedSalary(
                            amount=salary_amount or 0,
                            currency=r.get("salary_currency") or "USD",
                            period=r.get("salary_period") or "year",
                            raw=salary_raw,
                        )
                    if c.salary and c.salary_annual_usd is None:
                        annual = c.salary.annual_usd_equivalent
                        if annual:
                            c.salary_annual_usd = annual
                    for field in (
                        "jd_summary",
                        "company_description",
                        "role_summary",
                        "founders",
                        "funding_info",
                        "founder_socials",
                        "company_news",
                        "osint_signals",
                    ):
                        val = r.get(field)
                        if val:
                            setattr(c, field, val)
                    c.is_remote = bool(r.get("is_remote"))
                    extra_raw = r.get("extra")
                    if isinstance(extra_raw, dict):
                        c.extra = extra_raw
                    elif isinstance(extra_raw, str) and extra_raw.strip():
                        try:
                            import json

                            parsed = json.loads(extra_raw)
                            c.extra = parsed if isinstance(parsed, dict) else {}
                        except Exception:
                            c.extra = {}
                    else:
                        c.extra = {}
                    c.sponsors_visa = bool(c.extra.get("sponsors_visa"))
                    c.underdog_score = float(c.extra.get("underdog_score") or 0)
                    c.eligibility = EligibilityState.ACCEPTED
                    pending.append(c)
                if pending:
                    logger.info(f"Re-delivering {len(pending)} unnotified candidates to Telegram")
                    await _notify_telegram(ta, pending, store)
        except Exception as e:
            logger.warning(f"Startup re-delivery failed: {e}")

    sweep = 0
    last_discovery = 0.0
    last_index_scrape = -9999.0
    last_graph_metrics = 0.0
    last_mass_poll = -9999.0
    last_dorks = -9999.0
    _index_interval = 1800.0  # 30 min between GitHub index scrapes
    _graph_metrics_interval = 7200.0  # 2 hours between graph metric recomputes
    _mass_poll_interval = 14400.0  # 4 hours between mass ATS slug polls
    _dork_interval = 1800.0  # 30 min between SearXNG dork runs

    from src.radar.sources.dorking import DorkingEngine

    dork_engine = DorkingEngine()

    if is_worker:
        logger.info("Worker process listening for queued candidate matching tasks...")
        while not shutdown_requested.is_set():
            resume_ctx = full_text[:3000] if full_text else candidate_persona
            matched = await process_queue(
                ctx,
                resume_ctx,
                candidate_persona,
                store,
                max_candidates=10,
            )
            if not matched:
                await asyncio.sleep(2.0)
                continue
            # Workers compete with the master for the shared DB queue, so
            # they must run the full post-processing (enrichment, graph
            # events, Telegram cards) on what they matched - otherwise
            # accepted candidates are persisted but never surfaced.
            try:
                await _enrich_high_fit(matched, sa, store, graph)
                await _dispatch_company_events(matched, graph, bus)
                await _notify_telegram(ta, matched, store)
            except Exception as exc:
                logger.warning(f"Worker post-processing failed: {exc}")
        return

    while True:
        if shutdown_requested.is_set():
            break
        sweep += 1
        sweep_start = time.monotonic()
        if not is_worker:
            logger.info(f"=== Sweep {sweep} starting ===")
        set_pipeline_state(
            sweep=sweep, phase=f"sweep {sweep}: scraping", sweep_started_at=time.time()
        )

        try:
            if not is_worker:
                console.rule(
                    f"[bold cyan]RADAR PHASE 2 (sweep {sweep}): Source Polling + Gating[/bold cyan]"
                )

            all_obs: list[JobObservation] = []

            # 0. Never-gated observations from the corpus (Azure dumps etc).
            #    The sweep polls live sources, which re-returns the same few
            #    thousand postings; the 200K observations sitting in
            #    job_observations were never gated. Pull the freshest batch
            #    each sweep so the whole corpus eventually flows through the
            #    gate + matcher.
            if not is_worker:
                ungated = await _load_ungated_observations(store, limit=2000)
                all_obs.extend(ungated)
                if ungated:
                    logger.info(
                        f"Sweep {sweep}: {len(ungated)} never-gated observations from corpus"
                    )
            if not is_worker and time.monotonic() - last_mass_poll > _mass_poll_interval:
                try:
                    from src.radar.sources.ats_mass_poller import poll_all_mass_slugs

                    last_mass_poll = time.monotonic()
                    mass_obs = await poll_all_mass_slugs()
                    all_obs.extend(mass_obs)
                    logger.info(f"Sweep {sweep}: {len(mass_obs)} from mass ATS poller")
                except Exception as mass_err:
                    logger.debug(f"Mass ATS poller sweep: {mass_err}")

            # 2. GitHub Indexes (400+ jobs) - High speed instant scraping
            if time.monotonic() - last_index_scrape > _index_interval:
                last_index_scrape = time.monotonic()
                index_obs = await _scrape_indexes()
                all_obs.extend(index_obs)
                logger.info(f"Sweep {sweep}: {len(index_obs)} observations from GitHub indexes")

            # 3. Pillar 2 SearXNG Dorking (Junior/New Grad/Entry Level queries)
            try:
                if time.monotonic() - last_dorks > _dork_interval:
                    last_dorks = time.monotonic()
                    dork_obs = await dork_engine.execute_dorks()
                    all_obs.extend(dork_obs)
                    logger.info(f"Sweep {sweep}: {len(dork_obs)} observations from SearXNG dorks")
            except Exception as dork_err:
                logger.warning(f"SearXNG dorking sweep warning: {dork_err}")

            # 4. Load 110+ verified seed boards + active sources
            active_sources = await load_active_sources(store)
            for id_, url, source_type in _SEED_BOARDS:
                if should_poll(id_) and not any(s["id"] == id_ for s in active_sources):
                    active_sources.append({"id": id_, "url": url, "source_type": source_type})

            logger.info(f"Sweep {sweep}: {len(active_sources)} active sources to poll")

            # Parallel source polling across all registered seed boards
            poll_sem = asyncio.Semaphore(12)
            board_results: list[list[JobObservation]] = []

            async def _poll_one(board, sem):
                async with sem:
                    return await _poll_board(board, app)

            tasks = [asyncio.create_task(_poll_one(b, poll_sem)) for b in active_sources]
            for _board_done, task in enumerate(asyncio.as_completed(tasks), start=1):
                try:
                    res = await task
                    board_results.append(res)
                except Exception as exc:
                    logger.warning(f"Board poll failed: {exc}")

            for b_obs in board_results:
                all_obs.extend(b_obs)

            logger.info(f"Sweep {sweep}: Total raw observations fetched: {len(all_obs)}")

            # 4. Dynamic discovery + ATS detection (Master process background task)
            if not is_worker and (
                sweep == 1 or time.monotonic() - last_discovery > cfg.radar.poll_low_freq_seconds
            ):
                last_discovery = time.monotonic()

                async def _run_discovery(sweep_no: int = sweep) -> None:
                    try:
                        discovered = await asyncio.wait_for(_discover_new_companies(), timeout=300)
                        await _persist_discovered_sources(store, discovered)
                        await persist_checkpoints(store)
                        if ta.is_configured:
                            await ta.send_stage_progress(
                                f"Sweep {sweep_no}: Company Discovery",
                                f"Surfaced {len(discovered)} potential company sources.",
                            )
                    except Exception as exc:
                        logger.warning(f"Company discovery failed: {exc}")

                asyncio.create_task(_run_discovery())

            # 5. Periodic graph metrics (embeddings, PageRank, WCC, betweenness)
            if not is_worker and time.monotonic() - last_graph_metrics > _graph_metrics_interval:
                last_graph_metrics = time.monotonic()
                try:
                    scores = await graph.update_all_graph_metrics()
                    logger.info(f"Graph metrics: {scores}")
                except Exception as gerr:
                    logger.debug(f"Graph metrics update failed: {gerr}")

            candidates, gate_stats = await _fetch_postings_and_gate(all_obs, store)
            set_pipeline_state(scraped=len(all_obs), gated=len(candidates))
            logger.info(f"Gating: {len(candidates)} passed, {gate_stats['rejected']} rejected")
            gs = gate_stats.get("gate_stats", {})
            if gs:
                logger.info(f"Gate rejection breakdown: {gs}")
            if ta.is_configured:
                await ta.send_stage_progress(
                    f"Sweep {sweep}: Source Polling & Gating",
                    f"Polled {len(active_sources)} sources ({len(all_obs)} jobs). "
                    f"{len(candidates)} passed gating filter "
                    f"({gate_stats.get('rejected', 0)} rejected).",
                )

            if not is_worker:
                console.rule(f"[bold cyan]RADAR PHASE 3 (sweep {sweep}): LLM Matching[/bold cyan]")
            resume_ctx = full_text[:3000] if full_text else candidate_persona

            ranked = _rank_for_queue(candidates)
            logger.info(f"Sweep {sweep}: enqueuing {len(ranked)} candidates for LLM matching...")
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
                await enqueue_candidate(c, priority=prio, store=store)

            matched = await process_queue(
                ctx,
                resume_ctx,
                candidate_persona,
                store,
                max_candidates=cfg.radar.max_candidates_per_sweep,
            )
            logger.info(f"LLM queue: {len(matched)} matched")
            accepted = len([c for c in matched if c.is_accepted])
            logger.info(f"LLM queue: {len(matched)} total matched, {accepted} accepted")

            await _enrich_high_fit(matched, sa, store, graph)
            enriched_count = len([c for c in matched if c.is_accepted])
            logger.info(f"Sweep {sweep}: enriched {enriched_count} accepted candidates")
            await _dispatch_company_events(matched, graph, bus)
            actual_sent = await _notify_telegram(ta, matched, store)
            stealth_sent = await _check_and_notify_stealth_signals(ta, graph, store)
            accepted_count = len([c for c in matched if c.is_accepted])
            logger.info(
                f"Sweep {sweep}: {actual_sent + stealth_sent} Telegram alerts sent"
                f" ({actual_sent} jobs, {stealth_sent} stealth signals, {accepted_count} accepted)"
            )

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
            logger.info(
                f"=== Sweep {sweep} complete in {elapsed:.1f}s: "
                f"{accepted} accepted, {len(all_obs)} observations ===",
            )
            if ta.is_configured:
                await ta.send_sweep_summary(sweep, accepted, len(all_obs), elapsed)

            # Auto-backup: snapshot volumes after each sweep, keep latest 10.
            # Runs detached so it never blocks the next sweep cycle.
            if os.environ.get("AUTO_BACKUP", "true").lower() != "false":
                try:
                    import subprocess as _sp

                    _root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    )
                    _sp.Popen(
                        f"cd {_root} && PYTHONPATH={_root} "
                        f"nohup uv run python scripts/auto_backup.py "
                        f">> logs/auto_backup.log 2>&1 &",
                        shell=True,
                        start_new_session=True,
                    )
                    logger.info(f"Sweep {sweep}: auto-backup triggered (keep 10)")
                except Exception as bexc:
                    logger.warning(f"auto-backup trigger failed: {bexc}")

            gc.collect()
            if os.environ.get("OVERNIGHT_LOOP", "true").lower() != "true":
                break
            interval = cfg.pipeline.sweep_interval
            logger.info(f"Sweep {sweep}: sleeping for {interval}s before next sweep")
            await asyncio.sleep(interval)
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
