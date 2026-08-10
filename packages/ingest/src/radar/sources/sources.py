"""Source checkpoint persistence, snapshot/diff, quality scoring, and backoff.

Each source (ATS board, company career page, GitHub index, SearXNG query)
is tracked independently with quality scores that drift over time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from typing import Any

from src.configuration import get_config
from src.logging import get_logger
from src.radar.core.models import SourceCheckpoint, SourceState

logger = get_logger("radar_sources")

_SOURCE_CHECKPOINTS: dict[str, SourceCheckpoint] = {}
_LAST_SNAPSHOT_URLS: dict[str, set[str]] = {}

# Poll-lane tuning: how many consecutive unchanged polls before demoting a
# lane, and how often each lane is allowed to be polled (× sweep interval).
_LANE_DEMOTE_EMPTY = {5: "medium", 15: "low"}  # consecutive_empty -> lane
_LANE_INTERVAL_MULT = {"high": 0, "medium": 3, "low": 10}  # × sweep_interval
_LOW_LANE_FLOOR = 5  # min low-lane sources polled per sweep (never starve)
_YIELD_ALPHA = 0.3  # EWMA weight for yield_per_poll


def _demote_lane(cp: SourceCheckpoint) -> None:
    if cp.poll_lane == "high":
        cp.poll_lane = "medium"
    elif cp.poll_lane == "medium":
        cp.poll_lane = "low"


def _lane_rank(lane: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(lane, 0)


def _ewma(prev: float, value: float) -> float:
    """Exponential moving average: prev*(1-a) + value*a."""
    return prev * (1.0 - _YIELD_ALPHA) + value * _YIELD_ALPHA


def get_checkpoint(source_id: str) -> SourceCheckpoint:
    if source_id not in _SOURCE_CHECKPOINTS:
        _SOURCE_CHECKPOINTS[source_id] = SourceCheckpoint(
            source_id=source_id,
            source_type="unknown",
        )
    return _SOURCE_CHECKPOINTS[source_id]


def get_all_checkpoints() -> dict[str, SourceCheckpoint]:
    return dict(_SOURCE_CHECKPOINTS)


def register_source(
    source_id: str,
    source_type: str,
    initial_quality: float = 0.5,
) -> SourceCheckpoint:
    if source_id not in _SOURCE_CHECKPOINTS:
        _SOURCE_CHECKPOINTS[source_id] = SourceCheckpoint(
            source_id=source_id,
            source_type=source_type,
            quality_score=initial_quality,
        )
    return _SOURCE_CHECKPOINTS[source_id]


def compute_url_snapshot_hash(urls: list[str]) -> str:
    sorted_urls = sorted(urls)
    return hashlib.sha256(json.dumps(sorted_urls).encode()).hexdigest()[:16]


def diff_snapshots(
    source_id: str,
    current_urls: list[str],
) -> SourceState:
    checkpoint = get_checkpoint(source_id)
    current_set = set(current_urls)
    previous_set = _LAST_SNAPSHOT_URLS.get(source_id, set())

    new_hashes: list[str] = []
    removed_hashes: list[str] = []

    if previous_set:
        new_hashes = list(current_set - previous_set)
        removed_hashes = list(previous_set - current_set)
    else:
        new_hashes = list(current_set)

    new_hash = compute_url_snapshot_hash(current_urls)

    if new_hash != checkpoint.last_snapshot_hash:
        checkpoint.last_snapshot_hash = new_hash
        checkpoint.last_snapshot_count = len(current_urls)
        checkpoint.last_polled = time.time()
        checkpoint.consecutive_empty = 0
        checkpoint.last_change_at = time.time()
        checkpoint.poll_lane = "high"  # content changed -> promote
        if current_urls:
            checkpoint.yield_per_poll = _ewma(checkpoint.yield_per_poll, len(current_urls))
    else:
        checkpoint.consecutive_empty += 1
        checkpoint.last_polled = time.time()
        demoted_to = _LANE_DEMOTE_EMPTY.get(checkpoint.consecutive_empty)
        if demoted_to and _lane_rank(demoted_to) > _lane_rank(checkpoint.poll_lane):
            checkpoint.poll_lane = demoted_to
            logger.info(
                "Source demoted for unchanged snapshots",
                source=source_id,
                lane=demoted_to,
                consecutive_empty=checkpoint.consecutive_empty,
            )

    _LAST_SNAPSHOT_URLS[source_id] = current_set

    return SourceState(
        checkpoint=checkpoint,
        current_urls=current_set,
        new_urls=new_hashes,
        removed_urls=removed_hashes,
    )


async def persist_checkpoints(store) -> None:
    """Persist all checkpoint state to Postgres."""
    if store is None:
        return
    async with store._pool.acquire() as conn:
        for source_id, cp in _SOURCE_CHECKPOINTS.items():
            with contextlib.suppress(Exception):
                await conn.execute(
                    """
                    INSERT INTO source_checkpoints
                        (source_id, source_type, board_url, last_polled,
                         last_snapshot_hash, last_snapshot_count,
                         consecutive_failures, consecutive_empty,
                         quality_score, active, backoff_until,
                         total_jobs_produced, total_direct_url_rate,
                         company_name, discovery_origin,
                         poll_lane, yield_per_poll, last_change_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                            $16,$17,$18)
                    ON CONFLICT (source_id) DO UPDATE SET
                        board_url = EXCLUDED.board_url,
                        last_polled = EXCLUDED.last_polled,
                        last_snapshot_hash = EXCLUDED.last_snapshot_hash,
                        last_snapshot_count = EXCLUDED.last_snapshot_count,
                        consecutive_failures = EXCLUDED.consecutive_failures,
                        consecutive_empty = EXCLUDED.consecutive_empty,
                        quality_score = EXCLUDED.quality_score,
                        active = EXCLUDED.active,
                        backoff_until = EXCLUDED.backoff_until,
                        total_jobs_produced = EXCLUDED.total_jobs_produced,
                        total_direct_url_rate = EXCLUDED.total_direct_url_rate,
                        poll_lane = EXCLUDED.poll_lane,
                        yield_per_poll = EXCLUDED.yield_per_poll,
                        last_change_at = EXCLUDED.last_change_at
                    """,
                    source_id,
                    cp.source_type,
                    getattr(cp, "board_url", ""),
                    cp.last_polled,
                    cp.last_snapshot_hash,
                    cp.last_snapshot_count,
                    cp.consecutive_failures,
                    cp.consecutive_empty,
                    cp.quality_score,
                    cp.active,
                    cp.backoff_until,
                    cp.total_jobs_produced,
                    cp.total_direct_url_rate,
                    getattr(cp, "company_name", ""),
                    getattr(cp, "discovery_origin", ""),
                    cp.poll_lane,
                    cp.yield_per_poll,
                    cp.last_change_at,
                )
        for source_id, urls in _LAST_SNAPSHOT_URLS.items():
            with contextlib.suppress(Exception):
                snapshot_json = json.dumps(sorted(urls))
                await conn.execute(
                    """
                    INSERT INTO source_snapshots (source_id, snapshot_data)
                    VALUES ($1, $2)
                    ON CONFLICT (source_id) DO UPDATE SET
                        snapshot_data = EXCLUDED.snapshot_data, updated_at = NOW()
                    """,
                    source_id,
                    snapshot_json,
                )


async def load_checkpoints(store) -> None:
    """Load checkpoint state from Postgres."""
    if store is None:
        return
    async with store._pool.acquire() as conn:
        with contextlib.suppress(Exception):
            rows = await conn.fetch("SELECT * FROM source_checkpoints")
            for row in rows:
                _SOURCE_CHECKPOINTS[row["source_id"]] = SourceCheckpoint(
                    source_id=row["source_id"],
                    source_type=row["source_type"],
                    last_polled=row["last_polled"] or 0.0,
                    last_snapshot_hash=row["last_snapshot_hash"] or "",
                    last_snapshot_count=row["last_snapshot_count"] or 0,
                    consecutive_failures=row["consecutive_failures"] or 0,
                    consecutive_empty=row["consecutive_empty"] or 0,
                    quality_score=row["quality_score"] or 0.5,
                    active=row.get("active", True),
                    backoff_until=row["backoff_until"] or 0.0,
                    total_jobs_produced=row["total_jobs_produced"] or 0,
                    total_direct_url_rate=row["total_direct_url_rate"] or 0.0,
                    board_url=row.get("board_url", "") or "",
                    company_name=row.get("company_name", "") or "",
                    discovery_origin=row.get("discovery_origin", "") or "",
                    poll_lane=row.get("poll_lane", "high") or "high",
                    yield_per_poll=float(row.get("yield_per_poll") or 0.0),
                    last_change_at=float(row.get("last_change_at") or 0.0),
                )
        with contextlib.suppress(Exception):
            snap_rows = await conn.fetch("SELECT source_id, snapshot_data FROM source_snapshots")
            for sr in snap_rows:
                try:
                    urls = set(json.loads(sr["snapshot_data"]))
                    _LAST_SNAPSHOT_URLS[sr["source_id"]] = urls
                except Exception:
                    pass


async def load_active_sources(store) -> list[dict[str, str]]:
    """Return active board sources with their URLs for polling."""
    if store is None:
        return []
    sources: list[dict[str, str]] = []
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source_id, board_url FROM source_checkpoints "
                "WHERE source_type = 'ats_board' AND active = TRUE "
                "AND board_url != ''"
            )
            for r in rows:
                sources.append(
                    {
                        "id": r["source_id"],
                        "url": r["board_url"],
                        "source_type": "official_ats",
                    }
                )
    except Exception:
        pass
    return sources


def record_failure(source_id: str) -> None:
    cp = get_checkpoint(source_id)
    cp.consecutive_failures += 1
    cp.quality_score = max(0.1, cp.quality_score * 0.8)
    if cp.consecutive_failures == 3:
        _demote_lane(cp)
        logger.info("Source demoted after repeated failures", source=source_id, lane=cp.poll_lane)
    if cp.consecutive_failures >= 5:
        cp.active = False
        cp.backoff_until = time.time() + 3600
        logger.warning("Source disabled due to consecutive failures", source=source_id)


def record_success(source_id: str, job_count: int, direct_url_count: int) -> None:
    cp = get_checkpoint(source_id)
    cp.consecutive_failures = 0
    cp.total_jobs_produced += job_count
    cp.yield_per_poll = _ewma(cp.yield_per_poll, float(job_count))
    if job_count > 0:
        cp.consecutive_empty = 0
        cp.total_direct_url_rate = direct_url_count / job_count
        if cp.poll_lane != "high":
            cp.poll_lane = "high"  # productive source -> promote
            logger.info("Source promoted for job yield", source=source_id)
    cp.quality_score = min(1.0, cp.quality_score * 1.1 + 0.02)
    if not cp.active:
        cp.active = True
        cp.backoff_until = 0.0


def should_poll(source_id: str) -> bool:
    cp = get_checkpoint(source_id)
    if not cp.active:
        if cp.backoff_until > 0 and time.time() < cp.backoff_until:
            return False
        cp.active = True
    cfg = get_config().radar
    if cp.quality_score < cfg.source_min_confidence:
        return False
    # Lane frequency gate: medium/low lanes only poll every N sweeps.
    mult = _LANE_INTERVAL_MULT.get(cp.poll_lane, 0)
    if mult > 0 and cp.last_polled > 0:
        sweep = get_config().pipeline.sweep_interval
        if time.time() - cp.last_polled < mult * sweep:
            return False
    return True


def select_sources_for_sweep(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    """Order sources for this sweep: expected-value first, with a low-lane
    floor so quiet-but-important sources are never starved out entirely.
    Never-polled sources (``last_polled == 0``) rank FIRST — they have not
    been given a single chance yet, and without this they sit at EV=0 under
    every already-polled source and never enter the sweep cap.
    """
    scored: list[tuple[float, dict[str, str]]] = []
    for s in sources:
        cp = get_checkpoint(s.get("id", ""))
        age_days = max((time.time() - cp.last_polled) / 86400.0, 0.5)
        ev = cp.yield_per_poll / age_days
        # Never-polled sources always win the ordering.
        if cp.last_polled == 0:
            ev = float("inf")
        scored.append((ev, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [s for _, s in scored]

    lows = [s for s in ordered if get_checkpoint(s.get("id", "")).poll_lane == "low"]
    if lows and len(ordered) > len(lows):
        floor = min(_LOW_LANE_FLOOR, len(lows))
        chosen = lows[:floor]
        rest = [s for s in ordered if s not in chosen]
        return chosen + rest
    return ordered


def get_source_health() -> dict[str, dict[str, Any]]:
    return {
        sid: {
            "type": cp.source_type,
            "active": cp.active,
            "quality_score": round(cp.quality_score, 3),
            "poll_lane": cp.poll_lane,
            "yield_per_poll": round(cp.yield_per_poll, 2),
            "last_polled_ago_seconds": round(time.time() - cp.last_polled, 1)
            if cp.last_polled
            else -1,
            "consecutive_failures": cp.consecutive_failures,
            "consecutive_empty": cp.consecutive_empty,
            "jobs_produced": cp.total_jobs_produced,
        }
        for sid, cp in _SOURCE_CHECKPOINTS.items()
    }
