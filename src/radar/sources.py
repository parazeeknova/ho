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
from src.radar.models import SourceCheckpoint, SourceState

logger = get_logger("radar_sources")

_SOURCE_CHECKPOINTS: dict[str, SourceCheckpoint] = {}
_LAST_SNAPSHOT_URLS: dict[str, set[str]] = {}


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
    else:
        checkpoint.consecutive_empty += 1
        checkpoint.last_polled = time.time()

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
                         company_name, discovery_origin)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
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
                        total_direct_url_rate = EXCLUDED.total_direct_url_rate
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
    if cp.consecutive_failures >= 5:
        cp.active = False
        cp.backoff_until = time.time() + 3600
        logger.warning("Source disabled due to consecutive failures", source=source_id)


def record_success(source_id: str, job_count: int, direct_url_count: int) -> None:
    cp = get_checkpoint(source_id)
    cp.consecutive_failures = 0
    cp.consecutive_empty = 0
    cp.total_jobs_produced += job_count
    if job_count > 0:
        cp.total_direct_url_rate = direct_url_count / job_count
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
    return not cp.quality_score < cfg.source_min_confidence


def get_source_health() -> dict[str, dict[str, Any]]:
    return {
        sid: {
            "type": cp.source_type,
            "active": cp.active,
            "quality_score": round(cp.quality_score, 3),
            "last_polled_ago_seconds": round(time.time() - cp.last_polled, 1)
            if cp.last_polled
            else -1,
            "consecutive_failures": cp.consecutive_failures,
            "consecutive_empty": cp.consecutive_empty,
            "jobs_produced": cp.total_jobs_produced,
        }
        for sid, cp in _SOURCE_CHECKPOINTS.items()
    }
