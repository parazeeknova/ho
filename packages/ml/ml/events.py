"""Unified event stream — the foundation of the learning system.

Every ranking/recommendation decision is an impression (group of jobs competing
in the same context). Every downstream outcome (click, save, apply, screen,
interview, offer) is a separate reward event linked by job_id + impression_id.
This gives credit assignment, handles delayed rewards, and fixes selection bias
by distinguishing never-seen / seen-not-selected / selected-ignored / applied.

Impression is the ranking unit for LambdaMART (not date) — #8.
Candidate/job snapshots prevent profile drift — #5.
Propensity is logged on every action for counterfactual evaluation — #4.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import (
    EMBEDDING_VERSION,
    FEATURE_VERSION,
    POLICY_VERSION,
    PROMPT_VERSION,
    RANKER_VERSION,
)

# Helpers


def make_impression_id() -> str:
    return f"imp_{uuid.uuid4().hex[:8]}"


def make_snapshot_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# Event dataclass (also the DB row shape)


@dataclass
class DecisionEvent:
    job_id: str
    event_type: str
    # Linkage
    impression_id: str | None = None
    candidate_id: str | None = None
    # Snapshot IDs — what the model saw at decision time (#5)
    candidate_snapshot_id: str | None = None
    job_snapshot_id: str | None = None
    # Features at decision time (for training — temporal leakage guarded)
    features: dict[str, Any] = field(default_factory=dict)
    # Decision context
    rank: int | None = None
    policy: str = POLICY_VERSION
    model_version: str = RANKER_VERSION
    feature_version: str = FEATURE_VERSION
    prompt_version: str = PROMPT_VERSION
    embedding_version: str = EMBEDDING_VERSION
    exploration: bool = False
    propensity: float | None = None  # P(action|context) for IPS/SNIPS (#4)
    # Behavior-policy provenance (P1 fix): the behavior policy name and the
    # probability the BEHAVIOR policy assigned to this action (μ). Kept
    # separate from `propensity` (and from any evaluation π(a|x) computed
    # later) so offline evaluation can compute its own π without overwriting
    # the logged behavior propensity.
    behavior_policy: str = ""
    behavior_propensity: float | None = None
    action: str | None = None
    reward: float | None = None
    # Source attribution (#11)
    source: str | None = None
    query: str | None = None
    primary_discovery_source: str | None = None
    secondary_sources: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# DB helpers — thin wrappers over MemoryStore pool so callers don't import
# asyncpg directly. Event writes are DURABLE: a failed insert is retried once
# and then written to a dead-letter table (event_write_errors) so ground-truth
# events are never silently dropped. Losing an `application` but keeping a
# `rejection` would corrupt funnel histories, so this is not optional.

# In-memory retry buffer for events whose DB write transiently failed.
_failed_events: list[DecisionEvent] = []


async def _write_one(conn: Any, event: DecisionEvent) -> None:
    await conn.execute(
        """
        INSERT INTO decision_events (
            job_id, event_type, impression_id, candidate_id,
            candidate_snapshot_id, job_snapshot_id,
            features, rank, policy, model_version, feature_version,
            prompt_version, embedding_version,
            exploration, propensity, behavior_policy, behavior_propensity,
            action, reward,
            source, query, primary_discovery_source, secondary_sources,
            meta, created_at
        ) VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
            $19,$20,$21,$22,$23, NOW()
        )
        """,
        event.job_id,
        event.event_type,
        event.impression_id,
        event.candidate_id,
        event.candidate_snapshot_id,
        event.job_snapshot_id,
        event.features,  # jsonb codec serializes dicts; do NOT pre-json.dumps
        event.rank,
        event.policy,
        event.model_version,
        event.feature_version,
        event.prompt_version,
        event.embedding_version,
        event.exploration,
        event.propensity,
        event.behavior_policy,
        event.behavior_propensity,
        event.action,
        event.reward,
        event.source,
        event.query,
        event.primary_discovery_source,
        event.secondary_sources,
        event.meta,
    )


async def _dead_letter(store: Any, event: DecisionEvent, error: str) -> None:
    """Record a permanently-failed event write so it's observable, not lost."""
    try:
        async with store._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO event_write_errors (
                    job_id, event_type, impression_id, payload, error, created_at
                ) VALUES ($1,$2,$3,$4,$5, NOW())
                """,
                event.job_id,
                event.event_type,
                event.impression_id,
                asdict(event),
                error[:500],
            )
    except Exception:
        # If even the dead-letter table is unreachable, keep it in the retry
        # buffer so a later flush can salvage it.
        _failed_events.append(event)


async def flush_failed_events(store: Any) -> int:
    """Retry buffered events that failed a previous write. Returns flushed count."""
    if not _failed_events:
        return 0
    events = list(_failed_events)
    _failed_events.clear()
    flushed = 0
    for ev in events:
        try:
            async with store._pool.acquire() as conn:
                await _write_one(conn, ev)
            flushed += 1
        except Exception as e:
            await _dead_letter(store, ev, str(e))
    return flushed


async def emit_event(store: Any, event: DecisionEvent) -> None:
    """Persist one decision event (durable). Retries once, then dead-letters."""
    try:
        async with store._pool.acquire() as conn:
            await _write_one(conn, event)
    except Exception:
        # Retry once (transient DB blip).
        try:
            async with store._pool.acquire() as conn:
                await _write_one(conn, event)
        except Exception as e2:
            await _dead_letter(store, event, str(e2))
        # First attempt may have partially succeeded on the server; treat as
        # flushed if the retry succeeded. If both failed, it's dead-lettered.


async def emit_events(store: Any, events: list[DecisionEvent]) -> None:
    for ev in events:
        await emit_event(store, ev)


# Snapshot helpers


def snapshot_candidate(profile: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Freeze candidate state at decision time — prevents profile-drift leakage."""
    snap = {
        "skills": profile.get("skills") or profile.get("matching_skills") or [],
        "experience_years": profile.get("experience_years"),
        "preferences": {
            "role_family": profile.get("role_family"),
            "location": profile.get("location"),
            "salary_floor": profile.get("salary_floor"),
            "remote": profile.get("is_remote"),
        },
        "role": profile.get("role") or profile.get("normalized_role"),
        "version": profile.get("version") or profile.get("persona_version"),
    }
    return make_snapshot_id(snap), snap


def snapshot_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    snap = {
        "role_family": job.get("role_family"),
        "skills": job.get("matching_skills") or [],
        "missing_skills": job.get("missing_skills") or [],
        "salary": job.get("salary") or job.get("salary_amount"),
        "location": job.get("normalized_location") or job.get("location"),
        "source": job.get("source"),
        "funding_stage": job.get("funding_stage"),
        "company": job.get("normalized_company") or job.get("company"),
    }
    return make_snapshot_id(snap), snap
