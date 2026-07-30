"""LLM work queue with process-wide rate limiting, budget tracking,
and graceful 429 handling.

Replaces the current 24-way matching burst with a single budget-controlled
queue. Only candidates that survive deterministic filtering and rank near
the top consume LLM budget.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.configuration import LlmQueueConfig, get_config
from src.llm.context import ContextManager
from src.logging import get_logger
from src.radar.models import EligibilityState, JobCandidate, RejectionReason

logger = get_logger("llm_queue")


@dataclass
class QueueState:
    pending: deque = field(default_factory=deque)
    in_flight: int = 0
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    window_start: float = field(default_factory=time.monotonic)
    cooldown_until: float = 0.0
    total_enqueued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_429s: int = 0


_queue_state = QueueState()
_queue_lock = asyncio.Lock()
_queue_not_empty = asyncio.Event()

_ACTIVE_IDS: dict[str, float] = {}
_CANDIDATE_VERSIONS: dict[str, int] = {}
_ID_LOCK = asyncio.Lock()

MATCHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "role": {"type": "string"},
        "match_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "shortlist_probability": {"type": "integer", "minimum": 0, "maximum": 100},
        "matching_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "verdict": {
            "type": "string",
            "enum": ["STRONG_MATCH", "GOOD_MATCH", "WEAK_MATCH", "NO_MATCH"],
        },
        "jd_summary": {"type": "string"},
        "company_description": {"type": "string"},
        "role_summary": {"type": "string"},
        "salary": {"type": ["string", "null"]},
        "posted_date": {"type": ["string", "null"]},
        "location": {"type": "string"},
        "is_remote": {"type": "boolean"},
    },
    "required": [
        "company",
        "role",
        "match_percent",
        "shortlist_probability",
        "verdict",
        "matching_skills",
        "missing_skills",
    ],
}

MATCHER_PROMPT = """\
You are a job-resume matching engine. Evaluate this candidate against the job description.

Candidate profile:
{candidate_persona}

Resume skills context:
{resume_context}

Job listing:
{job_markdown}

If the text is a company homepage, job directory, error page, or lists multiple
different jobs, set match_percent=0 and verdict=NO_MATCH.

Return valid JSON matching the required schema.
"""


async def enqueue_candidate(candidate: JobCandidate, priority: int = 50) -> bool:
    key = f"{candidate.canonical_id}:v{candidate.extra.get('version', 1)}"
    async with _ID_LOCK:
        if key in _ACTIVE_IDS:
            return False
        _ACTIVE_IDS[key] = time.monotonic()
        _CANDIDATE_VERSIONS[candidate.canonical_id] = candidate.extra.get("version", 1)
    async with _queue_lock:
        _queue_state.pending.append((priority, candidate))
        _queue_state.total_enqueued += 1
        _queue_state.pending = deque(sorted(_queue_state.pending, key=lambda x: x[0], reverse=True))
        _queue_not_empty.set()
    return True


def _clear_queued(candidate: JobCandidate) -> None:
    key = f"{candidate.canonical_id}:v{candidate.extra.get('version', 1)}"
    _ACTIVE_IDS.pop(key, None)
    can_ver = _CANDIDATE_VERSIONS.get(candidate.canonical_id, 0)
    candidate_ver = candidate.extra.get("version", 1)
    if candidate_ver >= can_ver:
        _CANDIDATE_VERSIONS.pop(candidate.canonical_id, None)


def mark_retry(candidate: JobCandidate) -> None:
    """Allow a 429-retry candidate to be requeued."""
    key = f"{candidate.canonical_id}:v{candidate.extra.get('version', 1)}"
    _ACTIVE_IDS.pop(key, None)


async def process_queue(
    ctx: ContextManager,
    resume_context: str,
    candidate_persona: str,
    store,
    max_candidates: int = 50,
) -> list[JobCandidate]:
    cfg = get_config().llm_queue
    results: list[JobCandidate] = []
    sem = asyncio.Semaphore(cfg.max_in_flight)

    async def _worker(candidate: JobCandidate) -> None:
        async with sem:
            try:
                await _acquire_budget(cfg)
            except Exception:
                _clear_queued(candidate)
                return

            try:
                jd = candidate.extra.get("raw_markdown", "")[:8000]
                output_budget = cfg.match_token_budget

                prompt = MATCHER_PROMPT.replace("{candidate_persona}", candidate_persona)
                prompt = prompt.replace("{resume_context}", resume_context[:3000])
                prompt = prompt.replace("{job_markdown}", jd)

                result = await ctx.json_chat(
                    prompt,
                    schema=MATCHER_SCHEMA,
                    max_tokens=output_budget,
                )

                if isinstance(result, dict) and "match_percent" in result:
                    _apply_llm_result(candidate, result)
                else:
                    candidate.eligibility = EligibilityState.ERROR
                    candidate.rejection_reason = RejectionReason.MATCHER_LOW_SCORE

                async with _queue_lock:
                    _queue_state.total_completed += 1
                results.append(candidate)

                if store is not None:
                    await _persist_candidate(store, candidate)

            except Exception as e:
                err_msg = str(e)
                async with _queue_lock:
                    _queue_state.total_failed += 1
                if _is_429(err_msg):
                    await _handle_429(cfg)
                    attempts = candidate.extra.get("queue_attempts", 0) + 1
                    candidate.extra["queue_attempts"] = attempts
                    if attempts <= 3:
                        mark_retry(candidate)
                        await enqueue_candidate(candidate, priority=30)
                    else:
                        candidate.eligibility = EligibilityState.ERROR
                        if store is not None:
                            await _persist_candidate(store, candidate)
                else:
                    candidate.eligibility = EligibilityState.ERROR
                    candidate.rejection_reason = RejectionReason.UNKNOWN
                    if store is not None:
                        await _persist_candidate(store, candidate)
                    logger.warning("LLM queue worker failed", exception=err_msg)
            finally:
                async with _queue_lock:
                    _queue_state.in_flight -= 1

    tasks: list[asyncio.Task[None]] = []
    processed = 0
    while processed < max_candidates:
        entry = await _dequeue()
        if entry is None:
            break
        _, candidate = entry
        candidate.eligibility = EligibilityState.QUEUED
        tasks.append(asyncio.create_task(_worker(candidate)))
        processed += 1

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    return results


async def _acquire_budget(cfg: LlmQueueConfig) -> None:
    while True:
        now = time.monotonic()
        wait_secs: float | None = None

        async with _queue_lock:
            if _queue_state.cooldown_until > 0 and now < _queue_state.cooldown_until:
                wait_secs = _queue_state.cooldown_until - now
            elif now - _queue_state.window_start >= 60.0:
                _queue_state.window_start = now
                _queue_state.requests_this_minute = 0
                _queue_state.tokens_this_minute = 0
            elif (
                _queue_state.requests_this_minute >= cfg.requests_per_minute
                or _queue_state.tokens_this_minute + cfg.match_token_budget
                > cfg.estimated_tokens_per_minute
            ):
                wait_secs = 60.0 - (now - _queue_state.window_start)
            else:
                _queue_state.requests_this_minute += 1
                _queue_state.tokens_this_minute += cfg.match_token_budget
                _queue_state.in_flight += 1
                return

        if wait_secs is not None:
            await asyncio.sleep(wait_secs + 0.1)


async def _dequeue() -> tuple[int, JobCandidate] | None:
    try:
        await asyncio.wait_for(_queue_not_empty.wait(), timeout=5.0)
    except TimeoutError:
        return None

    async with _queue_lock:
        if not _queue_state.pending:
            _queue_not_empty.clear()
            return None
        entry = _queue_state.pending.popleft()
        if not _queue_state.pending:
            _queue_not_empty.clear()
        return entry


async def _handle_429(cfg: LlmQueueConfig) -> None:
    async with _queue_lock:
        _queue_state.total_429s += 1
        cooldown = cfg.cooldown_seconds + random.uniform(0, cfg.jitter_seconds)
        _queue_state.cooldown_until = max(
            _queue_state.cooldown_until,
            time.monotonic() + cooldown,
        )
        logger.warning("LLM queue: 429 received, cooldown", seconds=cooldown)


def _is_429(err_msg: str) -> bool:
    return (
        "429" in err_msg
        or "rate limit" in err_msg.lower()
        or "too many requests" in err_msg.lower()
    )


def _apply_llm_result(candidate: JobCandidate, result: dict[str, Any]) -> None:
    candidate.normalized_role = str(result.get("role", candidate.normalized_role))
    candidate.normalized_company = str(result.get("company", candidate.normalized_company))
    candidate.match_percent = int(result.get("match_percent", 0))
    candidate.shortlist_probability = int(result.get("shortlist_probability", 0))
    candidate.matching_skills = result.get("matching_skills", []) or []
    candidate.missing_skills = result.get("missing_skills", []) or []
    candidate.verdict = str(result.get("verdict", "NO_MATCH"))
    candidate.jd_summary = str(result.get("jd_summary", ""))
    candidate.company_description = str(result.get("company_description", ""))
    candidate.role_summary = str(result.get("role_summary", ""))
    candidate.is_remote = bool(result.get("is_remote", False))
    candidate.normalized_location = str(result.get("location", "Remote"))

    from src.radar.salary import normalize_salary

    raw_salary = result.get("salary")
    if raw_salary:
        candidate.salary = normalize_salary(str(raw_salary))

    # Keep posting_id as immutable canonical ID; store group key separately.
    # canonical_id is the URL hash, never overwritten by company/role/location.
    candidate.extra["group_key"] = _build_group_key(candidate)

    verdict = candidate.verdict
    if verdict == "NO_MATCH" or candidate.match_percent < 30:
        candidate.eligibility = EligibilityState.REJECTED
        candidate.rejection_reason = RejectionReason.MATCHER_NO_MATCH
    elif verdict in ("STRONG_MATCH", "GOOD_MATCH"):
        candidate.eligibility = EligibilityState.ACCEPTED
    elif verdict == "WEAK_MATCH":
        candidate.eligibility = EligibilityState.NEAR_MISS


def _build_group_key(candidate: JobCandidate) -> str:
    from src.radar.models import make_canonical_id

    return make_canonical_id(
        candidate.normalized_company,
        candidate.normalized_role,
        candidate.normalized_location,
    )


def _build_canonical_from_result(
    candidate: JobCandidate,
    result: dict[str, Any],
) -> str:
    from src.radar.models import make_canonical_id

    return make_canonical_id(
        str(result.get("company", candidate.normalized_company)),
        str(result.get("role", candidate.normalized_role)),
        str(result.get("location", candidate.normalized_location)),
    )


async def _persist_candidate(store, candidate: JobCandidate) -> None:
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
    except Exception as e:
        logger.warning(
            "Failed to persist radar candidate",
            canonical_id=candidate.canonical_id,
            exception=str(e),
        )


def get_queue_status() -> dict[str, Any]:
    return {
        "pending": len(_queue_state.pending),
        "in_flight": _queue_state.in_flight,
        "requests_this_minute": _queue_state.requests_this_minute,
        "tokens_this_minute": _queue_state.tokens_this_minute,
        "cooldown_active": _queue_state.cooldown_until > time.monotonic(),
        "total_enqueued": _queue_state.total_enqueued,
        "total_completed": _queue_state.total_completed,
        "total_failed": _queue_state.total_failed,
        "total_429s": _queue_state.total_429s,
    }
