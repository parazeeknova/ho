"""LLM matcher work queue. Budget control is handled by the shared
radar.governor; this module just manages queue dedup, ordering, and
worker dispatch.

Candidates are matched in two passes:

- Pass 1 (cheap vector gate): the JD is embedded and run against the
  pgvector store of resume chunks (``search_similar_chunks``). JDs whose
  average cosine similarity to the resume falls below the configured
  threshold are rejected instantly, without touching the LLM.
- Pass 2 (expensive LLM): only high-similarity JDs are sent to the
  GeneralCompute LLM for the final verdict and skill extraction.

If the gate cannot run (no store, empty embedding index, embed server
down) the candidate passes through to the LLM so infra hiccups never
drop candidates.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Any

from src.configuration import get_config
from src.llm.context import ContextManager
from src.logging import get_logger
from src.radar.core.governor import _is_429, handle_429
from src.radar.core.models import EligibilityState, JobCandidate, RejectionReason

logger = get_logger("llm_queue")


@dataclass
class QueueState:
    pending: list[tuple[int, int, JobCandidate]] = field(default_factory=list)
    total_enqueued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_429s: int = 0
    total_vector_rejects: int = 0


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
You are a job-resume matching engine.

Candidate profile:
{candidate_persona}

Resume skills context:
{resume_context}

Job listing:
{job_markdown}

If this is a company homepage, job directory, error page, or lists
multiple jobs, set match_percent=0 and verdict=NO_MATCH.

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
        seq = _queue_state.total_enqueued
        heapq.heappush(_queue_state.pending, (-priority, seq, candidate))
        _queue_state.total_enqueued += 1
        _queue_not_empty.set()
    return True


def mark_retry(candidate: JobCandidate) -> None:
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
    budget_per_call = cfg.match_token_budget

    async def _worker(candidate: JobCandidate) -> None:
        try:
            if cfg.vector_gate_enabled:
                similarity = await _vector_gate_similarity(candidate, store)
                if similarity is not None:
                    candidate.extra["vector_similarity"] = round(similarity, 4)
                    if similarity < cfg.vector_gate_threshold:
                        candidate.match_percent = 0
                        candidate.verdict = "NO_MATCH"
                        candidate.eligibility = EligibilityState.REJECTED
                        candidate.rejection_reason = RejectionReason.VECTOR_GATE
                        logger.info(
                            f"Vector gate: {candidate.normalized_role} at "
                            f"{candidate.normalized_company} -> {similarity:.3f} "
                            f"(below {cfg.vector_gate_threshold}), skipped LLM",
                        )
                        async with _queue_lock:
                            _queue_state.total_completed += 1
                            _queue_state.total_vector_rejects += 1
                        results.append(candidate)
                        if store is not None:
                            await _persist_candidate(store, candidate)
                        return

            jd = candidate.extra.get("raw_markdown", "")[:8000]

            prompt = MATCHER_PROMPT.replace("{candidate_persona}", candidate_persona)
            prompt = prompt.replace("{resume_context}", resume_context[:3000])
            prompt = prompt.replace("{job_markdown}", jd)

            result = await ctx.json_chat(
                prompt,
                schema=MATCHER_SCHEMA,
                max_tokens=budget_per_call,
            )

            if isinstance(result, dict) and "match_percent" in result:
                _apply_llm_result(candidate, result)
                logger.info(
                    f"LLM match: {candidate.normalized_role} at "
                    f"{candidate.normalized_company} -> {candidate.match_percent}% "
                    f"({candidate.verdict})",
                )
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
                await handle_429()
                _queue_state.total_429s += 1
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

    tasks: list[asyncio.Task[None]] = []
    processed = 0
    while processed < max_candidates:
        entry = await _dequeue()
        if entry is None:
            break
        _, _, candidate = entry
        candidate.eligibility = EligibilityState.QUEUED
        tasks.append(asyncio.create_task(_worker(candidate)))
        processed += 1

    if tasks:
        logger.info(f"LLM queue: dispatching {len(tasks)} candidates for matching...")
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(
            f"LLM queue: {len(tasks)} candidates matched "
            f"({len([r for r in results if r.is_accepted])} accepted, "
            f"{len([r for r in results if r.is_rejected])} rejected)",
        )

    return results


async def _vector_gate_similarity(candidate: JobCandidate, store: Any) -> float | None:
    """Pass 1: cheap pgvector gate against the resume chunk store.

    Returns the JD's average cosine similarity to the top resume chunks,
    or ``None`` when the gate cannot run (no store, no chunk index, embed
    server down). Callers must pass ``None`` through to the LLM.
    """
    if store is None or not hasattr(store, "search_similar_chunks"):
        return None

    jd = candidate.extra.get("raw_markdown", "")[:4000]
    if not jd.strip():
        return None

    from src.agent.enrichment_agent import _get_embedding

    jd_vector = await _get_embedding(jd)
    if jd_vector is None:
        return None

    chunks = await store.search_similar_chunks(jd_vector, top_k=5)
    if not chunks:
        return None

    similarities = [1.0 - ch.get("distance", 1.0) for ch in chunks]
    similarities = [s for s in similarities if s >= 0.0]
    if not similarities:
        return None
    return sum(similarities) / len(similarities)


async def _dequeue() -> tuple[int, int, JobCandidate] | None:
    loop = asyncio.get_running_loop()
    if getattr(_queue_not_empty, "_loop", None) is not loop:
        _queue_not_empty._loop = loop
    try:
        await asyncio.wait_for(_queue_not_empty.wait(), timeout=5.0)
    except TimeoutError:
        return None
    async with _queue_lock:
        if not _queue_state.pending:
            _queue_not_empty.clear()
            return None
        entry = heapq.heappop(_queue_state.pending)
        if not _queue_state.pending:
            _queue_not_empty.clear()
        return entry


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

    from src.radar.core.salary import normalize_salary

    raw_salary = result.get("salary")
    if raw_salary:
        candidate.salary = normalize_salary(str(raw_salary))

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
    from src.radar.core.models import make_canonical_id

    return make_canonical_id(
        candidate.normalized_company,
        candidate.normalized_role,
        candidate.normalized_location,
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
    from src.radar.core.governor import get_governor_status as _gs

    gs = _gs()
    return {
        "pending": len(_queue_state.pending),
        "in_flight": gs["in_flight"],
        "requests_this_minute": gs["requests_this_minute"],
        "rpm_limit": gs["rpm_limit"],
        "tokens_this_minute": gs["tokens_this_minute"],
        "tpm_limit": gs["tpm_limit"],
        "cooldown_active": gs["cooldown_active"],
        "total_enqueued": _queue_state.total_enqueued,
        "total_completed": _queue_state.total_completed,
        "total_failed": _queue_state.total_failed,
        "total_vector_rejects": _queue_state.total_vector_rejects,
        "total_429s": gs["total_429s"] + _queue_state.total_429s,
    }
