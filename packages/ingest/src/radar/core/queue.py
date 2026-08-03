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
import json
import re
import time
import traceback
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


def _has_db(store: Any) -> bool:
    return store is not None and hasattr(store, "_pool")


async def enqueue_candidate(candidate: JobCandidate, priority: int = 50, store: Any = None) -> bool:
    key = f"{candidate.canonical_id}:v{candidate.extra.get('version', 1)}"
    async with _ID_LOCK:
        if key in _ACTIVE_IDS:
            return False
        _ACTIVE_IDS[key] = time.monotonic()
        _CANDIDATE_VERSIONS[candidate.canonical_id] = candidate.extra.get("version", 1)

    if _has_db(store):
        try:
            async with store._pool.acquire() as conn:
                res = await conn.execute(
                    """
                    INSERT INTO llm_queue (canonical_id, version, priority, payload)
                    VALUES ($1, $2, $3, $4::jsonb)
                    ON CONFLICT (canonical_id, version) DO NOTHING
                    """,
                    candidate.canonical_id,
                    candidate.extra.get("version", 1),
                    priority,
                    _candidate_to_payload(candidate),
                )
            if res == "INSERT 0 1":
                return True
            # Already queued in the DB: do not also push to memory
            _ACTIVE_IDS.pop(key, None)
            return False
        except Exception as e:
            logger.warning("DB queue enqueue failed, falling back to memory", exception=str(e))
            _ACTIVE_IDS.pop(key, None)

    async with _queue_lock:
        seq = _queue_state.total_enqueued
        heapq.heappush(_queue_state.pending, (-priority, seq, candidate))
        _queue_state.total_enqueued += 1
        _queue_not_empty.set()
    return True


def _candidate_to_payload(c: JobCandidate) -> dict[str, Any]:
    """Minimal JSON-safe payload needed to rebuild a candidate for matching."""
    return {
        "canonical_id": c.canonical_id,
        "source": c.source,
        "direct_apply_url": c.direct_apply_url,
        "normalized_company": c.normalized_company,
        "normalized_role": c.normalized_role,
        "normalized_location": c.normalized_location,
        "extra": c.extra,
    }


def _candidate_from_payload(payload: dict[str, Any] | str) -> JobCandidate:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    if isinstance(payload, str):
        # Legacy rows written while the jsonb codec double-encoded
        # pre-serialized strings: unwrap the nested JSON text once more.
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return JobCandidate(
        canonical_id=payload.get("canonical_id", ""),
        source=payload.get("source", ""),
        direct_apply_url=payload.get("direct_apply_url", ""),
        normalized_company=payload.get("normalized_company", ""),
        normalized_role=payload.get("normalized_role", ""),
        normalized_location=payload.get("normalized_location", "Remote"),
        extra=payload.get("extra", {}),
    )


async def _db_claim(store: Any, limit: int) -> list[tuple[int, int, JobCandidate]]:
    """Claim up to ``limit`` pending queue rows with a 10-minute lease.

    Uses ``FOR UPDATE SKIP LOCKED`` so concurrent workers (or the dedicated
    ``HO_WORKER_ONLY`` process) never double-process the same row.
    """
    claimed: list[tuple[int, int, JobCandidate]] = []
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                UPDATE llm_queue
                SET status = 'processing',
                    lease_until = NOW() + INTERVAL '10 minutes'
                WHERE id IN (
                    SELECT id FROM llm_queue
                    WHERE status = 'pending'
                       OR (status = 'processing' AND lease_until < NOW())
                    ORDER BY priority DESC, id ASC
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, priority, payload
                """,
                limit,
            )
            for r in rows:
                claimed.append((r["id"], r["priority"], _candidate_from_payload(r["payload"])))
    except Exception as e:
        # The UPDATE already flipped the claimed rows to 'processing' with a
        # lease; if parsing/decoding fails, put them back so nothing strands
        # for the full lease duration.
        try:
            async with store._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE llm_queue SET status = 'pending', lease_until = NULL "
                    "WHERE id = ANY($1::bigint[])",
                    [r["id"] for r in rows],
                )
        except Exception:
            pass
        logger.warning(
            "DB queue claim failed",
            exception=str(e),
            traceback=traceback.format_exc(),
        )
    return claimed


async def _db_finish(store: Any, row_id: int, candidate: JobCandidate, outcome: str) -> None:
    """Update a claimed queue row after matching: re-lease for 429 retries,
    or settle it as done/error."""
    try:
        async with store._pool.acquire() as conn:
            if outcome == "retry":
                attempts = candidate.extra.get("queue_attempts", 1)
                if attempts <= 3:
                    await conn.execute(
                        "UPDATE llm_queue SET status = 'pending', priority = 30, "
                        "lease_until = NULL, attempts = $1 WHERE id = $2",
                        attempts,
                        row_id,
                    )
                    return
                outcome = "error"
            await conn.execute(
                "UPDATE llm_queue SET status = $1, completed_at = NOW(), "
                "lease_until = NULL WHERE id = $2",
                "done" if outcome in ("matched", "rejected") else "error",
                row_id,
            )
    except Exception as e:
        logger.warning("DB queue settle failed", exception=str(e))


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
    if _has_db(store):
        return await _process_db_queue(
            ctx, resume_context, candidate_persona, store, max_candidates
        )
    return await _process_memory_queue(
        ctx, resume_context, candidate_persona, store, max_candidates
    )


async def _process_db_queue(
    ctx: ContextManager,
    resume_context: str,
    candidate_persona: str,
    store: Any,
    max_candidates: int = 50,
) -> list[JobCandidate]:
    """Process candidates claimed from the shared Postgres queue.

    The main process enqueues, and either the same process or a dedicated
    ``HO_WORKER_ONLY`` process can match from the same queue.
    """
    cfg = get_config().llm_queue
    results: list[JobCandidate] = []
    claimed = await _db_claim(store, max_candidates)
    if not claimed:
        return results

    async def _worker(row_id: int, candidate: JobCandidate) -> None:
        outcome = "error"
        try:
            outcome = await _match_candidate(
                candidate, ctx, resume_context, candidate_persona, store, cfg, results
            )
        except Exception as e:
            logger.warning("DB queue worker crashed", exception=str(e))
        await _db_finish(store, row_id, candidate, outcome)

    tasks = [asyncio.create_task(_worker(rid, c)) for rid, _, c in claimed]
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info(
        f"DB LLM queue: {len(claimed)} candidates matched "
        f"({len([r for r in results if r.is_accepted])} accepted)",
    )
    return results


async def _process_memory_queue(
    ctx: ContextManager,
    resume_context: str,
    candidate_persona: str,
    store,
    max_candidates: int = 50,
) -> list[JobCandidate]:
    cfg = get_config().llm_queue
    results: list[JobCandidate] = []

    async def _worker(candidate: JobCandidate) -> None:
        outcome = await _match_candidate(
            candidate, ctx, resume_context, candidate_persona, store, cfg, results
        )
        if outcome == "retry":
            mark_retry(candidate)
            await enqueue_candidate(candidate, priority=30)

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


async def _match_candidate(
    candidate: JobCandidate,
    ctx: ContextManager,
    resume_context: str,
    candidate_persona: str,
    store,
    cfg,
    results: list[JobCandidate],
) -> str:
    """Run one candidate through the vector gate + LLM matcher.

    Appends the (possibly rejected) candidate to ``results`` and persists it,
    mirroring the in-memory worker path. Returns:
    - "matched": LLM call succeeded (or vector-gate rejected)
    - "retry":   LLM returned 429, caller should re-queue
    - "error":   terminal failure
    """
    try:
        resume_chunks: list[dict[str, Any]] | None = None
        if cfg.vector_gate_enabled or resume_context:
            retrieved = await _resume_chunks_for(candidate, store)
            if retrieved is not None:
                chunks, similarity = retrieved
                resume_chunks = chunks
                if cfg.vector_gate_enabled and similarity is not None:
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
                        return "matched"

        jd = candidate.extra.get("raw_markdown", "")[:12000]

        digest = _digest_from_chunks(resume_chunks) if resume_chunks else ""
        resume_block = digest if digest else resume_context[:3000]

        prompt = MATCHER_PROMPT.replace("{candidate_persona}", candidate_persona)
        prompt = prompt.replace("{resume_context}", resume_block)
        prompt = prompt.replace("{job_markdown}", jd)

        result = await ctx.json_chat(
            prompt,
            schema=MATCHER_SCHEMA,
            max_tokens=cfg.match_token_budget,
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
        return "matched"

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
                return "retry"
            candidate.eligibility = EligibilityState.ERROR
            if store is not None:
                await _persist_candidate(store, candidate)
            return "error"
        candidate.eligibility = EligibilityState.ERROR
        candidate.rejection_reason = RejectionReason.UNKNOWN
        if store is not None:
            await _persist_candidate(store, candidate)
        logger.warning("LLM queue worker failed", exception=err_msg)
        return "error"


async def _resume_chunks_for(
    candidate: JobCandidate, store: Any, top_k: int = 8
) -> tuple[list[dict[str, Any]], float | None] | None:
    """Pass 1: cheap pgvector retrieval against the resume chunk store.

    Embeds the JD once and returns (chunks, avg_cosine_similarity). The
    chunks feed both the vector gate (threshold reject) and the matcher
    prompt's resume digest, so a single embed + search serves both.

    Returns ``None`` when the gate cannot run (no store, no chunk index,
    embed server down). Callers must pass ``None`` through to the LLM.
    """
    if store is None or not hasattr(store, "search_similar_chunks"):
        return None

    jd = candidate.extra.get("raw_markdown", "")[:4000]
    if not jd.strip():
        return None

    try:
        from src.agent.enrichment_agent import _get_embedding

        jd_vector = await _get_embedding(jd, store)
        if jd_vector is None:
            return None

        chunks = await store.search_similar_chunks(jd_vector, top_k=top_k)
        if not chunks:
            return None

        similarities = [1.0 - ch.get("distance", 1.0) for ch in chunks]
        similarities = [s for s in similarities if s >= 0.0]
        similarity = sum(similarities) / len(similarities) if similarities else None
        return chunks, similarity
    except Exception:
        return None


def _digest_from_chunks(chunks: list[dict[str, Any]], max_chars: int = 2000) -> str:
    """Compress retrieved resume chunks into a tight prompt block."""
    parts: list[str] = []
    total = 0
    for c in chunks:
        text = (c.get("content") or "").strip()
        if not text:
            continue
        piece = text[:900]
        parts.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    return "\n".join(parts)


async def _dequeue() -> tuple[int, int, JobCandidate] | None:
    loop = asyncio.get_running_loop()
    if getattr(_queue_not_empty, "_loop", None) is not loop:
        _queue_not_empty._loop = loop  # type: ignore[attr-defined]
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
    # Never let a placeholder company name from the LLM overwrite a real one
    # extracted from the source/URL. Placeholders have no apply target.
    llm_company = str(result.get("company", "") or "").strip()
    _placeholder = re.match(
        r"^(not\s*specified|unknown|n/?a|n\.?a\.?|tbd|company|-+)$", llm_company, re.I
    )
    if llm_company and not _placeholder:
        candidate.normalized_company = llm_company
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
    if verdict in ("STRONG_MATCH", "GOOD_MATCH") and candidate.match_percent >= 50:
        candidate.eligibility = EligibilityState.ACCEPTED
    elif verdict == "WEAK_MATCH" or (
        candidate.match_percent >= 40 and len(candidate.missing_skills) <= 3
    ):
        # Fuzzy / LARP-able: close enough that the gaps are learnable.
        candidate.eligibility = EligibilityState.NEAR_MISS
    else:
        candidate.eligibility = EligibilityState.REJECTED
        candidate.rejection_reason = RejectionReason.MATCHER_NO_MATCH

    _apply_location_eligibility(candidate)


def _apply_location_eligibility(candidate: JobCandidate) -> None:
    """Reject US onsite roles: this candidate cannot attend them without
    visa sponsorship, so recommending them wastes both sides' time.

    US remote roles pass (the visa-sponsorship question is carried on the
    card as a warning when unconfirmed). Only applies when the matcher
    actually produced a location; unknown locations are not penalized.
    """
    from src.radar.core.signals import is_us_location

    if candidate.eligibility not in (EligibilityState.ACCEPTED, EligibilityState.NEAR_MISS):
        return
    if not get_config().radar.us_only_remote:
        return
    location = candidate.normalized_location or ""
    if not location or location.lower() in ("unknown", "n/a", "not specified"):
        return
    if candidate.is_remote:
        return
    if is_us_location(location):
        candidate.eligibility = EligibilityState.REJECTED
        candidate.rejection_reason = RejectionReason.US_ONSITE


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
        if candidate.eligibility == EligibilityState.ACCEPTED:
            await _record_accepted_evidence(store, candidate)
    except Exception as e:
        logger.warning(
            "Failed to persist radar candidate",
            canonical_id=candidate.canonical_id,
            exception=str(e),
        )


async def _record_accepted_evidence(store, candidate: JobCandidate) -> None:
    """Log a posted_job hiring signal into the evidence ledger (best-effort)."""
    try:
        from src.graph.entity import make_company_id

        await store.record_evidence(
            make_company_id(candidate.normalized_company),
            claim="posted_job",
            source=candidate.source or "radar",
            company_name=candidate.normalized_company,
            evidence_type="hiring",
            weight=0.35,
            ref_url=candidate.direct_apply_url or "",
        )
    except Exception:
        pass


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
