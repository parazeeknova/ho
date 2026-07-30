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
from src.radar.models import EligibilityState, JobCandidate

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

MATCHER_PROMPT = """\
You are a job-resume matching engine. Evaluate this candidate against the job description.

Candidate profile:
{candidate_persona}

Resume skills context:
{resume_context}

Job listing:
{job_markdown}

Return JSON with exactly these fields:
- company: company name (string)
- role: role title (string)
- match_percent: 0-100 integer
- shortlist_probability: 0-100 integer
- matching_skills: array of strings (skills the candidate HAS that match the JD)
- missing_skills: array of strings (skills the JD REQUIRES that the candidate LACKS)
- verdict: one of STRONG_MATCH, GOOD_MATCH, WEAK_MATCH, NO_MATCH
- jd_summary: 1-2 sentence summary of the role
- company_description: 1-2 sentence company overview
- role_summary: 1-2 sentence role overview
- salary: string or null
- posted_date: string or null
- location: string
- is_remote: boolean

If the text is a company homepage, job directory, error page, or lists multiple
different jobs, set match_percent=0 and verdict=NO_MATCH.
"""


async def enqueue_candidate(candidate: JobCandidate, priority: int = 50) -> bool:
    async with _queue_lock:
        if candidate.canonical_id in _seen_ids:
            return False
        _seen_ids.add(candidate.canonical_id)
        _queue_state.pending.append((priority, candidate))
        _queue_state.total_enqueued += 1
        _queue_state.pending = deque(sorted(_queue_state.pending, key=lambda x: x[0], reverse=True))
        _queue_not_empty.set()
    return True


async def process_queue(
    ctx: ContextManager,
    resume_context: str,
    candidate_persona: str,
    store,  # MemoryStore for persistence
    max_candidates: int = 50,
) -> list[JobCandidate]:
    cfg = get_config().llm_queue
    results: list[JobCandidate] = []
    sem = asyncio.Semaphore(cfg.max_in_flight)

    async def _worker(candidate: JobCandidate) -> None:
        async with sem:
            await _acquire_budget(cfg)
            try:
                prompt = MATCHER_PROMPT.replace("{candidate_persona}", candidate_persona)
                prompt = prompt.replace("{resume_context}", resume_context[:3000])
                jd = candidate.extra.get("raw_markdown", "")[:8000]
                prompt = prompt.replace("{job_markdown}", jd)

                result = await ctx.json_chat(prompt, TokenBudget=cfg.match_token_budget)
                if isinstance(result, dict):
                    _apply_llm_result(candidate, result)
                    async with _queue_lock:
                        _queue_state.total_completed += 1
                    results.append(candidate)
            except Exception as e:
                async with _queue_lock:
                    _queue_state.total_failed += 1
                if _is_429(str(e)):
                    await _handle_429(cfg)
                    await enqueue_candidate(candidate, priority=30)
                logger.warning("LLM queue worker failed", exception=str(e))

    tasks = []
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
        async with _queue_lock:
            now = time.monotonic()

            if _queue_state.cooldown_until > 0 and now < _queue_state.cooldown_until:
                wait = _queue_state.cooldown_until - now
                await asyncio.sleep(wait)
                continue

            if now - _queue_state.window_start >= 60.0:
                _queue_state.window_start = now
                _queue_state.requests_this_minute = 0
                _queue_state.tokens_this_minute = 0

            if _queue_state.requests_this_minute >= cfg.requests_per_minute:
                await asyncio.sleep(60.0 - (now - _queue_state.window_start))
                continue

            if (
                _queue_state.tokens_this_minute + cfg.match_token_budget
                > cfg.estimated_tokens_per_minute
            ):
                await asyncio.sleep(60.0 - (now - _queue_state.window_start))
                continue

            _queue_state.requests_this_minute += 1
            _queue_state.tokens_this_minute += cfg.match_token_budget
            return


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
        _queue_state.in_flight += 1
        return entry


async def _handle_429(cfg: LlmQueueConfig) -> None:
    async with _queue_lock:
        _queue_state.total_429s += 1
        cooldown = cfg.cooldown_seconds + random.uniform(0, cfg.jitter_seconds)
        _queue_state.cooldown_until = time.monotonic() + cooldown
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

    verdict = candidate.verdict
    if verdict == "NO_MATCH" or candidate.match_percent < 30:
        candidate.eligibility = EligibilityState.REJECTED
    elif verdict in ("STRONG_MATCH", "GOOD_MATCH"):
        candidate.eligibility = EligibilityState.ACCEPTED
    elif verdict == "WEAK_MATCH":
        candidate.eligibility = EligibilityState.NEAR_MISS


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


_seen_ids: set[str] = set()
