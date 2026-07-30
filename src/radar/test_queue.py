"""Tests for LLM work queue: budget enforcement, 429 handling, delayed resume."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.configuration import LlmQueueConfig
from src.radar.models import JobCandidate
from src.radar.queue import (
    _acquire_budget,
    _handle_429,
    _queue_state,
    enqueue_candidate,
    get_queue_status,
)


@pytest.fixture(autouse=True)
def reset_queue_state() -> None:
    from src.radar.queue import _seen_ids

    _queue_state.pending.clear()
    _queue_state.in_flight = 0
    _queue_state.requests_this_minute = 0
    _queue_state.tokens_this_minute = 0
    _queue_state.window_start = time.monotonic()
    _queue_state.cooldown_until = 0.0
    _queue_state.total_enqueued = 0
    _queue_state.total_completed = 0
    _queue_state.total_failed = 0
    _queue_state.total_429s = 0
    _seen_ids.clear()
    yield
    _seen_ids.clear()


class TestEnqueueCandidate:
    @pytest.mark.asyncio
    async def test_enqueue_single(self) -> None:
        candidate = JobCandidate(
            canonical_id="test:role:remote",
            source="greenhouse",
            direct_apply_url="https://example.com/job",
            normalized_company="Test",
            normalized_role="Role",
            normalized_location="Remote",
        )
        result = await enqueue_candidate(candidate, priority=60)
        assert result is True
        assert get_queue_status()["pending"] == 1

    @pytest.mark.asyncio
    async def test_enqueue_duplicate(self) -> None:
        candidate = JobCandidate(
            canonical_id="test:role:remote",
            source="greenhouse",
            direct_apply_url="https://example.com/job",
            normalized_company="Test",
            normalized_role="Role",
            normalized_location="Remote",
        )
        assert await enqueue_candidate(candidate)
        assert not await enqueue_candidate(candidate)
        assert get_queue_status()["pending"] == 1

    @pytest.mark.asyncio
    async def test_enqueue_priority_ordering(self) -> None:
        for i in range(5):
            candidate = JobCandidate(
                canonical_id=f"test:role{i}:remote",
                source="greenhouse",
                direct_apply_url=f"https://example.com/job/{i}",
                normalized_company="Test",
                normalized_role=f"Role{i}",
                normalized_location="Remote",
            )
            await enqueue_candidate(candidate, priority=i * 10)

        from src.radar.queue import _queue_lock

        async with _queue_lock:
            entries = list(_queue_state.pending)
            priorities = [e[0] for e in entries]
            assert priorities == sorted(priorities, reverse=True)


class TestBudgetAcquisition:
    @pytest.mark.asyncio
    async def test_acquire_budget_increments_counters(self) -> None:
        cfg = LlmQueueConfig(
            requests_per_minute=100,
            estimated_tokens_per_minute=100000,
            max_in_flight=5,
            match_token_budget=600,
        )
        initial_requests = _queue_state.requests_this_minute
        initial_tokens = _queue_state.tokens_this_minute

        await _acquire_budget(cfg)

        assert _queue_state.requests_this_minute == initial_requests + 1
        assert _queue_state.tokens_this_minute == initial_tokens + cfg.match_token_budget

    @pytest.mark.asyncio
    async def test_acquire_budget_respects_rpm_limit(self) -> None:
        cfg = LlmQueueConfig(
            requests_per_minute=1,
            estimated_tokens_per_minute=100000,
            max_in_flight=5,
            match_token_budget=600,
        )
        _queue_state.requests_this_minute = 1

        async def _acquire_with_timeout() -> None:
            await asyncio.wait_for(_acquire_budget(cfg), timeout=2.0)

        with pytest.raises(TimeoutError):
            await _acquire_with_timeout()


class Test429Handling:
    @pytest.mark.asyncio
    async def test_429_increments_counter(self) -> None:
        cfg = LlmQueueConfig(
            requests_per_minute=20,
            estimated_tokens_per_minute=30000,
            max_in_flight=2,
            cooldown_seconds=1.0,
            jitter_seconds=0.0,
        )
        initial_429s = _queue_state.total_429s
        await _handle_429(cfg)
        assert _queue_state.total_429s == initial_429s + 1
        assert _queue_state.cooldown_until > 0

    @pytest.mark.asyncio
    async def test_429_sets_cooldown(self) -> None:
        cfg = LlmQueueConfig(
            requests_per_minute=20,
            estimated_tokens_per_minute=30000,
            max_in_flight=2,
            cooldown_seconds=30.0,
            jitter_seconds=0.0,
        )
        await _handle_429(cfg)
        assert _queue_state.cooldown_until > time.monotonic()


class TestQueueStatus:
    def test_status_initial(self) -> None:
        status = get_queue_status()
        assert status["pending"] >= 0
        assert "cooldown_active" in status
        assert "total_429s" in status
        assert "total_completed" in status
