"""Tests for LLM work queue: dedup, ordering, 429 retry, status."""

from __future__ import annotations

import time

import pytest

from src.radar.models import JobCandidate
from src.radar.queue import (
    _queue_state,
    enqueue_candidate,
    get_queue_status,
    mark_retry,
)


@pytest.fixture(autouse=True)
def reset_queue_state() -> None:
    from src.radar.governor import _state as _gs
    from src.radar.queue import _ACTIVE_IDS, _CANDIDATE_VERSIONS

    _queue_state.pending.clear()
    _queue_state.total_enqueued = 0
    _queue_state.total_completed = 0
    _queue_state.total_failed = 0
    _queue_state.total_429s = 0
    _ACTIVE_IDS.clear()
    _CANDIDATE_VERSIONS.clear()
    # Reset governor state
    _gs.in_flight = 0
    _gs.requests_this_minute = 0
    _gs.tokens_this_minute = 0
    _gs.window_start = time.monotonic()
    _gs.cooldown_until = 0.0
    _gs.total_requests = 0
    _gs.total_429s = 0
    yield
    _ACTIVE_IDS.clear()
    _CANDIDATE_VERSIONS.clear()


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

        entries = list(_queue_state.pending)
        priorities = [e[0] for e in entries]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_mark_retry_allows_reenqueue(self) -> None:
        candidate = JobCandidate(
            canonical_id="test:retry:remote",
            source="greenhouse",
            direct_apply_url="https://example.com/retry",
            normalized_company="Test",
            normalized_role="Retry",
            normalized_location="Remote",
        )
        assert await enqueue_candidate(candidate)
        mark_retry(candidate)
        assert await enqueue_candidate(candidate)


class TestQueueStatus:
    def test_status_initial(self) -> None:
        status = get_queue_status()
        assert status["pending"] >= 0
        assert "cooldown_active" in status
        assert "total_429s" in status
        assert "total_completed" in status
        assert "total_enqueued" in status
        assert "in_flight" in status
