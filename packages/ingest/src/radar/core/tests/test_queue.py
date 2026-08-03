"""Tests for LLM work queue: dedup, ordering, 429 retry, status."""

from __future__ import annotations

import time

import pytest
from src.radar.core.models import EligibilityState, JobCandidate, RejectionReason
from src.radar.core.queue import (
    _queue_state,
    enqueue_candidate,
    get_queue_status,
    mark_retry,
    process_queue,
)


@pytest.fixture(autouse=True)
def reset_queue_state() -> None:
    from src.radar.core.governor import _state as _gs
    from src.radar.core.queue import _ACTIVE_IDS, _CANDIDATE_VERSIONS

    _queue_state.pending.clear()
    _queue_state.total_enqueued = 0
    _queue_state.total_completed = 0
    _queue_state.total_failed = 0
    _queue_state.total_429s = 0
    _queue_state.total_vector_rejects = 0
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

        import heapq

        popped = [heapq.heappop(_queue_state.pending)[0] for _ in range(5)]
        assert popped == [-40, -30, -20, -10, 0]

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


class _FakeStore:
    def __init__(self, distances: list[float] | None = None) -> None:
        self._distances = distances or [0.85, 0.9, 0.95]
        self.upserts: list[dict] = []

    async def search_similar_chunks(self, query_emb: list[float], top_k: int = 5) -> list[dict]:
        return [
            {"section": "skills", "content": "x", "distance": d} for d in self._distances[:top_k]
        ]

    async def upsert_radar_candidate(self, data: dict) -> None:
        self.upserts.append(data)


class _FakeCtx:
    def __init__(self) -> None:
        self.calls = 0

    async def json_chat(self, prompt: str, **_: object) -> dict:
        self.calls += 1
        return {
            "company": "Acme",
            "role": "Engineer",
            "match_percent": 85,
            "shortlist_probability": 70,
            "verdict": "STRONG_MATCH",
            "matching_skills": ["python"],
            "missing_skills": [],
            "location": "Remote",
        }


@pytest.fixture(autouse=True)
def _patch_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.agent.enrichment_agent as ea

    async def fake_embed(text: str, store=None) -> list[float] | None:  # noqa: ANN001
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(ea, "_get_embedding", fake_embed)


def _candidate(canonical_id: str) -> JobCandidate:
    return JobCandidate(
        canonical_id=canonical_id,
        source="greenhouse",
        direct_apply_url="https://example.com/job",
        normalized_company="Acme",
        normalized_role="Engineer",
        normalized_location="Remote",
        extra={"raw_markdown": "Senior backend engineer, Python, distributed systems"},
    )


class TestVectorGate:
    @pytest.mark.asyncio
    async def test_rejects_below_threshold_without_llm(self) -> None:
        store = _FakeStore(distances=[0.85, 0.9, 0.95])
        ctx = _FakeCtx()
        await enqueue_candidate(_candidate("gate:reject:remote"))
        results = await process_queue(ctx, "resume", "persona", store, max_candidates=10)

        assert ctx.calls == 0
        c = results[0]
        assert c.eligibility == EligibilityState.REJECTED
        assert c.rejection_reason == RejectionReason.VECTOR_GATE
        assert c.extra["vector_similarity"] < 0.35
        assert get_queue_status()["total_vector_rejects"] == 1

    @pytest.mark.asyncio
    async def test_passes_high_similarity_to_llm(self) -> None:
        store = _FakeStore(distances=[0.1, 0.15, 0.2])
        ctx = _FakeCtx()
        await enqueue_candidate(_candidate("gate:pass:remote"))
        results = await process_queue(ctx, "resume", "persona", store, max_candidates=10)

        assert ctx.calls == 1
        assert results[0].eligibility == EligibilityState.ACCEPTED
        assert results[0].extra["vector_similarity"] > 0.35
        assert get_queue_status()["total_vector_rejects"] == 0

    @pytest.mark.asyncio
    async def test_disabled_gate_skips_vector_search(self) -> None:
        from src.configuration import get_config

        cfg = get_config().llm_queue
        enabled, threshold = cfg.vector_gate_enabled, cfg.vector_gate_threshold
        cfg.vector_gate_enabled = False
        cfg.vector_gate_threshold = 0.99
        try:
            store = _FakeStore(distances=[0.85, 0.9, 0.95])
            ctx = _FakeCtx()
            await enqueue_candidate(_candidate("gate:disabled:remote"))
            await process_queue(ctx, "resume", "persona", store, max_candidates=10)
            assert ctx.calls == 1
        finally:
            cfg.vector_gate_enabled = enabled
            cfg.vector_gate_threshold = threshold

    @pytest.mark.asyncio
    async def test_passes_through_when_embedding_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.agent.enrichment_agent as ea

        async def failing_embed(text: str, store=None) -> list[float] | None:  # noqa: ANN001
            return None

        monkeypatch.setattr(ea, "_get_embedding", failing_embed)

        store = _FakeStore(distances=[0.85, 0.9, 0.95])
        ctx = _FakeCtx()
        await enqueue_candidate(_candidate("gate:embedfail:remote"))
        results = await process_queue(ctx, "resume", "persona", store, max_candidates=10)

        assert ctx.calls == 1
        assert results[0].eligibility == EligibilityState.ACCEPTED

    @pytest.mark.asyncio
    async def test_passes_through_when_store_is_none(self) -> None:
        ctx = _FakeCtx()
        await enqueue_candidate(_candidate("gate:nostore:remote"))
        results = await process_queue(ctx, "resume", "persona", None, max_candidates=10)

        assert ctx.calls == 1
        assert results[0].eligibility == EligibilityState.ACCEPTED


class TestCandidateFromPayload:
    def test_plain_dict_payload(self) -> None:
        from src.radar.core.queue import _candidate_from_payload

        c = _candidate_from_payload(
            {"canonical_id": "abc", "source": "lever", "direct_apply_url": "https://x", "extra": {}}
        )
        assert c.canonical_id == "abc"

    def test_clean_json_string_payload(self) -> None:
        import json

        from src.radar.core.queue import _candidate_from_payload

        c = _candidate_from_payload(json.dumps({"canonical_id": "abc"}))
        assert c.canonical_id == "abc"

    def test_legacy_double_encoded_payload(self) -> None:
        import json

        from src.radar.core.queue import _candidate_from_payload

        # Rows written while the jsonb codec double-encoded pre-serialized
        # strings: the stored value is a JSON string containing JSON text.
        c = _candidate_from_payload(json.dumps(json.dumps({"canonical_id": "abc"})))
        assert c.canonical_id == "abc"

    def test_garbage_payload_survives(self) -> None:
        from src.radar.core.queue import _candidate_from_payload

        c = _candidate_from_payload("not json at all")
        assert c.canonical_id == ""
