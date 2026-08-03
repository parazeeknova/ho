"""Tests for the evidence ledger: combination, uncertainty hook, storage."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.graph.entity import (
    Confidence,
    GraphNode,
    NodeType,
    combine_evidence,
    compute_uncertainty_score,
)
from src.memory.pgvector_store import CREATE_TABLES_SQL, MemoryStore


def _row(
    weight: float,
    *,
    contradicts: bool = False,
    age_days: float = 0.0,
) -> dict:
    return {
        "weight": weight,
        "contradicts": contradicts,
        "observed_at": time.time() - age_days * 86400,
    }


class TestCombineEvidence:
    def test_empty_evidence_is_neutral(self) -> None:
        c = combine_evidence([])
        assert c.score == pytest.approx(0.5, abs=0.01)
        assert c.source_count == 0

    def test_single_signal(self) -> None:
        c = combine_evidence([_row(0.4)])
        # 1 - (1 - 0.4) = 0.4
        assert c.score == pytest.approx(0.4, abs=0.01)
        assert c.source_count == 1

    def test_multiple_signals_accumulate_noisy_or(self) -> None:
        c = combine_evidence([_row(0.4), _row(0.35), _row(0.25)])
        expected = 1 - (1 - 0.4) * (1 - 0.35) * (1 - 0.25)
        assert c.score == pytest.approx(expected, abs=0.01)

    def test_contradiction_pulls_down(self) -> None:
        c = combine_evidence([_row(0.4), _row(0.5, contradicts=True)])
        expected = 0.4 * (1 - 0.5)
        assert c.score == pytest.approx(expected, abs=0.01)

    def test_strong_evidence_clamped_at_095(self) -> None:
        c = combine_evidence([_row(0.95), _row(0.95), _row(0.95), _row(0.95)])
        assert c.score <= 0.95

    def test_stale_evidence_decays(self) -> None:
        fresh = combine_evidence([_row(0.5)])
        stale = combine_evidence([_row(0.5, age_days=60)])
        assert stale.score < fresh.score

    def test_verification_method_evidence(self) -> None:
        c = combine_evidence([_row(0.3)])
        assert c.verification_method == "evidence"


class TestEvidenceUncertaintyHook:
    def test_evidence_score_lowers_uncertainty(self) -> None:
        node = GraphNode(id="n1", node_type=NodeType.COMPANY, data={"name": "Acme"})
        node.confidence = Confidence(score=0.1)  # graph says low confidence
        adjacency: dict = {"edges_out": set(), "edges_in": set()}
        from src.graph.entity import CAPABILITY_GRAPH

        base = compute_uncertainty_score(node, adjacency, CAPABILITY_GRAPH)
        with_evidence = compute_uncertainty_score(
            node, adjacency, CAPABILITY_GRAPH, evidence_score=0.8
        )
        assert with_evidence.uncertainty < base.uncertainty


async def _mock_store() -> MemoryStore:
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 0 1")
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    with (
        patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)),
        patch("pgvector.asyncpg.register_vector", new=AsyncMock()),
    ):
        store = await MemoryStore.create()
        store._pool = mock_pool
    return store


class TestEvidenceStore:
    def test_ddl_present(self) -> None:
        assert "CREATE TABLE IF NOT EXISTS evidence" in CREATE_TABLES_SQL
        assert "PRIMARY KEY (company_id, claim, source)" in CREATE_TABLES_SQL

    async def test_record_evidence_executes_upsert(self) -> None:
        store = await _mock_store()
        executed: list[tuple] = []

        async def _execute(sql: str, *args: object) -> str:
            executed.append((sql, args))
            return "INSERT 0 1"

        store._pool.acquire.return_value.__aenter__ = AsyncMock(
            return_value=AsyncMock(execute=_execute)
        )
        await store.record_evidence(
            "company-1", claim="ats_openings", source="ats_interceptor", weight=0.4
        )
        assert executed
        assert "evidence" in executed[0][0]
        assert "ON CONFLICT (company_id, claim, source)" in executed[0][0]
        assert executed[0][1][0] == "company-1"

    async def test_get_evidence_returns_rows(self) -> None:
        store = await _mock_store()
        row = {
            "company_id": "c1",
            "company_name": "Acme",
            "claim": "posted_job",
            "evidence_type": "hiring",
            "source": "radar",
            "weight": 0.35,
            "contradicts": False,
            "ref_url": "",
            "observed_at": time.time(),
        }
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[row])
        store._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        rows = await store.get_evidence("c1")
        assert rows == [row]

    async def test_evidence_summary_builds_confidence(self) -> None:
        store = await _mock_store()
        now = time.time()
        rows = [
            {"weight": 0.4, "contradicts": False, "observed_at": now},
            {"weight": 0.35, "contradicts": False, "observed_at": now},
        ]
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=rows)
        store._pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        summary = await store.evidence_summary("c1")
        assert summary["support"] == 2
        assert summary["contradict"] == 0
        assert 0.0 < summary["confidence"] < 1.0
