"""Tests for path-based hiring signals in predict_hiring_likelihood."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.graph.graph_store import GraphStore


def _make_store(queries: list[list[dict]] | None = None) -> tuple[GraphStore, AsyncMock]:
    store = GraphStore.__new__(GraphStore)
    store._driver = None
    mock_run = AsyncMock(side_effect=queries or [[]])
    store._run = mock_run  # type: ignore[method-assign]
    store.get_node = AsyncMock(  # type: ignore[method-assign]
        return_value=type(
            "Node",
            (),
            {
                "data": {},
                "name": "Acme",
                "confidence": type("C", (), {"score": 0.5})(),
            },
        )()
    )
    store.get_edges_from = AsyncMock(return_value=[])  # type: ignore[method-assign]
    store.find_structurally_similar = AsyncMock(return_value=[])  # type: ignore[method-assign]
    return store, mock_run


@pytest.mark.asyncio
async def test_shared_investor_signal_added() -> None:
    store, mock_run = _make_store([[{"peer_id": "p1"}, {"peer_id": "p2"}]])
    result = await store.predict_hiring_likelihood("company-1")
    signals = {s["type"]: s for s in result["signals"]}
    assert "shared_investor" in signals
    assert signals["shared_investor"]["weight"] == pytest.approx(0.12)
    assert signals["shared_investor"]["peer_count"] == 2


@pytest.mark.asyncio
async def test_same_ats_signal_added() -> None:
    store, _ = _make_store([[], [{"peer_id": "p1"}]])
    result = await store.predict_hiring_likelihood("company-1")
    signals = {s["type"]: s for s in result["signals"]}
    assert "same_ats" in signals
    assert signals["same_ats"]["peer_count"] == 1


@pytest.mark.asyncio
async def test_same_tech_signal_added() -> None:
    store, _ = _make_store([[], [], [{"tech": "python"}, {"tech": "postgres"}]])
    result = await store.predict_hiring_likelihood("company-1")
    signals = {s["type"]: s for s in result["signals"]}
    assert "same_tech" in signals
    assert signals["same_tech"]["tech_count"] == 2


@pytest.mark.asyncio
async def test_empty_graph_no_crash() -> None:
    store, mock_run = _make_store()
    result = await store.predict_hiring_likelihood("company-1")
    assert result["node_id"] == "company-1"
    assert result["score"] == 0.0
    assert result["signals"] == []
    assert mock_run.await_count == 3  # all three path queries ran, empty


@pytest.mark.asyncio
async def test_query_failure_is_defensive() -> None:
    store, _ = _make_store()

    async def _boom(query: str, params: dict | None = None) -> list[dict]:
        raise RuntimeError("neo4j down")

    store._run = _boom  # type: ignore[method-assign]
    result = await store.predict_hiring_likelihood("company-1")
    assert result["signals"] == []  # no crash, just no path signals
