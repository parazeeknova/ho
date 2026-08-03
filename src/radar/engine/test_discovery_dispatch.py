"""Tests for discovered-source founder dispatch (YC + discovery adapter flow)."""

from __future__ import annotations

import pytest

from src.radar.engine import orchestrator as orch
from src.radar.sources.sources import _LAST_SNAPSHOT_URLS, _SOURCE_CHECKPOINTS


@pytest.fixture(autouse=True)
def _clean_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    _SOURCE_CHECKPOINTS.clear()
    _LAST_SNAPSHOT_URLS.clear()
    monkeypatch.delenv("DISCOVERY_FOUNDER_MINE_LIMIT", raising=False)
    yield
    _SOURCE_CHECKPOINTS.clear()
    _LAST_SNAPSHOT_URLS.clear()


class _FakeStore:
    pass


class _FakeBus:
    def __init__(self) -> None:
        self.fired: list[dict] = []

    def new_event(self, event_type: str, node_id: str, node_type, payload: dict | None = None):
        return {
            "event_type": event_type,
            "node_id": node_id,
            "payload": payload or {},
        }

    async def fire(self, event: dict) -> None:
        self.fired.append(event)


class _FakeGraph:
    def __init__(self) -> None:
        self.upserted: list[str] = []

    async def get_node(self, node_id: str):
        return None  # always missing -> orchestrator creates the node

    async def upsert_node(self, node):
        self.upserted.append(node.id)
        return node, None


async def test_persist_returns_only_new_sources() -> None:
    store = _FakeStore()
    companies = [
        {
            "name": "Acme YC",
            "website": "https://acme.example",
            "ats_url": "https://boards.greenhouse.io/acme",
        },
        {"name": "Beta YC", "website": "https://beta.example"},
    ]
    added1 = await orch._persist_discovered_sources(store, companies)
    assert len(added1) == 2
    assert {a["name"] for a in added1} == {"Acme YC", "Beta YC"}

    # Re-persisting the same list adds nothing (idempotent registration)
    added2 = await orch._persist_discovered_sources(store, companies)
    assert added2 == []


async def test_dispatch_fires_bounded_events(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = _FakeBus()
    graph = _FakeGraph()
    added = [{"name": f"YC Company {i}", "url": f"https://c{i}.example"} for i in range(25)]
    monkeypatch.setenv("DISCOVERY_FOUNDER_MINE_LIMIT", "5")
    count = await orch._dispatch_discovered_founders(added, bus, graph)
    assert count == 5  # bounded by the env cap
    assert len(bus.fired) == 5
    assert bus.fired[0]["event_type"] == "company_discovered"
    assert bus.fired[0]["payload"]["name"] == "YC Company 0"
    assert len(graph.upserted) == 5  # missing nodes were created


async def test_dispatch_empty_noop() -> None:
    bus = _FakeBus()
    graph = _FakeGraph()
    count = await orch._dispatch_discovered_founders([], bus, graph)
    assert count == 0
    assert bus.fired == []
