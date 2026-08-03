"""Tests for TTLSet (OrderedDict-backed) and EventBus dedup."""

from __future__ import annotations

import asyncio

import pytest

from src.graph.entity import GraphEvent, NodeType
from src.graph.event_bus import EventBus, TTLSet


def test_add_and_contains() -> None:
    s = TTLSet(maxsize=10, ttl=60.0)
    s.add("a")
    assert "a" in s
    assert s.misses == 0
    assert s.hits == 1


def test_expiry_via_monotonic_time(monkeypatch: pytest.MonkeyPatch) -> None:
    s = TTLSet(maxsize=10, ttl=60.0)
    now = 1000.0
    monkeypatch.setattr(s, "_clock", lambda: now)
    s.add("a")
    now += 61.0
    assert "a" not in s
    assert "a" not in s  # still absent after first check
    assert s.misses == 2


def test_refresh_on_readd() -> None:
    s = TTLSet(maxsize=10, ttl=60.0)
    s.add("a")
    s.add("a")
    assert len(s) == 1
    assert "a" in s


def test_capacity_evicts_oldest_first() -> None:
    s = TTLSet(maxsize=3, ttl=60.0)
    s.add("a")
    s.add("b")
    s.add("c")
    s.add("d")  # evicts "a"
    assert "a" not in s
    assert "b" in s and "c" in s and "d" in s
    s.add("e")  # evicts "b"
    assert "b" not in s
    assert len(s) == 3


def test_readd_refreshes_eviction_order() -> None:
    s = TTLSet(maxsize=3, ttl=60.0)
    s.add("a")
    s.add("b")
    s.add("a")  # "a" is now the most-recently-touched
    s.add("c")
    s.add("d")  # evicts "b", not "a"
    assert "b" not in s
    assert "a" in s


def test_evict_stale_only_drops_expired_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    s = TTLSet(maxsize=10, ttl=60.0)
    now = 1000.0
    monkeypatch.setattr(s, "_clock", lambda: now)
    s.add("old1")
    s.add("old2")
    now += 61.0
    s.add("fresh")  # triggers _evict_stale; drops old1, old2 only
    assert len(s) == 1
    assert "fresh" in s


def test_len_counts_non_expired() -> None:
    s = TTLSet(maxsize=10, ttl=60.0)
    s.add("a")
    s.add("b")
    assert len(s) == 2


def test_stats() -> None:
    s = TTLSet(maxsize=10, ttl=60.0)
    s.add("a")
    assert "a" in s
    stats = s.stats
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 0


async def test_eventbus_dedup_single_fire() -> None:
    bus = EventBus()
    try:
        fired: list[str] = []

        async def handler(event: GraphEvent) -> None:
            fired.append(event.node_id)

        bus.subscribe("test.evt", handler)
        evt = bus.new_event("test.evt", "node-1", NodeType.COMPANY)
        await bus.fire(evt)
        await bus.fire(evt)  # duplicate event id -> dropped
        await asyncio.sleep(0.1)
        assert fired == ["node-1"]
        assert bus.duplicate_count == 1
        assert bus.fired_count == 1
    finally:
        await bus.shutdown(timeout=1.0)
