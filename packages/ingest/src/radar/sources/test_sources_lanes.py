"""Tests for adaptive poll lanes, EV source selection, and connector circuit."""

from __future__ import annotations

import time
from collections.abc import Generator

import pytest
from src.configuration import get_config
from src.connectors.base import (
    CONNECTOR_ERROR_RATE_CIRCUIT_OPEN,
    BaseConnector,
    ConnectorUnavailableError,
    DiscoveredEntity,
)
from src.radar.sources import sources as src_sources

LANES = ("high", "medium", "low")


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None]:
    src_sources._SOURCE_CHECKPOINTS.clear()
    src_sources._LAST_SNAPSHOT_URLS.clear()
    yield
    src_sources._SOURCE_CHECKPOINTS.clear()
    src_sources._LAST_SNAPSHOT_URLS.clear()


def _cp(source_id: str) -> None:
    src_sources.register_source(source_id, "ats_board", initial_quality=0.6)


def _poll_unchanged(source_id: str, n: int) -> None:
    for _ in range(n):
        src_sources.diff_snapshots(source_id, ["https://a.com/1"])


class TestLaneTransitions:
    def test_demotes_after_5_unchanged_polls(self) -> None:
        _cp("s1")
        _poll_unchanged("s1", 6)
        assert src_sources.get_checkpoint("s1").poll_lane == "medium"

    def test_demotes_again_after_15_unchanged_polls(self) -> None:
        _cp("s2")
        _poll_unchanged("s2", 16)
        assert src_sources.get_checkpoint("s2").poll_lane == "low"

    def test_change_promotes_and_marks_last_change(self) -> None:
        _cp("s3")
        _poll_unchanged("s3", 21)  # deep into low
        assert src_sources.get_checkpoint("s3").poll_lane == "low"
        src_sources.diff_snapshots("s3", ["https://a.com/1", "https://a.com/2"])  # change
        cp = src_sources.get_checkpoint("s3")
        assert cp.poll_lane == "high"
        assert cp.last_change_at > 0

    def test_success_with_jobs_promotes(self) -> None:
        _cp("s4")
        _poll_unchanged("s4", 21)
        src_sources.record_success("s4", job_count=3, direct_url_count=2)
        assert src_sources.get_checkpoint("s4").poll_lane == "high"

    def test_success_without_jobs_keeps_lane(self) -> None:
        _cp("s5")
        _poll_unchanged("s5", 6)
        src_sources.record_success("s5", job_count=0, direct_url_count=0)
        assert src_sources.get_checkpoint("s5").poll_lane == "medium"

    def test_three_failures_demote_one_lane(self) -> None:
        _cp("s6")
        for _ in range(3):
            src_sources.record_failure("s6")
        assert src_sources.get_checkpoint("s6").poll_lane == "medium"

    def test_yield_ewma_updates(self) -> None:
        _cp("s7")
        for _ in range(8):
            src_sources.record_success("s7", job_count=10, direct_url_count=10)
        cp = src_sources.get_checkpoint("s7")
        assert 9.0 < cp.yield_per_poll <= 10.0


class TestShouldPollLaneGate:
    def test_high_lane_polls_every_sweep(self) -> None:
        _cp("h1")
        cp = src_sources.get_checkpoint("h1")
        cp.last_polled = time.time()  # polled a moment ago
        assert src_sources.should_poll("h1") is True

    def test_medium_lane_gated_by_interval(self) -> None:
        sweep = get_config().pipeline.sweep_interval
        _cp("m1")
        cp = src_sources.get_checkpoint("m1")
        cp.poll_lane = "medium"
        cp.last_polled = time.time()
        assert src_sources.should_poll("m1") is False
        cp.last_polled = time.time() - 3 * sweep - 1
        assert src_sources.should_poll("m1") is True

    def test_low_lane_gated_harder(self) -> None:
        sweep = get_config().pipeline.sweep_interval
        _cp("l1")
        cp = src_sources.get_checkpoint("l1")
        cp.poll_lane = "low"
        cp.last_polled = time.time()
        assert src_sources.should_poll("l1") is False
        cp.last_polled = time.time() - 10 * sweep - 1
        assert src_sources.should_poll("l1") is True

    def test_never_polled_polls_immediately(self) -> None:
        _cp("n1")
        cp = src_sources.get_checkpoint("n1")
        cp.poll_lane = "low"
        cp.last_polled = 0.0
        assert src_sources.should_poll("n1") is True


class TestSelectSourcesForSweep:
    def test_orders_by_expected_value(self) -> None:
        for i, (lane, yield_, last) in enumerate(
            [("high", 5.0, time.time() - 1), ("high", 1.0, time.time() - 1)]
        ):
            sid = f"ev{i}"
            _cp(sid)
            cp = src_sources.get_checkpoint(sid)
            cp.poll_lane = lane
            cp.yield_per_poll = yield_
            cp.last_polled = last
        ordered = src_sources.select_sources_for_sweep([{"id": "ev1"}, {"id": "ev0"}])
        assert ordered[0]["id"] == "ev0"  # higher yield first

    def test_low_lane_floor_never_starves(self) -> None:
        for i in range(10):
            sid = f"src{i}"
            _cp(sid)
            cp = src_sources.get_checkpoint(sid)
            cp.poll_lane = "high" if i >= 3 else "low"
            cp.yield_per_poll = 0.0
            cp.last_polled = time.time() - 1
        sources = [{"id": f"src{i}"} for i in range(10)]
        selected = src_sources.select_sources_for_sweep(sources)
        lows = [s["id"] for s in selected if s["id"] in {f"src{i}" for i in range(3)}]
        assert len(lows) == 3  # all lows present despite zero yield


class _ConcreteConnector(BaseConnector):
    """Concrete connector for circuit-breaker tests."""

    source_name = "stub"

    async def discover(self) -> list[DiscoveredEntity]:
        return []

    async def enrich(self, entity: DiscoveredEntity) -> DiscoveredEntity:
        return entity


class TestConnectorCircuitBreaker:
    @pytest.mark.asyncio
    async def test_open_circuit_raises_without_fetching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connector = _ConcreteConnector()
        # Force a high error rate by faking class counters
        monkeypatch.setattr(BaseConnector, "_health_counter", 100)
        monkeypatch.setattr(BaseConnector, "_health_failures", 40)
        assert connector._error_rate() >= CONNECTOR_ERROR_RATE_CIRCUIT_OPEN

        async def _never_called() -> None:
            raise AssertionError("network fetch should not happen with an open circuit")

        monkeypatch.setattr(
            connector, "_rate_limiter", type("RL", (), {"acquire": _never_called})()
        )
        with pytest.raises(ConnectorUnavailableError):
            await connector._fetch("https://example.com")

    def test_health_registry_populated_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.connectors.base import CONNECTOR_HEALTH

        connector = _ConcreteConnector()
        monkeypatch.setattr(BaseConnector, "_health_counter", 5)
        monkeypatch.setattr(BaseConnector, "_health_failures", 2)
        connector._update_health_registry()
        assert CONNECTOR_HEALTH["stub"].error_rate == 0.4
        assert CONNECTOR_HEALTH["stub"].status == "degraded"
