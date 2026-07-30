"""Tests for shared LLM governor and domain validation."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.radar.discovery import is_aggregator_domain
from src.radar.governor import (
    _state as _gs,
)
from src.radar.governor import (
    acquire_budget,
    get_governor_status,
    release_budget,
)
from src.radar.governor import (
    is_aggregator_domain as _gov_is_agg,
)


@pytest.fixture(autouse=True)
def reset_governor() -> None:
    _gs.in_flight = 0
    _gs.requests_this_minute = 0
    _gs.tokens_this_minute = 0
    _gs.window_start = time.monotonic()
    _gs.cooldown_until = 0.0
    _gs.total_requests = 0
    _gs.total_429s = 0
    _gs.rpm_limit = 70
    _gs.tpm_limit = 50000
    _gs.max_in_flight = 2
    _gs.reserved_requests_this_minute = 0
    yield


class TestGovernor:
    @pytest.mark.asyncio
    async def test_acquire_increments_counters(self) -> None:
        initial_r = _gs.requests_this_minute
        initial_t = _gs.tokens_this_minute
        await acquire_budget(600)
        assert _gs.requests_this_minute == initial_r + 1
        assert _gs.tokens_this_minute == initial_t + 600
        assert _gs.in_flight == 1
        release_budget()
        assert _gs.in_flight == 0

    @pytest.mark.asyncio
    async def test_respects_rpm_limit(self) -> None:
        _gs.rpm_limit = 2
        _gs.requests_this_minute = 2
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(acquire_budget(100), timeout=0.3)

    def test_status_initial(self) -> None:
        s = get_governor_status()
        assert s["rpm_limit"] == 70
        assert s["tpm_limit"] == 50000
        assert s["max_in_flight"] == 2
        assert "cooldown_active" in s


class TestDomainValidation:
    def test_techcrunch_is_aggregator(self) -> None:
        assert is_aggregator_domain("techcrunch.com")
        assert _gov_is_agg("techcrunch.com")

    def test_linkedin_is_aggregator(self) -> None:
        assert is_aggregator_domain("linkedin.com")

    def test_company_domain_is_not_aggregator(self) -> None:
        assert not is_aggregator_domain("stripe.com")
        assert not is_aggregator_domain("example.io")
        assert not is_aggregator_domain("boards.greenhouse.io")

    def test_crunchbase_is_aggregator(self) -> None:
        assert is_aggregator_domain("crunchbase.com")

    def test_producthunt_is_aggregator(self) -> None:
        assert is_aggregator_domain("producthunt.com")

    def test_vc_sites_are_aggregators(self) -> None:
        assert is_aggregator_domain("a16z.com")
        assert is_aggregator_domain("sequoiacap.com")
        assert is_aggregator_domain("benchmark.com")
        assert is_aggregator_domain("accel.com")

    def test_job_boards_are_aggregators(self) -> None:
        assert is_aggregator_domain("wellfound.com")
        assert is_aggregator_domain("indeed.com")
        assert is_aggregator_domain("glassdoor.com")
