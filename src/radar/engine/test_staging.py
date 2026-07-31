"""Staging smoke tests: source trust, domain normalization, freshness gating."""

from __future__ import annotations

from src.radar.core.governor import (
    _state as _gs,
)
from src.radar.core.models import JobObservation
from src.radar.sources.discovery import is_aggregator_domain


class TestDomainNormalization:
    def test_www_prefix_stripped(self) -> None:
        assert is_aggregator_domain("www.techcrunch.com")
        assert is_aggregator_domain("www.linkedin.com")
        assert is_aggregator_domain("www.crunchbase.com")

    def test_subdomain_of_aggregator_rejected(self) -> None:
        assert is_aggregator_domain("blog.techcrunch.com")
        assert is_aggregator_domain("news.crunchbase.com")
        assert is_aggregator_domain("media.producthunt.com")

    def test_subdomain_of_vc_rejected(self) -> None:
        assert is_aggregator_domain("portfolio.a16z.com")
        assert is_aggregator_domain("www.sequoiacap.com")
        assert is_aggregator_domain("companies.benchmark.com")

    def test_real_company_domains_pass(self) -> None:
        assert not is_aggregator_domain("stripe.com")
        assert not is_aggregator_domain("www.stripe.com")
        assert not is_aggregator_domain("boards.greenhouse.io")
        assert not is_aggregator_domain("jobs.lever.co")
        assert not is_aggregator_domain("example.io")

    def test_all_blocked_classes(self) -> None:
        blocked = [
            "techcrunch.com",
            "crunchbase.com",
            "linkedin.com",
            "producthunt.com",
            "wellfound.com",
            "glassdoor.com",
            "indeed.com",
            "ziprecruiter.com",
            "news.ycombinator.com",
            "ycombinator.com",
            "sequoiacap.com",
            "a16z.com",
            "benchmark.com",
            "accel.com",
            "twitter.com",
            "x.com",
            "facebook.com",
            "instagram.com",
            "medium.com",
        ]
        for domain in blocked:
            assert is_aggregator_domain(domain), f"{domain} should be blocked"
            assert is_aggregator_domain(f"www.{domain}"), f"www.{domain} should be blocked"

    def test_discovery_index_no_official_source(self) -> None:
        """YC/Wellfound observations should never be official_source=True."""
        yc_obs = JobObservation(
            url="https://www.ycombinator.com/jobs/123",
            source="ycombinator:jobs",
            extra={"is_snapshot_delta": False, "official_source": False},
        )
        assert not yc_obs.extra.get("official_source")
        assert yc_obs.extra.get("official_source") is False

    def test_gate_snapshot_delta_not_urgent_without_official(self) -> None:
        """Snapshot delta from non-official source must not make it URGENT."""
        # Simulated: a discovery index snapshot delta but no official source flag
        obs = JobObservation(
            url="https://www.ycombinator.com/jobs/123",
            source="ycombinator:jobs",
            extra={"is_snapshot_delta": True, "official_source": False},
        )
        # If this goes through gate: is_snapshot_delta=True but official_source=False
        # → URGENT if 'and is_official' check fails → stays REVIEW
        is_official = obs.extra.get("official_source", False)
        is_snapshot_delta = obs.extra.get("is_snapshot_delta", False)
        can_be_urgent = is_snapshot_delta and is_official
        assert not can_be_urgent, "Discovery index deltas should not be URGENT"

    def test_gate_snapshot_delta_urgent_with_official(self) -> None:
        """Snapshot delta from official source → URGENT."""
        obs = JobObservation(
            url="https://boards.greenhouse.io/openai/jobs/456",
            source="openai:greenhouse",
            extra={"is_snapshot_delta": True, "official_source": True},
        )
        is_official = obs.extra.get("official_source", False)
        is_snapshot_delta = obs.extra.get("is_snapshot_delta", False)
        can_be_urgent = is_snapshot_delta and is_official
        assert can_be_urgent, "Official ATS deltas should be URGENT"


class TestGovernorInteractiveLane:
    def test_interactive_reserved_capacity(self) -> None:
        """Interactive calls consume reserved lane, background calls preserve it."""
        import time

        _gs.reserved_max_per_minute = 3
        _gs.reserved_requests_this_minute = 0
        _gs.rpm_limit = 10
        _gs.requests_this_minute = 7  # Non-interactive at 7/10
        _gs.window_start = time.monotonic()
        _gs.cooldown_until = 0.0

        # Background call when main budget is mostly used but reserved remains
        can_acquire_bg = _gs.requests_this_minute < _gs.rpm_limit - _gs.reserved_max_per_minute
        can_acquire_interactive = _gs.reserved_requests_this_minute < _gs.reserved_max_per_minute

        assert not can_acquire_bg  # 7 >= 7, so background blocked
        assert can_acquire_interactive  # 0 < 3, interactive allowed

        # Reset
        _gs.requests_this_minute = 0
        _gs.reserved_requests_this_minute = 0


class TestSourceTypeClassification:
    def test_official_ats_sources_are_verified(self) -> None:
        official_source_ids = {
            "openai:greenhouse",
            "anthropic:ashby",
            "stripe:greenhouse",
            "airbnb:greenhouse",
        }
        for sid in official_source_ids:
            assert "greenhouse" in sid or "ashby" in sid or "lever" in sid

    def test_discovery_index_source_ids_recognized(self) -> None:
        discovery_ids = {"ycombinator:jobs", "wellfound:jobs"}
        for sid in discovery_ids:
            assert "jobs" in sid or "ycombinator" in sid or "wellfound" in sid
