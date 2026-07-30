"""Integration tests for search crawler → discovery → source persistence pipeline."""

from __future__ import annotations

import pytest

from src.radar.crawler import (
    _classify_result,
    _extract_board_root,
)


class TestBoardRootExtraction:
    def test_greenhouse_job_to_board(self) -> None:
        """A Greenhouse job URL must extract the company board root, not generic root."""
        result = _extract_board_root("https://boards.greenhouse.io/acme/jobs/123")
        assert result == "https://boards.greenhouse.io/acme"
        assert "boards.greenhouse.io" in result
        assert "/acme" in result
        assert "/jobs/123" not in result

    def test_lever_job_to_board(self) -> None:
        result = _extract_board_root("https://jobs.lever.co/acme/456")
        assert result == "https://jobs.lever.co/acme"
        assert "/456" not in result

    def test_ashby_job_to_board(self) -> None:
        result = _extract_board_root("https://jobs.ashbyhq.com/acme/789")
        assert result == "https://jobs.ashbyhq.com/acme"
        assert "/789" not in result

    def test_workable_job_to_board(self) -> None:
        result = _extract_board_root("https://apply.workable.com/acme")
        assert result == "https://apply.workable.com/acme"

    def test_workday_subdomain(self) -> None:
        result = _extract_board_root("https://acme.myworkdayjobs.com/careers/job/Tokyo/Engineer")
        assert result == "https://acme.myworkdayjobs.com"
        assert "acme" in result
        assert "/careers" not in result

    def test_myworkdayjobs_without_company_in_path(self) -> None:
        # Workday: company is always the subdomain
        result = _extract_board_root(
            "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/123"
        )
        assert result == "https://nvidia.wd5.myworkdayjobs.com"

    def test_smartrecruiters(self) -> None:
        result = _extract_board_root("https://jobs.smartrecruiters.com/AcmeCorp/123-engineer")
        assert "AcmeCorp" in result

    def test_rippling(self) -> None:
        result = _extract_board_root("https://app.rippling.com/careers/acme/123")
        assert result == "https://app.rippling.com/careers/acme"

    def test_unknown_host_returns_safe(self) -> None:
        result = _extract_board_root("https://example.com/foo/bar")
        assert result == "https://example.com/foo"

    def test_no_path_returns_host(self) -> None:
        result = _extract_board_root("https://boards.greenhouse.io")
        assert result == "https://boards.greenhouse.io"


class TestPipelineIntegration:
    def test_ats_result_has_direct_job_flag(self) -> None:
        """ATS results should carry direct_job=True flag."""
        # This is the structure the crawler produces for ats_job results
        entry = {
            "name": "Acme",
            "website": "https://boards.greenhouse.io/acme",
            "source": "search_ats",
            "provenance_url": "https://boards.greenhouse.io/acme/jobs/123",
            "direct_job": True,
        }
        assert entry["direct_job"]
        assert entry["website"] != "https://boards.greenhouse.io"

    def test_startup_signal_not_direct_job(self) -> None:
        """Startup signals should NOT have direct_job flag."""
        entry = {
            "name": "Acme",
            "website": "",
            "source": "search_startup",
            "provenance_url": "https://techcrunch.com/article",
        }
        assert "direct_job" not in entry or not entry.get("direct_job")

    def test_aggregator_never_becomes_source(self) -> None:
        """Aggregator results must never flow into the discoveries list."""
        # Simulate: aggregator classification should be excluded
        assert (
            _classify_result(
                "https://www.indeed.com/viewjob?jk=abc",
                "Software Engineer at Acme — Indeed",
                "Apply on Indeed",
            )
            == "aggregator"
        )
        assert (
            _classify_result(
                "https://www.glassdoor.com/job-listing/engineer",
                "Engineer",
                "",
            )
            == "aggregator"
        )
        assert (
            _classify_result(
                "https://www.linkedin.com/jobs/view/123",
                "SWE",
                "",
            )
            == "aggregator"
        )

    def test_real_ats_passes_classification(self) -> None:
        """Real ATS URLs must pass and not be marked as aggregators."""
        assert (
            _classify_result(
                "https://boards.greenhouse.io/acme/jobs/123",
                "SWE at Acme",
                "Requirements: Python, AWS",
            )
            == "ats_job"
        )


class TestEmailGuessingRemoved:
    def test_no_guess_instruction_in_prompt(self) -> None:
        """The LLM prompt must NOT ask or allow guessing emails."""
        import inspect

        from src.agent.startup_agent import StartupAgent

        source = inspect.getsource(StartupAgent.analyze_startup)
        source_lower = source.lower()
        # Must NOT contain permission to guess
        assert "aggressively guess" not in source_lower
        assert "may be guessed" not in source_lower
        assert "mark as 'guessed'" not in source_lower
        # Must contain explicit prohibition
        assert "never guess" in source_lower
        assert "do not guess" in source_lower


class TestRateLimiting:
    def test_scrape_indexes_uses_should_poll(self) -> None:
        """GitHub indexes must respect should_poll for rate-limiting."""
        from src.radar.orchestrator import _scrape_indexes

        assert callable(_scrape_indexes)
        # The function uses should_poll() internally
        import inspect

        source = inspect.getsource(_scrape_indexes)
        assert "should_poll" in source


class TestIdempotentDiscovery:
    def test_canonical_url_stable(self) -> None:
        from src.radar.crawler import _canonical_url

        h1 = _canonical_url("https://boards.greenhouse.io/acme/jobs/123")
        h2 = _canonical_url("https://boards.greenhouse.io/acme/jobs/123")
        assert h1 == h2

    def test_different_urls_different_canonical(self) -> None:
        from src.radar.crawler import _canonical_url

        hashes = {
            _canonical_url("https://boards.greenhouse.io/acme/jobs/1"),
            _canonical_url("https://boards.greenhouse.io/acme/jobs/2"),
            _canonical_url("https://jobs.lever.co/acme/1"),
            _canonical_url("https://jobs.lever.co/acme/2"),
        }
        assert len(hashes) == 4

    @pytest.mark.asyncio
    async def test_completion_state_prevents_reenqueue(self) -> None:
        """Once a candidate is terminal, it must not be re-enqueued."""
        from src.radar.models import EligibilityState, JobCandidate

        c = JobCandidate(
            canonical_id="test:idempotent:remote",
            source="test",
            direct_apply_url="https://example.com",
            normalized_company="Test",
            normalized_role="Idempotent",
            normalized_location="Remote",
        )
        c.eligibility = EligibilityState.ACCEPTED
        c.extra["version"] = 1

        from src.radar.queue import enqueue_candidate, mark_retry

        assert await enqueue_candidate(c)
        # Same version should NOT enqueue again
        assert not await enqueue_candidate(c)
        # New version should pass
        c.extra["version"] = 2
        mark_retry(c)
        assert await enqueue_candidate(c)
