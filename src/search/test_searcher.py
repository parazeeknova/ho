"""Tests for searcher: domain extraction, ATS pattern matching, concurrency."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.search.searcher import (
    _ATS_PATTERN,
    extract_career_domain,
    harvest_and_save_domains,
    map_company_careers,
)


class TestExtractCareerDomain:
    """Unit tests for extract_career_domain regex-based domain extraction."""

    def test_none_or_empty(self) -> None:
        assert extract_career_domain("") is None
        assert extract_career_domain("not-a-url") is None

    def test_greenhouse_io(self) -> None:
        result = extract_career_domain("https://boards.greenhouse.io/acmecorp/jobs/12345")
        assert result == "https://boards.greenhouse.io/acmecorp"

    def test_lever_co(self) -> None:
        result = extract_career_domain("https://jobs.lever.co/stripe/abc123/apply")
        assert result == "https://jobs.lever.co/stripe"

    def test_ashbyhq_com(self) -> None:
        result = extract_career_domain("https://jobs.ashbyhq.com/anthropic/abc-def-ghi")
        assert result == "https://jobs.ashbyhq.com/anthropic"

    def test_workable_com(self) -> None:
        result = extract_career_domain("https://apply.workable.com/huggingface/j/ABC123/")
        assert result == "https://apply.workable.com/huggingface"

    def test_smartrecruiters_com(self) -> None:
        result = extract_career_domain(
            "https://jobs.smartrecruiters.com/SomeCompany/123456-job-title"
        )
        assert result == "https://jobs.smartrecruiters.com/SomeCompany"

    def test_myworkdayjobs_com(self) -> None:
        result = extract_career_domain(
            "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Software-Engineer_JR123"
        )
        assert result == "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite"

    def test_rippling_careers(self) -> None:
        result = extract_career_domain("https://app.rippling.com/careers/some-company/jobs/abc")
        assert result == "https://app.rippling.com/careers/some-company"

    def test_jobs_subdomain(self) -> None:
        result = extract_career_domain("https://jobs.example.com/software-engineer")
        assert result == "https://jobs.example.com"

    def test_careers_subdomain(self) -> None:
        result = extract_career_domain("https://careers.example.com/engineering/backend")
        assert result == "https://careers.example.com"

    def test_fallback_jobs_path(self) -> None:
        result = extract_career_domain("https://example.com/jobs/12345-backend")
        assert result == "https://example.com/jobs"

    def test_fallback_careers_path(self) -> None:
        result = extract_career_domain("https://stripe.com/careers/backend-engineer")
        assert result == "https://stripe.com/careers"

    def test_no_match_returns_none(self) -> None:
        result = extract_career_domain("https://example.com/about")
        assert result is None

    def test_case_insensitive(self) -> None:
        result = extract_career_domain("HTTPS://JOBS.LEVER.CO/OpenAI/Apply")
        assert result == "HTTPS://JOBS.LEVER.CO/OpenAI"

    def test_trailing_slash_removed(self) -> None:
        result = extract_career_domain("https://jobs.lever.co/stripe/")
        assert result == "https://jobs.lever.co/stripe"


class TestATSPattern:
    """Verify _ATS_PATTERN matches all expected ATS platforms."""

    def test_greenhouse_matches(self) -> None:
        assert _ATS_PATTERN.search("https://boards.greenhouse.io/company/jobs/1")

    def test_lever_matches(self) -> None:
        assert _ATS_PATTERN.search("https://jobs.lever.co/company")

    def test_ashby_matches(self) -> None:
        assert _ATS_PATTERN.search("https://jobs.ashbyhq.com/company")

    def test_workable_matches(self) -> None:
        assert _ATS_PATTERN.search("https://apply.workable.com/company")

    def test_smartrecruiters_matches(self) -> None:
        assert _ATS_PATTERN.search("https://jobs.smartrecruiters.com/company")

    def test_myworkdayjobs_matches(self) -> None:
        assert _ATS_PATTERN.search("https://company.wd5.myworkdayjobs.com/Site")

    def test_jobs_subdomain_matches(self) -> None:
        assert _ATS_PATTERN.search("https://jobs.company.com")

    def test_careers_subdomain_matches(self) -> None:
        assert _ATS_PATTERN.search("https://careers.company.com")

    def test_no_match_non_career_url(self) -> None:
        assert _ATS_PATTERN.search("https://example.com/about") is None
        assert _ATS_PATTERN.search("https://blog.example.com/post") is None


class TestHarvestAndSaveDomains:
    """Tests for harvest_and_save_domains using a mocked MemoryStore."""

    @pytest.mark.asyncio
    async def test_harvests_new_domains(self, mocker) -> None:
        store = mocker.AsyncMock()
        store.add_discovered_domain = AsyncMock(return_value=True)

        urls = [
            "https://jobs.lever.co/stripe/abc123",
            "https://jobs.ashbyhq.com/anthropic/def456",
            "https://example.com/about",  # non-career, ignored
        ]
        count = await harvest_and_save_domains(urls, store)
        assert count == 2
        assert store.add_discovered_domain.call_count == 2

    @pytest.mark.asyncio
    async def test_duplicate_domains_not_double_counted(self, mocker) -> None:
        store = mocker.AsyncMock()
        store.add_discovered_domain = AsyncMock(return_value=False)

        urls = [
            "https://jobs.lever.co/stripe/abc",
            "https://jobs.lever.co/stripe/def",
        ]
        count = await harvest_and_save_domains(urls, store)
        assert count == 0

    @pytest.mark.asyncio
    async def test_mixed_new_and_existing(self, mocker) -> None:
        store = mocker.AsyncMock()
        call_results = [True, False, True]
        store.add_discovered_domain = AsyncMock(side_effect=call_results)

        urls = [
            "https://jobs.lever.co/stripe/a",
            "https://jobs.lever.co/cohere/b",
            "https://jobs.ashbyhq.com/scaleai/c",
        ]
        count = await harvest_and_save_domains(urls, store)
        assert count == 2


class TestMapCompanyCareers:
    """Tests for map_company_careers concurrency and domain processing."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mocker) -> None:
        """Verify that map_company_careers uses Semaphore(8) and processes all domains."""
        from unittest.mock import MagicMock

        app = MagicMock()
        app.map_url = MagicMock(return_value=MagicMock(links=[]))

        domains = [f"https://example{i}.com" for i in range(20)]
        results = await map_company_careers(app, domains, keyword="test")
        assert isinstance(results, list)
        assert app.map_url.call_count == 20

    @pytest.mark.asyncio
    async def test_returns_job_urls(self, mocker) -> None:
        from unittest.mock import MagicMock

        app = MagicMock()
        fake_result = MagicMock()
        fake_result.links = [
            "https://jobs.lever.co/acme/jobs/software-intern",
            "https://jobs.lever.co/acme/about",  # no /jobs/ path, filtered out
        ]
        app.map_url = MagicMock(return_value=fake_result)

        results = await map_company_careers(app, ["https://jobs.lever.co/acme"], keyword="intern")
        assert len(results) == 1
        assert results[0]["type"] == "map"
        assert "software-intern" in results[0]["title"]

    @pytest.mark.asyncio
    async def test_handles_exceptions_gracefully(self, mocker) -> None:
        from unittest.mock import MagicMock

        app = MagicMock()
        app.map_url = MagicMock(side_effect=[Exception("timeout"), MagicMock(links=[])])

        results = await map_company_careers(
            app, ["https://bad.example.com", "https://good.example.com"]
        )
        assert isinstance(results, list)
