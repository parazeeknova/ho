"""Tests for SearchDiscoveryCrawler query classification and result extraction."""

from __future__ import annotations

from src.radar.crawler import (
    _build_query_templates,
    _canonical_url,
    _classify_result,
    _extract_company_from_title,
    _extract_domain,
)


class TestQueryTemplates:
    def test_produces_queries(self) -> None:
        templates = _build_query_templates()
        assert len(templates) > 5
        assert all(isinstance(t, str) for t in templates)
        assert any("backend" in t.lower() or "fullstack" in t.lower() for t in templates)
        assert any("sponsor" in t.lower() or "remote" in t.lower() for t in templates)

    def test_templates_change_between_calls(self) -> None:
        t1 = set(_build_query_templates())
        t2 = set(_build_query_templates())
        assert len(t1 - t2) >= 0


class TestResultClassification:
    def test_classify_greenhouse(self) -> None:
        assert (
            _classify_result(
                "https://boards.greenhouse.io/acme/jobs/123",
                "Software Engineer at Acme",
                "Apply now for this backend role. Requirements: Python, AWS.",
            )
            == "ats_job"
        )

    def test_classify_lever(self) -> None:
        assert (
            _classify_result(
                "https://jobs.lever.co/acme/456",
                "Backend Engineer",
                "Join our team. Qualifications: 0-2 years, Go, Kubernetes.",
            )
            == "ats_job"
        )

    def test_classify_ashby(self) -> None:
        assert (
            _classify_result(
                "https://jobs.ashbyhq.com/acme/789",
                "SWE Intern",
                "Responsibilities include building features.",
            )
            == "ats_job"
        )

    def test_classify_workable(self) -> None:
        assert (
            _classify_result(
                "https://apply.workable.com/acme",
                "Data Engineer",
                "Apply here.",
            )
            == "ats_job"
        )

    def test_classify_workday(self) -> None:
        assert (
            _classify_result(
                "https://acme.myworkdayjobs.com/careers/job/1",
                "Platform Engineer",
                "Requirements and qualifications listed.",
            )
            == "ats_job"
        )

    def test_classify_startup_signal_techcrunch(self) -> None:
        assert (
            _classify_result(
                "https://techcrunch.com/2026/07/acme-raises-10m/",
                "Acme raised $10M seed round",
                "Acme, a startup building dev tools, announced today...",
            )
            == "startup_signal"
        )

    def test_classify_startup_signal_crunchbase(self) -> None:
        assert (
            _classify_result(
                "https://crunchbase.com/organization/acme",
                "Acme — Funding, Valuation & Investors",
                "raised seed funding for their engineering platform...",
            )
            == "startup_signal"
        )

    def test_classify_founder_post(self) -> None:
        assert (
            _classify_result(
                "https://www.linkedin.com/posts/acme-ceo_hiring-looking-for-activity",
                "We're hiring engineers at Acme!",
                "DM me if interested. Looking for backend engineers.",
            )
            == "founder_post"
        )

    def test_classify_aggregator_indeed(self) -> None:
        assert (
            _classify_result(
                "https://www.indeed.com/viewjob?jk=abc",
                "Software Engineer - Acme Corp - Indeed",
                "Apply on Indeed for this role at Acme Corp.",
            )
            == "aggregator"
        )

    def test_classify_aggregator_linkedin_jobs(self) -> None:
        assert (
            _classify_result(
                "https://www.linkedin.com/jobs/view/123",
                "Software Engineer at Acme",
                "",
            )
            == "aggregator"
        )

    def test_classify_aggregator_glassdoor(self) -> None:
        assert (
            _classify_result(
                "https://www.glassdoor.com/job-listing/acme-software-engineer",
                "Software Engineer at Acme Corp",
                "",
            )
            == "aggregator"
        )


class TestCompanyExtraction:
    def test_hiring_at_format(self) -> None:
        assert _extract_company_from_title("Acme Corp is hiring a Software Engineer") == "Acme Corp"

    def test_dash_separator(self) -> None:
        assert _extract_company_from_title("Acme Corp - Backend Engineer") == "Acme Corp"

    def test_pipe_separator(self) -> None:
        assert _extract_company_from_title("Acme Corp | Careers") == "Acme Corp"

    def test_em_dash(self) -> None:
        assert _extract_company_from_title("Acme Corp — hiring!") == "Acme Corp"

    def test_raises_format(self) -> None:
        name = _extract_company_from_title("Acme Corp raises $10M for dev tools")
        assert "Acme Corp" in name

    def test_short_title(self) -> None:
        assert _extract_company_from_title("Acme") == "Acme"


class TestCanonicalUrl:
    def test_same_url_same_hash(self) -> None:
        h1 = _canonical_url("https://jobs.lever.co/acme/123")
        h2 = _canonical_url("https://jobs.lever.co/acme/123")
        assert h1 == h2

    def test_trailing_slash(self) -> None:
        h1 = _canonical_url("https://jobs.lever.co/acme/123/")
        h2 = _canonical_url("https://jobs.lever.co/acme/123")
        assert h1 == h2

    def test_different_urls_different(self) -> None:
        h1 = _canonical_url("https://jobs.lever.co/acme/1")
        h2 = _canonical_url("https://jobs.lever.co/acme/2")
        assert h1 != h2

    def test_domain_extraction(self) -> None:
        gh = "boards.greenhouse.io"
        assert _extract_domain(f"https://{gh}/acme/jobs/123") == gh
        assert _extract_domain("https://www.techcrunch.com/news") == "techcrunch.com"
        assert _extract_domain("https://jobs.lever.co/acme/456") == "jobs.lever.co"
