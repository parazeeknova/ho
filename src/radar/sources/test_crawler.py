"""Integration tests: board roots, company parsing, classification, end-to-end pipeline."""

from __future__ import annotations

import inspect

from src.radar.sources.crawler import (
    _canonical_url,
    _classify_result,
    _extract_board_root,
    _extract_company_from_title,
    _extract_domain,
)


class TestBoardRootExtraction:
    def test_greenhouse_job_to_board(self) -> None:
        result = _extract_board_root("https://boards.greenhouse.io/acme/jobs/123")
        assert result == "https://boards.greenhouse.io/acme"
        assert "/jobs/123" not in result

    def test_lever_job_to_board(self) -> None:
        result = _extract_board_root("https://jobs.lever.co/acme/456")
        assert result == "https://jobs.lever.co/acme"

    def test_ashby_job_to_board(self) -> None:
        result = _extract_board_root("https://jobs.ashbyhq.com/acme/789")
        assert result == "https://jobs.ashbyhq.com/acme"

    def test_workable_job_to_board(self) -> None:
        assert (
            _extract_board_root("https://apply.workable.com/acme")
            == "https://apply.workable.com/acme"
        )

    def test_workday_subdomain(self) -> None:
        result = _extract_board_root("https://acme.myworkdayjobs.com/careers/job/Tokyo/Engineer")
        assert result == "https://acme.myworkdayjobs.com"

    def test_myworkdayjobs_without_company_in_path(self) -> None:
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
        assert _extract_board_root("https://example.com/foo/bar") == "https://example.com/foo"


class TestCompanyNameExtraction:
    def test_role_at_company(self) -> None:
        """'SWE at Acme' → Acme (company is on the RIGHT of ' at ')."""
        assert _extract_company_from_title("Software Engineer at Acme") == "Acme"

    def test_backend_at_company(self) -> None:
        assert _extract_company_from_title("Backend Engineer at Acme Corp") == "Acme Corp"

    def test_intern_at_company(self) -> None:
        assert _extract_company_from_title("SWE Intern at Stripe") == "Stripe"

    def test_company_is_hiring(self) -> None:
        assert _extract_company_from_title("Acme Corp is hiring a Software Engineer") == "Acme Corp"

    def test_company_hiring(self) -> None:
        assert _extract_company_from_title("Acme Corp hiring Backend Engineers") == "Acme Corp"

    def test_dash_separator(self) -> None:
        assert _extract_company_from_title("Acme Corp - Backend Engineer") == "Acme Corp"

    def test_pipe_separator(self) -> None:
        assert _extract_company_from_title("Acme Corp | Careers") == "Acme Corp"

    def test_em_dash(self) -> None:
        assert _extract_company_from_title("Acme Corp — hiring!") == "Acme Corp"

    def test_raises_format(self) -> None:
        name = _extract_company_from_title("Acme Corp raises $10M for dev tools")
        assert "Acme Corp" in name

    def test_short_title_fallback(self) -> None:
        assert _extract_company_from_title("Acme") == "Acme"

    def test_role_at_short_name_still_works(self) -> None:
        """Even with single-word company, 'at' pattern correctly extracts."""
        assert _extract_company_from_title("Engineer at Foo") == "Foo"

    def test_long_role_at_short_company(self) -> None:
        assert _extract_company_from_title("Senior Software Engineer at Z") == "Z"


class TestClassification:
    def test_classify_greenhouse(self) -> None:
        assert (
            _classify_result(
                "https://boards.greenhouse.io/acme/jobs/123",
                "Software Engineer at Acme",
                "Requirements: Python, AWS.",
            )
            == "ats_job"
        )

    def test_classify_lever(self) -> None:
        assert (
            _classify_result(
                "https://jobs.lever.co/acme/456",
                "Backend Engineer",
                "Join our team. Qualifications: Go, K8s.",
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

    def test_real_ats_passes_not_aggregator(self) -> None:
        assert (
            _classify_result(
                "https://boards.greenhouse.io/acme/jobs/123",
                "SWE at Acme",
                "Requirements: Python",
            )
            == "ats_job"
        )


class TestSourceIdCollisionProtection:
    def test_different_companies_produce_different_ids(self) -> None:
        """Two different companies discovered via ats_job get distinct source IDs."""
        ids = [
            f"discovered:{_extract_company_from_title('Backend Engineer at Stripe').lower().replace(' ', '-')[:60]}",  # noqa: E501
            f"discovered:{_extract_company_from_title('Backend Engineer at Acme').lower().replace(' ', '-')[:60]}",  # noqa: E501
        ]
        assert ids[0] != ids[1]
        assert "stripe" in ids[0]
        assert "acme" in ids[1]

    def test_canonical_board_url_dedup(self) -> None:
        """Two URLs pointing to same board should canonicalize to same source."""
        b1 = _extract_board_root("https://boards.greenhouse.io/acme/jobs/1")
        b2 = _extract_board_root("https://boards.greenhouse.io/acme/jobs/2")
        assert b1 == b2

    def test_board_url_hash_protects_against_name_collision(self) -> None:
        """Same company name at different board URLs → distinct source IDs."""
        import hashlib

        name1 = "Acme"
        name2 = "Acme"  # same name
        url1 = "https://boards.greenhouse.io/acme-ai"
        url2 = "https://jobs.lever.co/acme-labs"

        def _make_id(name: str, url: str) -> str:
            slug = name.lower().replace(" ", "-")[:40]
            board_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
            return f"discovered:{slug}:{board_hash}"

        id1 = _make_id(name1, url1)
        id2 = _make_id(name2, url2)

        assert id1 != id2, (
            f"Same company name / different board URL must produce distinct IDs: {id1} vs {id2}"
        )
        assert "acme" in id1
        assert "acme" in id2


class TestEmailGuessingRemoved:
    def test_no_guess_in_analyze_startup_source(self) -> None:
        from src.agent.startup_agent import StartupAgent

        source = inspect.getsource(StartupAgent.analyze_startup).lower()
        assert "aggressively guess" not in source
        assert "may be guessed" not in source
        assert "never guess" in source
        assert "do not guess" in source


class TestCanonicalUrl:
    def test_same_url(self) -> None:
        assert _canonical_url("https://jobs.lever.co/acme/123") == _canonical_url(
            "https://jobs.lever.co/acme/123"
        )

    def test_trailing_slash(self) -> None:
        assert _canonical_url("https://jobs.lever.co/acme/123/") == _canonical_url(
            "https://jobs.lever.co/acme/123"
        )

    def test_different_urls(self) -> None:
        assert _canonical_url("https://jobs.lever.co/acme/1") != _canonical_url(
            "https://jobs.lever.co/acme/2"
        )

    def test_domain_extraction_strips_www(self) -> None:
        assert _extract_domain("https://www.techcrunch.com/news") == "techcrunch.com"

    def test_four_urls_produce_four_unique_hashes(self) -> None:
        hashes = {
            _canonical_url("https://boards.greenhouse.io/acme/jobs/1"),
            _canonical_url("https://boards.greenhouse.io/acme/jobs/2"),
            _canonical_url("https://jobs.lever.co/acme/1"),
            _canonical_url("https://jobs.lever.co/acme/2"),
        }
        assert len(hashes) == 4


class TestEndToEndPipeline:
    def test_discovery_output_structure(self) -> None:
        """Simulate crawler output and verify downstream ingestion would work."""
        search_result = {
            "url": "https://boards.greenhouse.io/stellar/jobs/789",
            "title": "Backend Engineer at Stellar",
            "content": "Apply now. Requirements: Python, AWS. Qualifications.",
        }
        classification = _classify_result(
            search_result["url"],
            search_result["title"],
            search_result["content"],
        )
        assert classification == "ats_job"

        board_url = _extract_board_root(search_result["url"])
        assert board_url == "https://boards.greenhouse.io/stellar"
        assert board_url != "https://boards.greenhouse.io"

        company = _extract_company_from_title(search_result["title"])
        assert company == "Stellar"
        assert company != "Backend Engineer"

        source_id = f"discovered:{company.lower().replace(' ', '-')[:60]}"
        assert "stellar" in source_id
        assert "backend-engineer" not in source_id

        discovery = {
            "name": company,
            "website": board_url,
            "source": "search_ats",
            "provenance_url": search_result["url"],
            "direct_job": True,
        }
        assert discovery["direct_job"]
        assert discovery["website"] == "https://boards.greenhouse.io/stellar"
        assert discovery["name"] == "Stellar"

    def test_aggregator_never_creates_source(self) -> None:
        search_result = {
            "url": "https://www.indeed.com/viewjob?jk=abc",
            "title": "Software Engineer - Acme Corp - Indeed",
            "content": "Apply on Indeed for this role.",
        }
        assert (
            _classify_result(
                search_result["url"],
                search_result["title"],
                search_result["content"],
            )
            == "aggregator"
        )

    def test_startup_signal_structure(self) -> None:
        search_result = {
            "url": "https://techcrunch.com/2026/07/nova-raises-20m/",
            "title": "Nova raises $20M Series A for AI observability",
            "content": "Nova, a YC-backed startup, announced a $20M round...",
        }
        assert (
            _classify_result(
                search_result["url"],
                search_result["title"],
                search_result["content"],
            )
            == "startup_signal"
        )
        assert "Nova" in _extract_company_from_title(search_result["title"])

    def test_two_distinct_companies_produce_distinct_source_ids(self) -> None:
        r1 = {
            "url": "https://boards.greenhouse.io/stellar/jobs/1",
            "title": "SWE Intern at Stellar",
            "content": "Apply now.",
        }
        r2 = {
            "url": "https://jobs.lever.co/nova/42",
            "title": "Backend Engineer at Nova",
            "content": "Apply now.",
        }
        d1 = {
            "name": _extract_company_from_title(r1["title"]),
            "website": _extract_board_root(r1["url"]),
        }
        d2 = {
            "name": _extract_company_from_title(r2["title"]),
            "website": _extract_board_root(r2["url"]),
        }
        assert d1["name"] == "Stellar"
        assert d2["name"] == "Nova"
        assert d1["website"] == "https://boards.greenhouse.io/stellar"
        assert d2["website"] == "https://jobs.lever.co/nova"
        assert d1["website"] != d2["website"]
        assert (
            f"discovered:{d1['name'].lower().replace(' ', '-')[:60]}"
            != f"discovered:{d2['name'].lower().replace(' ', '-')[:60]}"
        )
