"""Tests for radar v2 pipeline: models, gates, salary, canonicalization,
freshness, and GitHub index extraction.
"""

from __future__ import annotations

import pytest

from src.radar.extractors import (
    _extract_link,
    extract_github_index_markdown,
)
from src.radar.gates import (
    gate_explicit_experience,
    gate_role_family,
    gate_title_seniority,
    gate_url_duplicate,
    gate_url_quality,
    run_gates,
)
from src.radar.models import (
    EligibilityState,
    JobCandidate,
    JobObservation,
    RejectionReason,
    RoleFamily,
    make_canonical_id,
)
from src.radar.salary import normalize_salary


class TestSalaryNormalizer:
    def test_usd_single(self) -> None:
        s = normalize_salary("$120,000 per year")
        assert s is not None
        assert s.amount == 120000
        assert s.currency == "USD"
        assert s.period == "year"

    def test_usd_range_midpoint(self) -> None:
        s = normalize_salary("$50 - $70/hr")
        assert s is not None
        assert s.amount == 60.0
        assert s.currency == "USD"
        assert s.period == "hour"

    def test_inr_lpa(self) -> None:
        s = normalize_salary("15 LPA")
        assert s is not None
        assert s.amount == 1500000
        assert s.currency == "INR"
        assert s.period == "year"

    def test_inr_monthly(self) -> None:
        s = normalize_salary("₹70,000 per month")
        assert s is not None
        assert s.amount == 70000
        assert s.currency == "INR"
        assert s.period == "month"

    def test_usd_k_format(self) -> None:
        s = normalize_salary("120K USD per year")
        assert s is not None
        assert s.amount == 120000
        assert s.currency == "USD"

    def test_none_input(self) -> None:
        assert normalize_salary(None) is None

    def test_empty_input(self) -> None:
        assert normalize_salary("") is None
        assert normalize_salary("  ") is None

    def test_inr_lakhs(self) -> None:
        s = normalize_salary("12 lakhs per annum")
        assert s is not None
        assert s.amount == 1200000
        assert s.currency == "INR"
        assert s.period == "year"

    def test_eur(self) -> None:
        s = normalize_salary("€60,000 per year")
        assert s is not None
        assert s.amount == 60000
        assert s.currency == "EUR"


class TestGatesUrlQuality:
    def test_no_url(self) -> None:
        obs = JobObservation(url="", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_quality(obs, candidate, set(), {})
        assert result == RejectionReason.URL_BAD

    def test_directory_domain(self) -> None:
        obs = JobObservation(url="https://www.indeed.com/jobs", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_quality(obs, candidate, set(), {})
        assert result == RejectionReason.URL_DIRECTORY

    def test_landing_page(self) -> None:
        obs = JobObservation(url="https://company.com/careers", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_quality(obs, candidate, set(), {})
        assert result == RejectionReason.URL_LANDING_PAGE

    def test_error_url(self) -> None:
        obs = JobObservation(url="https://company.com/jobs/404", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_quality(obs, candidate, set(), {})
        assert result == RejectionReason.URL_ERROR_404

    def test_image_url(self) -> None:
        obs = JobObservation(url="https://company.com/logo.png", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_quality(obs, candidate, set(), {})
        assert result == RejectionReason.URL_BAD

    def test_valid_ats_url(self) -> None:
        obs = JobObservation(url="https://jobs.lever.co/company/role-id", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_quality(obs, candidate, set(), {})
        assert result is None


class TestGatesUrlDuplicate:
    def test_new_url(self) -> None:
        obs = JobObservation(url="https://example.com/job/1", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_duplicate(obs, candidate, {"abc123"}, {})
        assert result is None

    def test_duplicate_url(self) -> None:
        obs = JobObservation(url="https://example.com/job/1", source="test")
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_url_duplicate(obs, candidate, {obs.canonical_url_hash()}, {})
        assert result == RejectionReason.URL_DUPLICATE


class TestGatesTitleSeniority:
    def make_obs(self, title: str) -> JobObservation:
        return JobObservation(
            url="https://example.com/job/1", source="test", title=title, snippet=""
        )

    def test_senior_rejected(self) -> None:
        result = gate_title_seniority(
            self.make_obs("Senior Software Engineer"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.TITLE_SENIOR

    def test_staff_rejected(self) -> None:
        result = gate_title_seniority(
            self.make_obs("Staff Engineer"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.TITLE_SENIOR

    def test_manager_rejected(self) -> None:
        result = gate_title_seniority(
            self.make_obs("Engineering Manager"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.TITLE_MANAGER

    def test_product_manager_passes(self) -> None:
        result = gate_title_seniority(
            self.make_obs("Product Manager"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result is None

    def test_internship_passes(self) -> None:
        result = gate_title_seniority(
            self.make_obs("Software Engineering Intern"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result is None

    def test_newgrad_passes(self) -> None:
        result = gate_title_seniority(
            self.make_obs("New Grad Software Engineer"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result is None

    def test_junior_passes(self) -> None:
        result = gate_title_seniority(
            self.make_obs("Junior Developer"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result is None

    def test_non_tech_rejected(self) -> None:
        obs = JobObservation(
            url="https://example.com", source="test", title="Content Creator", snippet=""
        )
        result = gate_title_seniority(
            obs,
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.TITLE_NON_TECHNICAL


class TestGatesRoleFamily:
    def make_obs(self, title: str, snippet: str = "", raw: str = "") -> JobObservation:
        return JobObservation(
            url="https://example.com/job/1",
            source="test",
            title=title,
            snippet=snippet,
            raw_markdown=raw,
        )

    def test_backend(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Backend Engineer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.BACKEND

    def test_infra(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Site Reliability Engineer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.INFRA_PLATFORM

    def test_fullstack(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Fullstack Developer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.FULLSTACK_FRONTEND

    def test_swe(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Software Engineer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.GENERAL_SWE

    def test_data_engineer(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Data Engineer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.DATA_ENGINEERING

    def test_ml_engineer(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("ML Engineer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.AI_ML

    def test_devtools(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Developer Tools Engineer"), candidate, set(), {})
        assert result is None
        assert candidate.role_family == RoleFamily.DEVELOPER_TOOLS

    def test_non_tech_rejected(self) -> None:
        candidate = JobCandidate(
            canonical_id="x",
            source="test",
            direct_apply_url="",
            normalized_company="",
            normalized_role="",
            normalized_location="",
        )
        result = gate_role_family(self.make_obs("Account Manager"), candidate, set(), {})
        assert result == RejectionReason.ROLE_FAMILY_MISMATCH


class TestGatesExplicitExperience:
    def make_obs(self, raw: str = "") -> JobObservation:
        return JobObservation(
            url="https://example.com/job/1",
            source="test",
            title="Engineer",
            snippet="",
            raw_markdown=raw,
        )

    def test_5_years_passes(self) -> None:
        """5+ years should no longer be a hard reject (threshold is 7+)."""
        result = gate_explicit_experience(
            self.make_obs("Requires 5+ years of experience"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result is None

    def test_10_years(self) -> None:
        result = gate_explicit_experience(
            self.make_obs("10+ years of software engineering experience required"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.EXPERIENCE_HIGH

    def test_phd(self) -> None:
        result = gate_explicit_experience(
            self.make_obs("PhD in Computer Science required"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.EXPERIENCE_PHD

    def test_clearance(self) -> None:
        result = gate_explicit_experience(
            self.make_obs("Must have security clearance"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result == RejectionReason.CLEARANCE_REQUIRED

    def test_ok_jd(self) -> None:
        result = gate_explicit_experience(
            self.make_obs("Looking for a motivated engineer with 0-3 years experience"),
            JobCandidate(
                canonical_id="x",
                source="test",
                direct_apply_url="",
                normalized_company="",
                normalized_role="",
                normalized_location="",
            ),
            set(),
            {},
        )
        assert result is None


class TestCanonicalId:
    def test_make_id(self) -> None:
        cid = make_canonical_id("Acme Corp", "Backend Engineer", "Remote")
        assert "acmecorp" in cid
        assert "backendengineer" in cid
        assert "remote" in cid

    def test_empty_company(self) -> None:
        cid = make_canonical_id("", "Engineer", "Remote")
        assert cid.startswith("unknown:")


class TestJobObservation:
    def test_url_hash(self) -> None:
        obs1 = JobObservation(url="https://jobs.lever.co/company/role-123", source="test")
        obs2 = JobObservation(url="https://jobs.lever.co/company/role-123", source="test2")
        assert obs1.canonical_url_hash() == obs2.canonical_url_hash()

    def test_different_url_different_hash(self) -> None:
        obs1 = JobObservation(url="https://jobs.lever.co/a/1", source="test")
        obs2 = JobObservation(url="https://jobs.lever.co/a/2", source="test")
        assert obs1.canonical_url_hash() != obs2.canonical_url_hash()


class TestGitHubIndexExtractor:
    def test_simple_table(self) -> None:
        md = """# Internships

| Company | Role | Location | Link |
| --- | --- | --- | --- |
| Acme | Backend Intern | Remote | [Apply](https://jobs.lever.co/acme/123) |
| Beta | Frontend Intern | SF | [Apply](https://boards.greenhouse.io/beta/456) |
"""
        results = extract_github_index_markdown(md, "https://raw.github.com/test/README.md")
        assert len(results) == 2
        assert results[0].url == "https://jobs.lever.co/acme/123"
        assert results[1].url == "https://boards.greenhouse.io/beta/456"
        assert results[0].source.startswith("github_index:")

    def test_no_header(self) -> None:
        md = """Some random text without a table.

No jobs here at all.
"""
        results = extract_github_index_markdown(md, "https://raw.github.com/test/README.md")
        assert len(results) == 0

    def test_skip_github_links(self) -> None:
        md = """| Company | Role | Link |
| --- | --- | --- |
| Acme | Intern | https://github.com/acme/careers |
"""
        results = extract_github_index_markdown(md, "https://raw.github.com/test/README.md")
        assert len(results) == 0

    def test_skip_image_links(self) -> None:
        md = """| Company | Role | Link |
| --- | --- | --- |
| Acme | Intern | https://company.com/logo.png |
"""
        results = extract_github_index_markdown(md, "https://raw.github.com/test/README.md")
        assert len(results) == 0

    def test_extract_link_from_cell(self) -> None:
        assert _extract_link("[Apply](https://example.com/job)") == "https://example.com/job"
        assert _extract_link("https://example.com/job") == "https://example.com/job"
        assert _extract_link("no link here") == ""

    def test_alternate_columns(self) -> None:
        md = """| Name | Position | Location | Application |
| --- | --- | --- | --- |
| Acme | Backend Intern | Remote | https://jobs.lever.co/acme/1 |
"""
        results = extract_github_index_markdown(md, "https://raw.github.com/test/README.md")
        assert len(results) == 1
        assert results[0].url == "https://jobs.lever.co/acme/1"
        assert "Backend Intern" in results[0].title
        assert "Acme" in results[0].snippet

    def test_complex_table(self) -> None:
        md = """# Summer 2026 Internships

| Company | Role | Location | Notes |
| --- | --- | --- | --- |
| [Acme](https://acme.com) | SWE Intern | [Remote](https://jobs.lever.co/acme/1) | Apply ASAP |
| Beta Inc | [Data Engineering](https://boards.greenhouse.io/beta/2) | NY | |
"""
        results = extract_github_index_markdown(md, "https://raw.github.com/test/README.md")
        assert len(results) >= 1

    def test_realistic_github_index(self) -> None:
        md = (
            "# Summer2026-Internships\n\n"
            "| Company | Role | Location | Application/Link |\n"
            "| --- | --- | --- | --- |\n"
            "| **Airbnb** | SWE Intern | SF | <a href='https://airbnb.com/jobs/123'>Apply</a> |\n"
            "| Stripe | BE Intern | Remote | [Apply](https://stripe.com/jobs/456) |\n"
        )
        results = extract_github_index_markdown(
            md, "https://raw.github.com/SimplifyJobs/Summer2026-Internships/dev/README.md"
        )
        assert len(results) == 2
        urls = {r.url for r in results}
        assert "https://airbnb.com/jobs/123" in urls
        assert "https://stripe.com/jobs/456" in urls


class TestRunGates:
    @pytest.mark.asyncio
    async def test_full_pipeline_reject_senior(self) -> None:
        obs = JobObservation(
            url="https://jobs.lever.co/company/role-1",
            source="lever",
            title="Senior Engineer",
            snippet="Senior role",
        )
        result, rejections = await run_gates(obs, set(), {})
        assert result is None
        assert len(rejections) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_pass(self) -> None:
        obs = JobObservation(
            url="https://jobs.lever.co/company/role-1",
            source="lever",
            title="Software Engineering Intern",
            snippet="Summer internship at Acme Corp",
        )
        result, rejections = await run_gates(obs, set(), {})
        if result is not None:
            assert result.eligibility != EligibilityState.REJECTED
