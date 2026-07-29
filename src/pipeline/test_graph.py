"""Tests for regex-based hard constraints and JD truncation in graph.py."""

from src.pipeline.graph import _apply_hard_constraints


def test_ok_entry_level() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Software Engineer Intern",
            "jd_summary": "Build backend services in Python and Go.",
        }
    )
    assert r.passed
    assert r.critique_reason == "Pre-checks passed"


def test_reports_to_manager_not_filtered() -> None:
    """'reports to the Engineering Manager' in JD must NOT filter the role."""
    r = _apply_hard_constraints(
        {
            "role": "Backend Engineer",
            "jd_summary": "Reports to the Engineering Manager. Works with CI/CD.",
        }
    )
    assert r.passed


def test_senior_in_role_title_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Senior Software Engineer",
            "jd_summary": "Lead the platform team.",
        }
    )
    assert not r.passed


def test_staff_in_role_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Staff Engineer",
            "jd_summary": "",
        }
    )
    assert not r.passed


def test_manager_in_role_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Engineering Manager",
            "jd_summary": "",
        }
    )
    assert not r.passed


def test_director_in_role_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Director of Engineering",
            "jd_summary": "",
        }
    )
    assert not r.passed


def test_principal_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Principal Architect",
            "jd_summary": "",
        }
    )
    assert not r.passed


def test_lead_not_in_role_ok() -> None:
    """Lead by itself (not 'lead engineer') in JD should NOT filter."""
    r = _apply_hard_constraints(
        {
            "role": "Fullstack Developer",
            "jd_summary": "Will lead code reviews and mentor juniors.",
        }
    )
    assert r.passed


def test_phd_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Research Scientist",
            "jd_summary": "Requires PhD in Computer Science.",
        }
    )
    assert not r.passed


def test_five_years_exp_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Backend Engineer",
            "jd_summary": "Must have 5+ years of professional experience.",
        }
    )
    assert not r.passed


def test_ten_years_exp_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Backend Engineer",
            "jd_summary": "10+ years required.",
        }
    )
    assert not r.passed


def test_sales_role_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Sales Executive",
            "jd_summary": "B2B sales role.",
        }
    )
    assert not r.passed


def test_marketing_role_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Marketing Intern",
            "jd_summary": "",
        }
    )
    assert not r.passed


def test_sr_abbrev_rejected() -> None:
    r = _apply_hard_constraints(
        {
            "role": "Sr. Data Engineer",
            "jd_summary": "",
        }
    )
    assert not r.passed


def test_vp_rejected() -> None:
    r = _apply_hard_constraints({"role": "VP of Engineering", "jd_summary": ""})
    assert not r.passed
