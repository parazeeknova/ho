"""Unit tests for CleanupAgent."""

from src.agent.cleanup_agent import CleanupAgent


def test_is_valid_undergrad_role() -> None:
    agent = CleanupAgent()

    # Valid undergrad backend role
    valid_job = {
        "role": "Backend Engineer Intern",
        "company": "Stripe",
        "jd_summary": "Looking for CS undergrad intern",
        "verdict": "STRONG_MATCH",
        "match_percent": 85,
    }
    assert agent.is_valid_undergrad_role(valid_job) is True

    # Reject PhD role
    phd_job = {
        "role": "Software PhD Internships",
        "company": "Apple",
        "jd_summary": "PhD candidate required",
        "verdict": "MATCH",
        "match_percent": 70,
    }
    assert agent.is_valid_undergrad_role(phd_job) is False

    # Reject Senior role
    senior_job = {
        "role": "Senior Staff Engineer",
        "company": "Google",
        "jd_summary": "10+ years experience",
        "verdict": "MATCH",
        "match_percent": 60,
    }
    assert agent.is_valid_undergrad_role(senior_job) is False

    # Reject N/A directory entry
    na_job = {
        "role": "N/A",
        "company": "N/A",
        "verdict": "NO_MATCH",
        "match_percent": 0,
    }
    assert agent.is_valid_undergrad_role(na_job) is False
