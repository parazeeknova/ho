"""Tests for Pydantic schemas and canonicalization utilities."""

from __future__ import annotations

import pytest

from src.llm.schemas import CriticReview, JobMatch, canonicalize_markdown


class TestJobMatch:
    def test_valid_data(self) -> None:
        data = {
            "role": "Software Engineer",
            "company": "Acme Corp",
            "match_percent": 85,
            "shortlist_probability": 70,
            "matching_skills": ["Python", "React"],
            "missing_skills": ["Rust"],
            "jd_summary": "Exciting startup role",
            "salary": "80K INR/month",
            "location": "Bangalore",
            "is_remote": True,
            "verdict": "STRONG_MATCH",
        }
        match = JobMatch.model_validate(data)
        assert match.role == "Software Engineer"
        assert match.match_percent == 85
        assert match.verdict == "STRONG_MATCH"

    def test_invalid_verdict_clamps_to_no_match(self) -> None:
        data = {
            "role": "Dev",
            "company": "Inc",
            "match_percent": 50,
            "shortlist_probability": 50,
            "verdict": "BOGUS_VERDICT",
        }
        match = JobMatch.model_validate(data)
        assert match.verdict == "NO_MATCH"

    def test_match_percent_over_100_fails(self) -> None:
        data = {
            "role": "Dev",
            "company": "Inc",
            "match_percent": 150,
            "shortlist_probability": 50,
        }
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            JobMatch.model_validate(data)


class TestCriticReview:
    def test_valid_data(self) -> None:
        data = {
            "passed": False,
            "critique_reason": "Role is senior-level — hard-constraint violation.",
            "requires_rescore": False,
        }
        review = CriticReview.model_validate(data)
        assert review.passed is False
        assert "senior" in review.critique_reason
        assert review.requires_rescore is False


class TestCanonicalizeMarkdown:
    def test_strips_html_tags(self) -> None:
        raw = "".join(
            [
                "<h1>Job Title</h1>",
                "<p>A helpful description of the role " * 3,
                "with various responsibilities and requirements that make it ",
                "at least forty characters long.</p>",
            ]
        )
        result = canonicalize_markdown(raw)
        assert result is not None
        assert "<h1>" not in result
        assert "<p>" not in result
        assert "Job Title" in result

    def test_drops_internshala_urls(self) -> None:
        raw = "A " + "very " * 20 + "long job posting from internshala."
        result = canonicalize_markdown(raw, url="https://internshala.com/jobs/some-cool-job")
        assert result is None

    def test_short_input_returns_none(self) -> None:
        assert canonicalize_markdown("too short") is None
        assert canonicalize_markdown("") is None

    def test_generic_jobs_path_returns_none(self) -> None:
        raw = "A " + "very " * 20 + "long listing on a generic jobs page."
        result = canonicalize_markdown(raw, url="https://example.com/jobs")
        assert result is None

    def test_valid_job_posting_preserved(self) -> None:
        raw = "A " + "specific " * 20 + "job description for a Python developer."
        result = canonicalize_markdown(
            raw,
            url="https://example.com/careers/python-dev-2024",
        )
        assert result is not None
        assert "Python developer" in result
