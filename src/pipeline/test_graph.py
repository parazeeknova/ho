"""Tests for pipeline graph helpers, prompts, and state definitions."""

from __future__ import annotations

from typing import Any, cast

from src.pipeline.graph import (
    CRITIC_PROMPT,
    MATCHER_PROMPT,
    GraphState,
    _apply_hard_constraints,
)


class TestHardConstraints:
    def test_junior_role_passes(self) -> None:
        match = {
            "role": "Junior Software Engineer",
            "company": "Acme Corp",
            "match_percent": 85,
            "shortlist_probability": 70,
            "matching_skills": ["Python"],
            "missing_skills": [],
            "jd_summary": "Great entry-level role for recent grads.",
            "verdict": "STRONG_MATCH",
        }
        review = _apply_hard_constraints(match)
        assert review.passed is True
        assert "Pre-checks passed" in review.critique_reason

    def test_senior_role_fails(self) -> None:
        match = {
            "role": "Senior Software Engineer",
            "company": "Acme Corp",
            "match_percent": 85,
            "shortlist_probability": 70,
            "matching_skills": ["Python"],
            "missing_skills": [],
            "jd_summary": "Looking for an experienced engineer.",
            "verdict": "STRONG_MATCH",
        }
        review = _apply_hard_constraints(match)
        assert review.passed is False
        assert "senior" in review.critique_reason.lower()

    def test_five_plus_years_in_summary_fails(self) -> None:
        match = {
            "role": "Software Engineer",
            "company": "Acme Corp",
            "match_percent": 85,
            "shortlist_probability": 70,
            "matching_skills": ["Python"],
            "missing_skills": [],
            "jd_summary": "Must have 5+ years of experience building web apps.",
            "verdict": "STRONG_MATCH",
        }
        review = _apply_hard_constraints(match)
        assert review.passed is False
        assert "hard-constraint" in review.critique_reason.lower()


class TestGraphState:
    def test_accepts_skip_match_critique(self) -> None:
        state: GraphState = {
            "markdown": "some jd text",
            "url": "https://example.com/job/1",
            "title": "Test Job",
            "skip": True,
            "match": None,
            "critique": None,
            "retries": 0,
        }
        assert state["skip"] is True
        assert state["match"] is None
        assert state["critique"] is None

    def test_accepts_match_dict(self) -> None:
        state: GraphState = {
            "markdown": "jd",
            "url": "https://example.com/job/2",
            "title": "",
            "skip": False,
            "match": {"role": "Dev", "match_percent": 90},
            "critique": {"passed": True},
            "retries": 0,
        }
        match = cast(dict[str, Any], state["match"])
        critique = cast(dict[str, Any], state["critique"])
        assert match["role"] == "Dev"
        assert critique["passed"] is True


class TestPrompts:
    def test_matcher_prompt_keywords(self) -> None:
        assert "job-resume matching" in MATCHER_PROMPT
        assert "relevant resume snippets" in MATCHER_PROMPT.lower()
        assert "{job_description}" in MATCHER_PROMPT
        assert "NO_MATCH" in MATCHER_PROMPT

    def test_critic_prompt_keywords(self) -> None:
        assert "strict job-match auditor" in CRITIC_PROMPT
        assert "hard constraints" in CRITIC_PROMPT.lower()
        assert "{result}" in CRITIC_PROMPT
        assert "passed" in CRITIC_PROMPT
