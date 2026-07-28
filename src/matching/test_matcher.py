"""Tests for job matcher: LLM scoring, batch processing."""

from unittest.mock import MagicMock

from src.matching.matcher import batch_match, match_job


class TestMatchJob:
    def test_returns_dict_on_valid_json(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.json_chat.return_value = {
            "role": "SDE Intern",
            "company": "Google",
            "match_percent": "85",
            "shortlist_probability": "70",
            "matching_skills": ["Python"],
            "missing_skills": ["Go"],
            "jd_summary": "Build APIs",
            "salary": None,
            "posted_date": "2026-07-20",
            "apply_link": "https://apply.example.com",
            "is_undergrad_friendly": True,
            "is_remote": True,
            "location": "Remote",
            "verdict": "STRONG_MATCH",
        }

        result = match_job("some job description", "relevant chunks", ctx)
        assert result is not None
        assert result["match_percent"] == 85
        assert result["shortlist_probability"] == 70
        assert result["role"] == "SDE Intern"
        assert result["company"] == "Google"
        assert result["verdict"] == "STRONG_MATCH"

    def test_returns_none_on_invalid_json(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.json_chat.return_value = {"not": "a match"}

        result = match_job("jd", "chunks", ctx)
        assert result is None

    def test_returns_none_on_non_dict(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.json_chat.return_value = ["list", "not", "dict"]

        result = match_job("jd", "chunks", ctx)
        assert result is None

    def test_calls_maybe_flush(self) -> None:
        ctx = MagicMock()
        ctx.json_chat.return_value = {
            "role": "x",
            "company": "y",
            "match_percent": "50",
            "shortlist_probability": "50",
        }
        match_job("jd", "chunks", ctx)
        ctx.maybe_flush.assert_called_once()


class TestBatchMatch:
    def test_empty_jobs(self) -> None:
        ctx = MagicMock()
        rag = MagicMock()
        rag.retrieve.return_value = [("skills_0", "python django", 0.9)]
        result = batch_match([], rag, ctx)
        assert result == []

    def test_filters_below_threshold(self) -> None:
        ctx = MagicMock()
        ctx.json_chat.return_value = {
            "role": "Dev",
            "company": "Co",
            "match_percent": "30",
            "shortlist_probability": "20",
        }
        rag = MagicMock()
        rag.retrieve.return_value = [("s", "python", 0.8)]

        jobs = [{"markdown": "some job desc", "url": "http://x.com", "title": "Dev"}]
        result = batch_match(jobs, rag, ctx)
        assert result == []  # 30% < 40% threshold

    def test_keeps_above_threshold(self) -> None:
        ctx = MagicMock()
        ctx.json_chat.return_value = {
            "role": "SDE",
            "company": "FAANG",
            "match_percent": "85",
            "shortlist_probability": "75",
        }
        rag = MagicMock()
        rag.retrieve.return_value = [("s", "python react", 0.9)]

        jobs = [{"markdown": "x" * 100, "url": "http://x.com", "title": "SDE"}]
        result = batch_match(jobs, rag, ctx)
        assert len(result) == 1
        assert result[0]["match_percent"] == 85

    def test_sorts_by_match_percent(self) -> None:
        ctx = MagicMock()
        responses = [
            {"role": "A", "company": "A", "match_percent": "50", "shortlist_probability": "40"},
            {"role": "B", "company": "B", "match_percent": "90", "shortlist_probability": "80"},
            {"role": "C", "company": "C", "match_percent": "70", "shortlist_probability": "60"},
        ]
        ctx.json_chat.side_effect = responses
        rag = MagicMock()
        rag.retrieve.return_value = [("s", "skills", 0.8)]

        jobs = [
            {"markdown": "a" * 100, "url": "u1", "title": "t1"},
            {"markdown": "b" * 100, "url": "u2", "title": "t2"},
            {"markdown": "c" * 100, "url": "u3", "title": "t3"},
        ]
        result = batch_match(jobs, rag, ctx)
        assert result[0]["match_percent"] == 90
        assert result[1]["match_percent"] == 70
        assert result[2]["match_percent"] == 50

    def test_skips_short_jds(self) -> None:
        ctx = MagicMock()
        rag = MagicMock()
        rag.retrieve.return_value = [("s", "skills", 0.8)]
        jobs = [{"markdown": "short", "url": "u", "title": "t"}]
        result = batch_match(jobs, rag, ctx)
        assert result == []  # too short, skipped
