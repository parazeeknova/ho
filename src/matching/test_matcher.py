"""Tests for job matcher: async concurrent LLM scoring, batch processing."""

from unittest.mock import MagicMock

import pytest

from src.matching.matcher import batch_match


class TestBatchMatch:
    @pytest.mark.asyncio
    async def test_empty_jobs(self, mocker) -> None:
        ctx = MagicMock()
        rag = MagicMock()
        rag.retrieve.return_value = [("skills_0", "python django", 0.9)]
        result = await batch_match([], rag, ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_below_threshold(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.json_chat.return_value = {
            "role": "Dev",
            "company": "Co",
            "match_percent": "30",
            "shortlist_probability": "20",
        }
        ctx.chat = MagicMock()

        rag = MagicMock()
        rag.retrieve.return_value = [("s", "python", 0.8)]

        jobs = [{"markdown": "some job desc", "url": "http://x.com", "title": "Dev"}]
        result = await batch_match(jobs, rag, ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_keeps_above_threshold(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.json_chat.return_value = {
            "role": "SDE",
            "company": "FAANG",
            "match_percent": "85",
            "shortlist_probability": "75",
        }
        ctx.chat = MagicMock()

        rag = MagicMock()
        rag.retrieve.return_value = [("s", "python react", 0.9)]

        jobs = [{"markdown": "x" * 100, "url": "http://x.com", "title": "SDE"}]
        result = await batch_match(jobs, rag, ctx)
        assert len(result) == 1
        assert result[0]["match_percent"] == 85

    @pytest.mark.asyncio
    async def test_sorts_by_match_percent(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.chat = MagicMock()
        responses = [
            {
                "role": "A",
                "company": "A",
                "match_percent": "50",
                "shortlist_probability": "40",
            },
            {
                "role": "B",
                "company": "B",
                "match_percent": "90",
                "shortlist_probability": "80",
            },
            {
                "role": "C",
                "company": "C",
                "match_percent": "70",
                "shortlist_probability": "60",
            },
        ]
        ctx.json_chat.side_effect = responses

        rag = MagicMock()
        rag.retrieve.return_value = [("s", "skills", 0.8)]

        jobs = [
            {"markdown": "a" * 100, "url": "u1", "title": "t1"},
            {"markdown": "b" * 100, "url": "u2", "title": "t2"},
            {"markdown": "c" * 100, "url": "u3", "title": "t3"},
        ]
        result = await batch_match(jobs, rag, ctx)
        assert result[0]["match_percent"] == 90
        assert result[1]["match_percent"] == 70
        assert result[2]["match_percent"] == 50

    @pytest.mark.asyncio
    async def test_skips_short_jds(self, mocker) -> None:
        ctx = MagicMock()
        rag = MagicMock()
        rag.retrieve.return_value = [("s", "skills", 0.8)]
        jobs = [{"markdown": "short", "url": "u", "title": "t"}]
        result = await batch_match(jobs, rag, ctx)
        assert result == []

    @pytest.mark.asyncio
    async def test_concurrent_matching(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.chat = MagicMock()
        ctx.json_chat.return_value = {
            "role": "Dev",
            "company": "Co",
            "match_percent": "75",
            "shortlist_probability": "65",
        }

        rag = MagicMock()
        rag.retrieve.return_value = [("s", "skills here", 0.9)]

        jobs = [
            {"markdown": f"job desc {i}" * 10 + " " * 30, "url": f"u{i}", "title": f"t{i}"}
            for i in range(10)
        ]
        result = await batch_match(jobs, rag, ctx, concurrency=4)
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_handles_match_exceptions(self, mocker) -> None:
        ctx = MagicMock()
        ctx.maybe_flush = MagicMock()
        ctx.chat = MagicMock()
        ctx.json_chat.side_effect = [
            RuntimeError("boom"),
            {
                "role": "Good",
                "company": "Co",
                "match_percent": "80",
                "shortlist_probability": "70",
            },
        ]

        rag = MagicMock()
        rag.retrieve.return_value = [("s", "skills", 0.8)]

        jobs = [
            {"markdown": "bad" * 40, "url": "u1", "title": "bad"},
            {"markdown": "good" * 40, "url": "u2", "title": "good"},
        ]
        result = await batch_match(jobs, rag, ctx)
        assert len(result) == 1
        assert result[0]["role"] == "Good"
