"""Tests for JobsAgent backed by pgvector MemoryStore."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock

import pytest
from src.agent.jobs_agent import JobsAgent, _normalize_key


class TestNormalizeKey:
    def test_basic(self) -> None:
        assert (
            _normalize_key("Acme Corp!", "Full-Stack Engineer")
            == "acmecorp:fullstackengineer:remote"
        )

    def test_case_insensitive(self) -> None:
        assert _normalize_key("AUTOACE", "SOFTWARE ENGINEER") == "autoace:softwareengineer:remote"


@pytest.mark.asyncio
async def test_add_or_merge_empty_input() -> None:
    store = AsyncMock()
    agent = JobsAgent(store=store)
    result = await agent.add_or_merge_jobs([])
    assert result == []
    store.upsert_job_ledger.assert_not_called()


@pytest.mark.asyncio
async def test_add_or_merge_zero_jobs_no_store() -> None:
    agent = JobsAgent()
    result = await agent.add_or_merge_jobs([{"role": "X", "company": "Y"}])
    assert result == []


@pytest.mark.asyncio
async def test_add_or_merge_calls_store_and_writes_md() -> None:
    store = AsyncMock()
    store.upsert_job_ledger.return_value = 1
    store.get_all_jobs_ledger.return_value = [
        {
            "role": "Backend Engineer",
            "company": "TechCo",
            "match_percent": 80,
            "verdict": "GOOD_MATCH",
            "apply_link": "https://example.com/jobs/1",
            "shortlist_probability": 70,
            "location": "Remote",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        path = f.name
    try:
        agent = JobsAgent(output_path=path, store=store)
        jobs = [
            {
                "role": "Backend Engineer",
                "company": "TechCo",
                "match_percent": 80,
                "shortlist_probability": 70,
                "location": "Remote",
                "apply_link": "https://example.com/jobs/1",
            }
        ]
        result = await agent.add_or_merge_jobs(jobs)
        assert len(result) == 1
        assert result[0]["role"] == "Backend Engineer"
        store.upsert_job_ledger.assert_called_once()
    finally:
        if os.path.exists(path):
            os.unlink(path)
