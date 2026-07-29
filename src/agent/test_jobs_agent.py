"""Tests for JobsAgent: persistent Qdrant vector indexing and atomic jobs.md updates."""

import contextlib
import os
import tempfile

import pytest

from src.agent.jobs_agent import JobsAgent, _normalize_key, get_embedding


class TestNormalizeKey:
    def test_basic(self) -> None:
        assert _normalize_key("Acme Corp!", "Full-Stack Engineer") == "acmecorp:fullstackengineer"

    def test_case_insensitive(self) -> None:
        assert _normalize_key("AUTOACE", "SOFTWARE ENGINEER") == "autoace:softwareengineer"


@pytest.mark.asyncio
async def test_get_embedding() -> None:
    emb = await get_embedding("test job description")
    assert isinstance(emb, list)
    assert len(emb) == 1024


@pytest.mark.asyncio
async def test_jobs_agent_add_and_merge() -> None:
    test_collection = "jobs_ledger_test"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        path = f.name

    try:
        agent = JobsAgent(output_path=path, collection_name=test_collection)

        jobs1 = [
            {
                "role": "Backend Engineer",
                "company": "TechCo",
                "match_percent": 80,
                "shortlist_probability": 70,
                "location": "Remote",
                "apply_link": "https://techco.example.com/jobs/1",
            }
        ]

        merged1 = await agent.add_or_merge_jobs(jobs1)
        assert len(merged1) >= 1
        assert any(j["company"] == "TechCo" for j in merged1)

        # Merge update with higher match_percent
        jobs2 = [
            {
                "role": "Backend Engineer",
                "company": "TechCo",
                "match_percent": 95,
                "shortlist_probability": 85,
                "location": "Remote",
                "apply_link": "https://techco.example.com/jobs/1",
            }
        ]

        merged2 = await agent.add_or_merge_jobs(jobs2)
        assert len(merged2) >= 1
        matched = [j for j in merged2 if j["company"] == "TechCo"][0]
        assert matched["match_percent"] == 95

        with open(path) as f:
            content = f.read()
        assert "Backend Engineer" in content
        assert "TechCo" in content
        assert "[Apply]" in content

    finally:
        if os.path.exists(path):
            os.unlink(path)
        with contextlib.suppress(Exception):
            agent.qdrant.delete_collection(collection_name=test_collection)
