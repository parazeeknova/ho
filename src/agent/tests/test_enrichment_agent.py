"""Tests for enrichment rescoring against pgvector resume chunks."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.enrichment_agent import EnrichmentAgent


@pytest.mark.asyncio
async def test_enrich_no_store_rescore_skipped() -> None:
    store = AsyncMock()
    store.search_similar_chunks.return_value = []
    agent = EnrichmentAgent(store)
    job = {"role": "BE", "company": "Co", "match_percent": 50}
    result = await agent.enrich_and_rescore(job)
    assert result["match_percent"] == 50  # unchanged


@pytest.mark.asyncio
async def test_enrich_with_chunks_rescores() -> None:
    store = AsyncMock()
    store.search_similar_chunks.return_value = [
        {"embedding": [0.5] * 1024, "section": "skills", "content": "Python"},
    ]
    with patch("src.agent.enrichment_agent._get_embedding") as mock_embed:
        mock_embed.return_value = [0.5] * 1024
        agent = EnrichmentAgent(store)
        job = {"role": "BE", "company": "Co", "match_percent": 40, "jd_summary": "Test"}
        result = await agent.enrich_and_rescore(job)
        # Similar vectors should give high score
        assert result["match_percent"] >= 40


@pytest.mark.asyncio
async def test_batch_enrich_empty() -> None:
    agent = EnrichmentAgent(AsyncMock())
    result = await agent.batch_enrich_and_rescore([], concurrency=2)
    assert result == []
