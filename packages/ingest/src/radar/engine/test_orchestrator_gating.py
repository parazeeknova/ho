"""Regression tests for high-volume sweep gate partitioning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.radar.core.models import JobObservation
from src.radar.engine.orchestrator import _enrich_graph_features, _partition_known_observations


def test_partition_deduplicates_known_postings_before_gating() -> None:
    known_url = "https://boards.greenhouse.io/acme/jobs/known"
    new_url = "https://boards.greenhouse.io/acme/jobs/new"
    known = JobObservation(url=known_url, source="greenhouse", raw_markdown="short")
    richer_known = JobObservation(url=known_url, source="greenhouse", raw_markdown="richer text")
    new = JobObservation(url=new_url, source="greenhouse", raw_markdown="new job")

    unseen, refresh_ids, known_count = _partition_known_observations(
        [known, richer_known, new],
        {known.canonical_url_hash()},
        {known.canonical_url_hash(): 0.0},
        now=7 * 3600,
    )

    assert known_count == 1
    assert refresh_ids == [known.canonical_url_hash()]
    assert unseen == [new]


async def test_graph_features_are_calculated_once_per_company() -> None:
    graph = SimpleNamespace(
        get_node=AsyncMock(return_value=object()),
        get_graph_metrics_for_node=AsyncMock(return_value={"pagerank": 0.2}),
        predict_hiring_likelihood=AsyncMock(return_value={"score": 0.7}),
        get_local_graph=AsyncMock(
            return_value={"nodes": [{"node_type": "technology", "data": {"name": "Python"}}]}
        ),
    )
    candidates = [
        SimpleNamespace(normalized_company="Canonical", canonical_id="canonical:one"),
        SimpleNamespace(normalized_company="Canonical", canonical_id="canonical:two"),
    ]

    features = await _enrich_graph_features(candidates, graph)

    assert set(features) == {"canonical:one", "canonical:two"}
    assert graph.get_node.await_count == 1
    assert graph.get_graph_metrics_for_node.await_count == 1
    assert graph.predict_hiring_likelihood.await_count == 1
    assert graph.get_local_graph.await_count == 1
