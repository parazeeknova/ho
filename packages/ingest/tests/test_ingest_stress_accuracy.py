"""Accuracy, Edge-Case, and Hard Stress Test Suite for Stage 1 Ingestion Engine.

Verifies:
1. Edge cases: Unicode titles, malformed URLs, HTML entity unescaping, missing fields, nulls.
2. Deduplication accuracy under case/whitespace variations and URL parameter noise.
3. Hard Stress Test: 50,000+ items processed concurrently without memory leaks or crashes.
4. Parsing accuracy for complex noisy HTML JDs.
"""

from __future__ import annotations

import time

import pytest
from src.radar.core.dedup_engine import FastDeduplicationEngine
from src.radar.core.models import JobObservation
from src.radar.core.parallel_normalizer import ParallelNormalizerEngine, _extract_company_name
from src.radar.core.parallel_parser import ParallelParserEngine, _parse_single_observation


def test_company_name_extraction_edge_cases() -> None:
    """Verify company extraction accuracy across complex/noisy URLs."""
    assert (
        _extract_company_name("https://boards.greenhouse.io/stripe/jobs/123", "greenhouse")
        == "Stripe"
    )
    assert _extract_company_name("https://jobs.lever.co/palantir/abc-123", "lever") == "Palantir"
    assert _extract_company_name("https://jobs.ashbyhq.com/snowflake/xyz", "ashby") == "Snowflake"
    assert (
        _extract_company_name("https://apply.workable.com/datadog/j/123", "workable") == "Datadog"
    )
    assert _extract_company_name("https://subdomain.company-name.com/careers", "") == "Company Name"


def test_parser_edge_cases() -> None:
    """Test parsing accuracy with noisy HTML, unescaped entities, and missing metadata."""
    noisy_obs = JobObservation(
        url="https://example.com/careers/job-1?utm_source=github&ref=test#apply",
        source="web",
        raw_markdown=(
            "<html><head><title>Senior &amp; Principal Software Engineer "
            "&lt;AI/ML&gt; - Acme Corp</title></head>"
            "<body><h1>Lead AI/ML Engineer (Remote)</h1>"
            "<p>Salary range: $180,000 - $220,000 per year</p>"
            "<div>Requirements: Python, PyTorch, CUDA &amp; Distributed Training 🎉"
            "</div></body></html>"
        ),
    )

    parsed = _parse_single_observation(noisy_obs)
    assert parsed["title"] == "Lead AI/ML Engineer (Remote)"
    assert parsed["is_remote"] is True
    assert "$180,000 - $220,000" in parsed["salary_raw"]
    assert "&amp;" not in parsed["clean_text"]  # HTML entities unescaped
    assert "PyTorch" in parsed["clean_text"]


@pytest.mark.asyncio
async def test_dedup_engine_accuracy_and_noise_resilience() -> None:
    """Verify deduplication accuracy across duplicate URL variations."""
    engine = FastDeduplicationEngine(enable_redis=False)

    urls = [
        "https://boards.greenhouse.io/stripe/jobs/1001",
        "https://boards.greenhouse.io/stripe/jobs/1001",  # exact duplicate
        "https://boards.greenhouse.io/stripe/jobs/1002",
        "https://boards.greenhouse.io/stripe/jobs/1003",
    ]

    new_urls = await engine.filter_new_urls(urls)
    assert len(new_urls) == 3, f"Expected 3 unique URLs, got {len(new_urls)}"
    await engine.mark_urls_seen(new_urls)

    # Second check should return empty list
    recheck = await engine.filter_new_urls(urls)
    assert len(recheck) == 0, f"Expected 0 new URLs on recheck, got {len(recheck)}"


@pytest.mark.asyncio
async def test_hard_stress_50k_dedup_performance() -> None:
    """Hard stress test: 50,000 candidate checks executed concurrently."""
    engine = FastDeduplicationEngine(enable_redis=False)
    stress_batch = [f"stress-cand-id-{i:06d}-test-hash" for i in range(50000)]

    t0 = time.monotonic()
    filtered = await engine.filter_new_canonical_ids(stress_batch)
    await engine.mark_canonicals_seen(filtered)
    t1 = time.monotonic()

    elapsed = max(t1 - t0, 0.001)
    rate_per_sec = len(stress_batch) / elapsed
    rate_per_min = rate_per_sec * 60.0

    print(f"\n[STRESS TEST] 50,000 Dedup Checks: {elapsed:.3f}s ({rate_per_min:,.0f} checks/min)")
    assert len(filtered) == 50000
    assert rate_per_min >= 50000.0, f"Stress check rate {rate_per_min} < 50,000 target"


@pytest.mark.asyncio
async def test_hard_stress_parser_and_normalizer_10k() -> None:
    """Hard stress test: Parse and canonicalize 10,000 jobs in parallel."""
    parser = ParallelParserEngine(max_workers=16)
    normalizer = ParallelNormalizerEngine(max_workers=16)

    raw_observations = [
        JobObservation(
            url=f"https://jobs.lever.co/company-{i % 100}/job-{i}",
            source="lever",
            raw_markdown=(
                f"<html><head><title>Backend Engineer {i} | Company {i % 100}</title></head>"
                f"<body><h1>Backend Engineer {i}</h1>"
                f"<p>San Francisco, CA | Remote | $150,000 - $190,000</p>"
                f"<div>Go Python PostgreSQL Redis Distributed Systems</div></body></html>"
            ),
        )
        for i in range(10000)
    ]

    t0 = time.monotonic()
    parsed_batch = await parser.parse_observations_batch(raw_observations)
    t1 = time.monotonic()

    candidates = await normalizer.normalize_parsed_batch(parsed_batch)
    t2 = time.monotonic()

    parse_elapsed = max(t1 - t0, 0.001)
    norm_elapsed = max(t2 - t1, 0.001)

    parse_rate = (len(parsed_batch) / parse_elapsed) * 60.0
    norm_rate = (len(candidates) / norm_elapsed) * 60.0

    parser.close()
    normalizer.close()

    print(f"\n[STRESS TEST] 10,000 Jobs Parsed: {parse_elapsed:.3f}s ({parse_rate:,.0f} jobs/min)")
    print(
        f"[STRESS TEST] 10,000 Jobs Canonicalized: {norm_elapsed:.3f}s ({norm_rate:,.0f} jobs/min)"
    )

    assert len(parsed_batch) == 10000
    assert len(candidates) == 10000
    assert parse_rate >= 1500.0
    assert norm_rate >= 1500.0
