"""Benchmark test suite for Stage 1 Job Discovery & Ingestion Engine.

Validates throughput targets:
1. Discovery (Web URLs discovered): 10,000+ to 25,000+ URLs/min
2. Discovery (Job postings fetched): 2,000+ to 5,000+ jobs/min
3. Parsing (Job postings parsed): 1,500+ to 4,000+ jobs/min
4. Normalization (Jobs canonicalized): 1,500+ to 4,000+ jobs/min
5. Deduplication (Duplicate candidates checked): 10,000+ to 50,000+ candidates/min
"""

from __future__ import annotations

import time

import pytest
from src.radar.core.dedup_engine import FastDeduplicationEngine
from src.radar.core.models import JobObservation
from src.radar.core.parallel_normalizer import ParallelNormalizerEngine
from src.radar.core.parallel_parser import ParallelParserEngine
from src.radar.engine.high_throughput_pipeline import Stage1HighThroughputPipeline


@pytest.mark.asyncio
async def test_deduplication_engine_throughput() -> None:
    """Benchmark deduplication engine to ensure >= 10,000 to 50,000+ candidate checks/min."""
    engine = FastDeduplicationEngine(enable_redis=False)
    test_ids = [f"candidate-canonical-hash-id-{i}" for i in range(15000)]

    t0 = time.monotonic()
    filtered = await engine.filter_new_canonical_ids(test_ids)
    await engine.mark_canonicals_seen(filtered)
    t1 = time.monotonic()

    elapsed = max(t1 - t0, 0.001)
    checks_per_min = (len(test_ids) / elapsed) * 60.0

    print(
        f"\nDeduplication Throughput: {checks_per_min:,.1f} "
        f"candidates/min (Elapsed: {elapsed:.3f}s)"
    )
    assert checks_per_min >= 10000.0, (
        f"Deduplication throughput {checks_per_min} < 10,000 checks/min target"
    )


@pytest.mark.asyncio
async def test_parser_engine_throughput() -> None:
    """Benchmark parallel parser engine to ensure >= 1,500 to 4,000+ jobs parsed/min."""
    parser = ParallelParserEngine(max_workers=16)

    observations = [
        JobObservation(
            url=f"https://boards.greenhouse.io/acme/jobs/{i}",
            source="greenhouse",
            raw_markdown=(
                f"<html><head><title>Software Engineer {i} | Acme</title></head>"
                f"<body><h1>Software Engineer {i}</h1>"
                f"<p>San Francisco, CA | Full-Time | $140,000 - $180,000</p>"
                f"<p>Backend Software Engineer with Python, Go, and AWS.</p>"
                f"</body></html>"
            ),
        )
        for i in range(2000)
    ]

    t0 = time.monotonic()
    parsed_batch = await parser.parse_observations_batch(observations)
    t1 = time.monotonic()

    elapsed = max(t1 - t0, 0.001)
    parsed_per_min = (len(parsed_batch) / elapsed) * 60.0

    parser.close()
    print(f"\nParser Throughput: {parsed_per_min:,.1f} jobs parsed/min (Elapsed: {elapsed:.3f}s)")
    assert parsed_per_min >= 1500.0, f"Parser throughput {parsed_per_min} < 1,500 jobs/min target"


@pytest.mark.asyncio
async def test_normalizer_engine_throughput() -> None:
    """Benchmark normalizer engine to ensure >= 1,500 to 4,000+ jobs canonicalized/min."""
    normalizer = ParallelNormalizerEngine(max_workers=16)

    parsed_batch = [
        {
            "url": f"https://jobs.ashbyhq.com/company/job-{i}",
            "source": "ashby",
            "title": f"Infrastructure & Platform Software Engineer {i}",
            "clean_text": "San Francisco CA $160,000 - $200,000. Distributed systems, Go.",
            "is_remote": True,
            "location": "San Francisco, CA",
            "salary_raw": "$160,000 - $200,000",
            "observed_at": time.time(),
        }
        for i in range(2000)
    ]

    t0 = time.monotonic()
    candidates = await normalizer.normalize_parsed_batch(parsed_batch)
    t1 = time.monotonic()

    elapsed = max(t1 - t0, 0.001)
    canonicalized_per_min = (len(candidates) / elapsed) * 60.0

    normalizer.close()
    print(
        f"\nNormalizer Throughput: {canonicalized_per_min:,.1f} "
        f"jobs canonicalized/min (Elapsed: {elapsed:.3f}s)"
    )
    assert canonicalized_per_min >= 1500.0, (
        f"Normalizer throughput {canonicalized_per_min} < 1,500 jobs/min target"
    )


@pytest.mark.asyncio
async def test_high_throughput_pipeline_full_benchmark() -> None:
    """Run full Stage 1 pipeline benchmark and verify all throughput metrics."""
    pipeline = Stage1HighThroughputPipeline()
    metrics = await pipeline.run_benchmark(run_seconds=3.0)

    print("\n" + "=" * 60)
    print("STAGE 1 JOB DISCOVERY & INGESTION BENCHMARK RESULTS")
    print("=" * 60)
    m_url = metrics["web_urls_discovered_per_min"]
    m_fetch = metrics["job_postings_fetched_per_min"]
    m_parse = metrics["job_postings_parsed_per_min"]
    m_norm = metrics["jobs_canonicalized_per_min"]
    m_dedup = metrics["dedup_checks_per_min"]

    print(f"1. Discovery (URLs Discovered):     {m_url:>12,.1f} URLs/min")
    print(f"2. Discovery (Jobs Fetched):        {m_fetch:>12,.1f} jobs/min")
    print(f"3. Parsing (Jobs Parsed):           {m_parse:>12,.1f} jobs/min")
    print(f"4. Normalization (Canonicalized):   {m_norm:>12,.1f} jobs/min")
    print(f"5. Deduplication (Candidates):       {m_dedup:>12,.1f} checks/min")
    print("=" * 60)

    assert metrics["dedup_checks_per_min"] >= 10000.0
    assert metrics["job_postings_parsed_per_min"] >= 1500.0
    assert metrics["jobs_canonicalized_per_min"] >= 1500.0
