"""Unit tests for Common Crawl discovery (slug parsing, platform mapping)."""

from __future__ import annotations

from src.radar.sources.common_crawl import (
    _ats_marker_for_url,
    _slug_from_url,
)


def test_ats_marker_detection():
    assert (
        _ats_marker_for_url("https://boards.greenhouse.io/0x/jobs/123") == "boards.greenhouse.io/"
    )
    assert _ats_marker_for_url("https://jobs.lever.co/exampleco") == "jobs.lever.co/"
    assert _ats_marker_for_url("https://jobs.ashbyhq.com/someco") == "jobs.ashbyhq.com/"
    assert _ats_marker_for_url("https://apply.workable.com/co") == "apply.workable.com/"
    assert _ats_marker_for_url("https://example.com/careers") is None


def test_slug_extraction_cleans_query_and_fragment():
    assert (
        _slug_from_url("https://boards.greenhouse.io/1047games?gh_src=abc", "boards.greenhouse.io/")
        == "1047games"
    )
    assert (
        _slug_from_url("https://boards.greenhouse.io/10xgenomics/jobs/5", "boards.greenhouse.io/")
        == "10xgenomics"
    )
    assert _slug_from_url("https://jobs.lever.co/acme#apply", "jobs.lever.co/") == "acme"
    assert _slug_from_url("https://boards.greenhouse.io/", "boards.greenhouse.io/") is None
    assert _slug_from_url("https://boards.greenhouse.io/123456", "boards.greenhouse.io/") is None
