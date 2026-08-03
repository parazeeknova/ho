"""Unit tests for the 3 Radar Architecture Layers:

1. Layer 1: ATS API Interceptor (Greenhouse, Lever, Ashby, Workable, SmartRecruiters)
2. Layer 2: Open-Source GitHub Index ETag Poller (304 Not Modified & ETag conditional fetching)
3. Layer 3: Time-Restricted Search Dorking (48h SearXNG time_range filtering)
"""

from __future__ import annotations

import pytest
from src.radar.sources.ats_interceptor import intercept_ats_board, parse_ats_slug
from src.radar.sources.dorking import DorkingEngine
from src.radar.sources.github_poller import poll_github_index_etag


class TestLayer1ATSInterceptor:
    def test_parse_ats_slug_greenhouse(self) -> None:
        parsed = parse_ats_slug("https://boards.greenhouse.io/stripe/jobs/12345")
        assert parsed == ("greenhouse", "stripe")

    def test_parse_ats_slug_lever(self) -> None:
        parsed = parse_ats_slug("https://jobs.lever.co/palantir/5678-uuid")
        assert parsed == ("lever", "palantir")

    def test_parse_ats_slug_ashby(self) -> None:
        parsed = parse_ats_slug("https://jobs.ashbyhq.com/notion/9012-uuid")
        assert parsed == ("ashby", "notion")

    def test_parse_ats_slug_workable(self) -> None:
        parsed = parse_ats_slug("https://apply.workable.com/razorpay/j/3456/")
        assert parsed == ("workable", "razorpay")

    def test_parse_ats_slug_smartrecruiters(self) -> None:
        parsed = parse_ats_slug("https://jobs.smartrecruiters.com/SquarePoint/7890")
        assert parsed == ("smartrecruiters", "squarepoint")

    def test_parse_ats_slug_invalid(self) -> None:
        parsed = parse_ats_slug("https://example.com/careers")
        assert parsed is None

    @pytest.mark.asyncio
    async def test_intercept_ats_board_fallback_non_ats(self) -> None:
        result = await intercept_ats_board("https://example.com/careers", "example:careers")
        assert result is None


class TestLayer2GitHubETagPoller:
    @pytest.mark.asyncio
    async def test_poll_github_index_etag_execution(self) -> None:
        url = "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md"
        obs, modified = await poll_github_index_etag(url)
        assert isinstance(obs, list)
        assert isinstance(modified, bool)


class TestLayer3TimeRestrictedDorking:
    @pytest.mark.asyncio
    async def test_dorking_engine_time_restriction(self) -> None:
        engine = DorkingEngine(searxng_url="http://localhost:8080")
        # Run 1 sample dork query with time_range="day"
        obs = await engine.execute_dorks(
            queries=['site:boards.greenhouse.io intitle:"intern" OR intitle:"new grad" "2026"'],
            time_range="day",
        )
        assert isinstance(obs, list)
