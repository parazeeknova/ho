"""Unit tests for Centralized Board Registry (src/radar/board_registry.py)."""

import pytest

from src.radar.board_registry import (
    REGISTERED_BOARDS,
    get_all_registered_boards,
    get_discovery_index_sources,
)


def test_board_registry_structure():
    boards = get_all_registered_boards()
    assert len(boards) > 100, f"Expected >100 registered boards, got {len(boards)}"

    seen_ids = set()
    for sid, url, source_type in boards:
        assert sid, "source_id cannot be empty"
        assert sid not in seen_ids, f"Duplicate source_id found: {sid}"
        seen_ids.add(sid)

        assert url.startswith("http://") or url.startswith("https://"), f"Invalid URL: {url}"
        assert source_type in {"official_ats", "discovery_index"}, f"Invalid type: {source_type}"


def test_discovery_index_sources():
    indexes = get_discovery_index_sources()
    assert "google:careers" in indexes
    assert "microsoft:careers" in indexes
    assert "ycombinator:jobs" in indexes


@pytest.mark.asyncio
async def test_live_verify_sample_boards():
    # Live HTTP test of sample high-priority boards
    sample = [
        b
        for b in REGISTERED_BOARDS
        if b[0]
        in {
            "google:careers",
            "stripe:greenhouse",
            "anthropic:ashby",
            "swiggy:ashby",
            "razorpay:ashby",
        }
    ]
    assert len(sample) == 5

    import httpx

    headers = {"User-Agent": "Mozilla/5.0"}
    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        for sid, url, _ in sample:
            resp = await client.get(url, follow_redirects=True)
            assert resp.status_code == 200, (
                f"Failed verification for {sid} ({url}): HTTP {resp.status_code}"
            )
