"""Regression tests for bounded, API-first board polling."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from src.radar.core.models import JobObservation
from src.radar.engine import orchestrator
from src.radar.sources import ats_interceptor, sources


@pytest.fixture(autouse=True)
def _clean_source_state() -> Generator[None]:
    sources._SOURCE_CHECKPOINTS.clear()
    sources._LAST_SNAPSHOT_URLS.clear()
    yield
    sources._SOURCE_CHECKPOINTS.clear()
    sources._LAST_SNAPSHOT_URLS.clear()


@pytest.mark.asyncio
async def test_direct_ats_poll_returns_only_snapshot_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _direct_api(_url: str, _source_id: str) -> list[JobObservation]:
        return [
            JobObservation(
                url="https://boards.greenhouse.io/test/jobs/1",
                source="test:greenhouse",
                title="Software Engineer",
                raw_markdown="job description",
            )
        ]

    monkeypatch.setattr(ats_interceptor, "intercept_ats_board", _direct_api)
    board = {
        "id": "test:greenhouse",
        "url": "https://boards.greenhouse.io/test",
        "source_type": "official_ats",
    }

    assert len(await orchestrator._poll_board(board)) == 1
    assert await orchestrator._poll_board(board) == []


@pytest.mark.asyncio
async def test_unavailable_official_ats_does_not_render_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unavailable_api(_url: str, _source_id: str) -> None:
        return None

    monkeypatch.setattr(ats_interceptor, "intercept_ats_board", _unavailable_api)
    board = {
        "id": "test:greenhouse",
        "url": "https://boards.greenhouse.io/test",
        "source_type": "official_ats",
    }

    assert await orchestrator._poll_board(board) == []
    assert sources.get_checkpoint("test:greenhouse").consecutive_failures == 1
