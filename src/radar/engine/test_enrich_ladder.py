"""Tests for the cost-aware enrichment ladder (_enrich_high_fit)."""

from __future__ import annotations

import pytest

from src.radar.core.models import EligibilityState, JobCandidate
from src.radar.engine import orchestrator as orch


def _accepted(
    company: str,
    role: str,
    match: int,
    *,
    eligibility: EligibilityState = EligibilityState.ACCEPTED,
) -> JobCandidate:
    return JobCandidate(
        canonical_id=f"{company}:{role}",
        source="greenhouse",
        direct_apply_url=f"https://boards.greenhouse.io/{company}/jobs/1",
        normalized_company=company,
        normalized_role=role,
        normalized_location="Remote",
        match_percent=match,
        eligibility=eligibility,
    )


class _FakeSA:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def batch_analyze_startups(self, jobs: list[dict], concurrency: int = 8) -> list[dict]:
        self.calls.append(jobs)
        return [
            {
                **j,
                "funding_stage": "Seed",
                "founders": [{"name": "Ada"}],
                "osint_signals": ["Raised Seed (2026)"],
            }
            for j in jobs
        ]


class _FakeStore:
    def __init__(self, summary: dict | None = None) -> None:
        self._summary = summary or {"rows": [], "support": 0, "contradict": 0, "confidence": 0.5}

    async def evidence_summary(self, company_id: str) -> dict:
        return self._summary


class _FakeGraph:
    async def generate_graph_insights_for_llm(self, company_id: str) -> str | None:
        return None


@pytest.mark.asyncio
async def test_dedupes_by_company(monkeypatch: pytest.MonkeyPatch) -> None:
    sa = _FakeSA()
    await orch._enrich_high_fit(
        [
            _accepted("acme", "Backend Engineer", 80),
            _accepted("acme", "Frontend Engineer", 75),
            _accepted("zeta", "ML Engineer", 85),
        ],
        sa,
        _FakeStore(),
        _FakeGraph(),
    )
    assert len(sa.calls) == 1
    jobs = sa.calls[0]
    assert len(jobs) == 2  # one per unique company
    assert {j["company"].lower() for j in jobs} == {"acme", "zeta"}


@pytest.mark.asyncio
async def test_deep_enrich_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    sa = _FakeSA()
    await orch._enrich_high_fit(
        [
            _accepted("strong", "Engineer", 92),  # high match -> deep
            _accepted("weak", "Engineer", 45),  # low match, single role -> shallow
            _accepted("multi", "Engineer A", 60),
            _accepted("multi", "Engineer B", 55),  # multi-role -> deep
        ],
        sa,
        _FakeStore(),
        _FakeGraph(),
    )
    by_company = {j["company"]: j for j in sa.calls[0]}
    assert by_company["strong"]["deep_enrich"] is True
    assert by_company["weak"]["deep_enrich"] is False
    assert by_company["multi"]["deep_enrich"] is True


@pytest.mark.asyncio
async def test_strong_evidence_skips_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    sa = _FakeSA()
    store = _FakeStore({"rows": [], "support": 2, "contradict": 0, "confidence": 0.8})
    await orch._enrich_high_fit(
        [_accepted("covered", "Engineer", 80)],
        sa,
        store,
        _FakeGraph(),
    )
    assert sa.calls == []  # evidence already strong -> no OSINT run


@pytest.mark.asyncio
async def test_weak_evidence_still_enriches(monkeypatch: pytest.MonkeyPatch) -> None:
    sa = _FakeSA()
    await orch._enrich_high_fit(
        [_accepted("fresh", "Engineer", 80)],
        sa,
        _FakeStore(),
        _FakeGraph(),
    )
    assert len(sa.calls) == 1
