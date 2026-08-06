"""Tests for US-location eligibility filtering and Telegram card warnings."""

from __future__ import annotations

import pytest

from src.radar.core.models import EligibilityState, JobCandidate, RejectionReason
from src.radar.core.queue import _apply_llm_result
from src.radar.core.signals import is_us_location


class TestIsUsLocation:
    @pytest.mark.parametrize(
        "loc",
        [
            "New York, NY",
            "San Francisco, CA (Hybrid)",
            "Remote - US",
            "United States",
            "USA",
            "Remote (US)",
            "Austin, TX",
            "Seattle, WA",
            "Washington DC",
        ],
    )
    def test_us_locations(self, loc: str) -> None:
        assert is_us_location(loc) is True

    @pytest.mark.parametrize(
        "loc",
        [
            "London, UK",
            "Berlin, Germany",
            "Remote (Worldwide)",
            "Bengaluru, India",
            "Toronto, Canada",
            "Paris, France",
            "Remote",
            "",
            "Amsterdam, Netherlands",
        ],
    )
    def test_non_us_locations(self, loc: str) -> None:
        assert is_us_location(loc) is False


class _FakeCfg:
    class _Radar:
        us_only_remote = True

    radar = _Radar()


def _matched(location: str, *, is_remote: bool = False) -> JobCandidate:
    c = JobCandidate(
        canonical_id="t",
        source="greenhouse",
        direct_apply_url="https://example.com/job",
        normalized_company="Acme",
        normalized_role="Engineer",
        normalized_location="",
    )
    result = {
        "role": "Engineer",
        "company": "Acme",
        "match_percent": 85,
        "shortlist_probability": 70,
        "verdict": "STRONG_MATCH",
        "location": location,
        "is_remote": is_remote,
        "matching_skills": ["python"],
        "missing_skills": [],
    }
    _apply_llm_result(c, result)
    return c


@pytest.mark.asyncio
async def test_us_onsite_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.configuration import get_config

    monkeypatch.setattr(get_config().radar, "us_only_remote", True)
    c = _matched("San Francisco, CA")
    assert c.eligibility == EligibilityState.REJECTED
    assert c.rejection_reason == RejectionReason.US_ONSITE


@pytest.mark.asyncio
async def test_us_remote_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.configuration import get_config

    monkeypatch.setattr(get_config().radar, "us_only_remote", True)
    c = _matched("Remote (US)", is_remote=True)
    assert c.eligibility == EligibilityState.ACCEPTED


@pytest.mark.asyncio
async def test_non_us_onsite_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.configuration import get_config

    monkeypatch.setattr(get_config().radar, "us_only_remote", True)
    c = _matched("London, UK")
    assert c.eligibility == EligibilityState.ACCEPTED


@pytest.mark.asyncio
async def test_filter_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.configuration import get_config

    monkeypatch.setattr(get_config().radar, "us_only_remote", False)
    c = _matched("San Francisco, CA")
    assert c.eligibility == EligibilityState.ACCEPTED


class TestCardWarnings:
    def _warnings(self, job: dict) -> list[str]:
        from src.agent.discord_agent import _build_job_embed

        embed = _build_job_embed("eligible", job)
        for field in embed.fields:
            if field.name.startswith("⚠"):
                return field.value.split("\n")
        return []

    def test_onsite_us_card_warns(self) -> None:
        warnings = self._warnings(
            {
                "role": "Engineer",
                "company": "Acme",
                "match_percent": 85,
                "location": "New York, NY",
                "is_remote": False,
                "sponsors_visa": False,
            }
        )
        assert any("Onsite role" in w for w in warnings)
        assert any("US role - visa sponsorship not confirmed" in w for w in warnings)

    def test_remote_us_card_no_onsite_warning(self) -> None:
        warnings = self._warnings(
            {
                "role": "Engineer",
                "company": "Acme",
                "match_percent": 85,
                "location": "Remote (US)",
                "is_remote": True,
                "sponsors_visa": True,
            }
        )
        assert not any("Onsite role" in w for w in warnings)
        assert not any("visa sponsorship not confirmed" in w for w in warnings)

    def test_onsite_non_us_still_warns(self) -> None:
        warnings = self._warnings(
            {
                "role": "Engineer",
                "company": "Acme",
                "match_percent": 85,
                "location": "London, UK",
                "is_remote": False,
                "sponsors_visa": False,
            }
        )
        assert any("Onsite role" in w for w in warnings)
        assert not any("US role" in w for w in warnings)
