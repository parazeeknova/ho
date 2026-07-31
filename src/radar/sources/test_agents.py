"""Tests for radar agent implementations and Telegram categorized alerts."""

from __future__ import annotations

import pytest

from src.agent.telegram_agent import TelegramAgent
from src.graph.entity import FrontierEntry, NodeType
from src.radar.core.models import EligibilityState, JobCandidate


class TestTelegramCategorizedAlerts:
    def test_category_icons_defined(self) -> None:
        agent = TelegramAgent(bot_token="test", chat_id="123")
        assert "urgent" in agent._CATEGORY_ICONS
        assert agent._CATEGORY_ICONS["urgent"] == "[URGENT]"
        assert agent._CATEGORY_ICONS["startup_signal"] == "[SIGNAL]"
        assert agent._CATEGORY_ICONS["outreach"] == "[OUTREACH]"
        assert agent._CATEGORY_ICONS["eligible"] == "[ELIGIBLE]"
        assert agent._CATEGORY_ICONS["review"] == "[REVIEW]"

    def test_category_labels_defined(self) -> None:
        agent = TelegramAgent(bot_token="test", chat_id="123")
        labels = agent._CATEGORY_LABELS
        assert "urgent" in labels
        assert "startup_signal" in labels
        assert "eligible" in labels

    def test_format_job_card_structure(self) -> None:
        agent = TelegramAgent(bot_token="test", chat_id="123")
        job = {
            "role": "Backend Engineer",
            "company": "TestCo",
            "match_percent": 85,
            "shortlist_probability": 72,
            "salary": "$120K/yr",
            "location": "Remote",
            "company_description": "A test company.",
        }
        card = agent.format_job_card(job)
        assert "Backend Engineer" in card
        assert "TestCo" in card
        assert "85%" in card
        assert "Remote" in card
        assert "est." not in card

    def test_unconfigured_agent_noop(self) -> None:
        agent = TelegramAgent(bot_token="", chat_id="")
        assert not agent.is_configured


class TestDedupNotification:
    @pytest.mark.asyncio
    async def test_send_categorized_alert_dedup(self) -> None:
        agent = TelegramAgent(bot_token="test", chat_id="")
        agent._notified_keys.add("test:role:remote")

        job = {"role": "TestRole", "company": "TestCo", "match_percent": 80}
        result = await agent.send_categorized_alert("urgent", job, dedup_key="test:role:remote")
        assert result is False

    def test_dedup_key_inheritance(self) -> None:
        job = {"role": "SWE", "company": "Acme", "match_percent": 90}
        dedup = f"{job['company'].lower()}:{job['role'].lower()}"
        assert dedup == "acme:swe"


class TestAgentHandlers:
    @pytest.mark.asyncio
    async def test_career_site_detector_no_url(self) -> None:
        from src.radar.sources.agents import career_site_detector

        entry = FrontierEntry(
            id="test",
            agent="career_site_detector",
            node_id="node1",
            node_type=NodeType.COMPANY,
            priority=60,
            depth=1,
            payload={"company": "TestCo", "url": ""},
        )
        result = await career_site_detector(entry)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_ats_crawler_no_url(self) -> None:
        from src.radar.sources.agents import ats_crawler

        entry = FrontierEntry(
            id="test",
            agent="ats_crawler",
            node_id="node1",
            node_type=NodeType.CAREER_SITE,
            priority=60,
            depth=1,
            payload={"company": "TestCo", "ats_url": ""},
        )
        result = await ats_crawler(entry)
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_founder_social_agent_no_name(self) -> None:
        from src.radar.sources.agents import founder_social_agent

        entry = FrontierEntry(
            id="test",
            agent="founder_social_agent",
            node_id="node1",
            node_type=NodeType.FOUNDER,
            priority=50,
            depth=1,
            payload={"founder_name": "", "company": "TestCo"},
        )
        result = await founder_social_agent(entry)
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_employee_discovery_no_company(self) -> None:
        from src.radar.sources.agents import employee_discovery_agent

        entry = FrontierEntry(
            id="test",
            agent="employee_discovery",
            node_id="node1",
            node_type=NodeType.FOUNDER,
            priority=45,
            depth=1,
            payload={"company": ""},
        )
        result = await employee_discovery_agent(entry)
        assert isinstance(result, list)
        assert len(result) == 0


class TestATSIdentification:
    def test_identify_greenhouse(self) -> None:
        from src.radar.sources.agents import _identify_ats

        assert _identify_ats("https://boards.greenhouse.io/company") == "greenhouse"

    def test_identify_lever(self) -> None:
        from src.radar.sources.agents import _identify_ats

        assert _identify_ats("https://jobs.lever.co/company/123") == "lever"

    def test_identify_ashby(self) -> None:
        from src.radar.sources.agents import _identify_ats

        assert _identify_ats("https://jobs.ashbyhq.com/company/123") == "ashby"

    def test_identify_workable(self) -> None:
        from src.radar.sources.agents import _identify_ats

        assert _identify_ats("https://apply.workable.com/company") == "workable"

    def test_identify_workday(self) -> None:
        from src.radar.sources.agents import _identify_ats

        assert _identify_ats("https://company.myworkdayjobs.com/careers") == "workday"

    def test_identify_unknown(self) -> None:
        from src.radar.sources.agents import _identify_ats

        assert _identify_ats("https://company.com/careers") == "careers_page"


class TestColdOutreach:
    def test_generate_outreach_for_urgent_accepted(self) -> None:
        from src.radar.core.models import FreshnessLane
        from src.radar.engine.outreach import generate_outreach_card

        candidate = JobCandidate(
            canonical_id="test:role:remote",
            source="lever",
            direct_apply_url="https://jobs.lever.co/test/1",
            normalized_company="TestCo",
            normalized_role="Backend Engineer",
            normalized_location="Remote",
            freshness_lane=FreshnessLane.URGENT,
        )
        candidate.eligibility = EligibilityState.ACCEPTED

        card = generate_outreach_card(candidate)
        assert card is not None
        assert card.company == "TestCo"
        assert card.hiring_signal == "open_role"

    def test_generate_outreach_for_funding(self) -> None:
        from src.radar.engine.outreach import generate_outreach_card

        candidate = JobCandidate(
            canonical_id="test:role:remote",
            source="lever",
            direct_apply_url="https://jobs.lever.co/test/1",
            normalized_company="TestCo",
            normalized_role="Backend Engineer",
            normalized_location="Remote",
            funding_stage="Series A",
        )
        card = generate_outreach_card(candidate)
        assert card is not None
        assert card.hiring_signal == "funding"
        assert "Series A" in card.why_now

    def test_no_outreach_without_signal(self) -> None:
        from src.radar.engine.outreach import generate_outreach_card

        candidate = JobCandidate(
            canonical_id="test:role:remote",
            source="lever",
            direct_apply_url="https://jobs.lever.co/test/1",
            normalized_company="TestCo",
            normalized_role="Backend Engineer",
            normalized_location="Remote",
        )
        card = generate_outreach_card(candidate)
        assert card is None


class TestSourceModules:
    def test_register_and_get_checkpoint(self) -> None:
        from src.radar.sources.sources import get_checkpoint, register_source

        register_source("greenhouse:acme", "ats_board")
        cp = get_checkpoint("greenhouse:acme")
        assert cp.source_id == "greenhouse:acme"
        assert cp.source_type == "ats_board"

    def test_compute_snapshot_hash(self) -> None:
        from src.radar.sources.sources import compute_url_snapshot_hash

        urls = ["https://a.com/1", "https://a.com/2", "https://a.com/3"]
        h1 = compute_url_snapshot_hash(urls)
        h2 = compute_url_snapshot_hash(list(reversed(urls)))
        assert h1 == h2

    def test_should_poll_inactive_source(self) -> None:
        import time

        from src.radar.sources.sources import get_checkpoint, register_source, should_poll

        register_source("test:inactive", "ats_board", initial_quality=0.5)
        cp = get_checkpoint("test:inactive")
        cp.active = False
        cp.backoff_until = time.time() + 3600
        assert not should_poll("test:inactive")

    def test_should_poll_active_source(self) -> None:
        from src.radar.sources.sources import register_source, should_poll

        register_source("test:active", "ats_board", initial_quality=0.8)
        assert should_poll("test:active")

    def test_record_failure_decreases_quality(self) -> None:
        from src.radar.sources.sources import get_checkpoint, record_failure, register_source

        register_source("test:failing", "ats_board", initial_quality=0.8)
        initial_score = get_checkpoint("test:failing").quality_score
        record_failure("test:failing")
        assert get_checkpoint("test:failing").quality_score < initial_score

    def test_record_success_increases_quality(self) -> None:
        from src.radar.sources.sources import get_checkpoint, record_success, register_source

        register_source("test:succeeding", "ats_board", initial_quality=0.5)
        initial_score = get_checkpoint("test:succeeding").quality_score
        record_success("test:succeeding", job_count=10, direct_url_count=10)
        assert get_checkpoint("test:succeeding").quality_score > initial_score

    def test_get_source_health(self) -> None:
        from src.radar.sources.sources import get_source_health, register_source

        register_source("test:health", "ats_board")
        health = get_source_health()
        assert "test:health" in health
        assert health["test:health"]["active"]
