"""Unit tests for TelegramAgent."""

import pytest
from src.agent.telegram_agent import TelegramAgent


def test_telegram_agent_init_unconfigured() -> None:
    agent = TelegramAgent(bot_token="", chat_id="")
    assert agent.is_configured is False


def test_telegram_agent_init_configured() -> None:
    agent = TelegramAgent(
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11", chat_id="987654321"
    )
    assert agent.is_configured is True


def test_format_job_card() -> None:
    agent = TelegramAgent()
    job = {
        "role": "Backend Engineer",
        "company": "Stripe",
        "match_percent": 90,
        "shortlist_probability": 85,
        "salary": "$120,000 - $140,000",
        "location": "Remote",
        "apply_link": "https://stripe.com/jobs/123",
        "company_description": "Stripe is a financial infrastructure platform for the internet.",
        "founders": ["Patrick Collison", "John Collison"],
        "funding_stage": "Series I",
    }

    card = agent.format_job_card(job)
    assert "Backend Engineer" in card
    assert "Stripe" in card
    assert "90%" in card
    assert "Patrick Collison" in card
    assert "Flexible / Competitive" not in card


@pytest.mark.asyncio
async def test_notify_unconfigured_noop() -> None:
    agent = TelegramAgent(bot_token="", chat_id="")
    job = {"role": "Engineer", "company": "Acme", "match_percent": 80}
    sent = await agent.notify_verified_jobs([job])
    assert sent == 0


@pytest.mark.asyncio
async def test_handle_resend_unconfigured() -> None:
    agent = TelegramAgent(bot_token="", chat_id="")
    # Should safely return without sending if unconfigured
    await agent._handle_resend("/resend 5")
    await agent._handle_resend("/resend --dry")
    await agent._handle_resend("/resend --dry 5")


@pytest.mark.asyncio
async def test_clear_command_removed() -> None:
    """The /clear command was removed by royal decree; ensure it's gone."""
    agent = TelegramAgent(bot_token="", chat_id="")
    assert not hasattr(agent, "_handle_clear")
