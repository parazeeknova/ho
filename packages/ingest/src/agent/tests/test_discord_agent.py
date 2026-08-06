"""Unit tests for DiscordAgent (ingest-side gateway client)."""

from src.agent.discord_agent import DiscordAgent


def test_discord_agent_init_unconfigured() -> None:
    agent = DiscordAgent(bot_token="", channel_id="")
    assert agent.is_configured is False


def test_discord_agent_init_configured() -> None:
    agent = DiscordAgent(bot_token="abc", channel_id="123")
    assert agent.is_configured is True


def test_chat_id_alias() -> None:
    agent = DiscordAgent(bot_token="abc", chat_id="999")
    assert agent.is_configured is True
    assert agent.channel_id == "999"


async def test_build_job_embed() -> None:
    from src.agent.discord_agent import _build_job_embed

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
    embed = await _build_job_embed("eligible", job)
    assert embed.title is not None and "Backend Engineer" in embed.title
    assert "Stripe" in str(embed.fields)
    assert "90% match" in str(embed.fields)
    assert "Remote" in str(embed.fields)


def test_autofill_queue_lines_guards() -> None:
    import asyncio

    from src.agent.discord_agent import autofill_queue_lines

    lines = asyncio.run(autofill_queue_lines())
    assert isinstance(lines, list)
    assert any("Applied" in line for line in lines)
