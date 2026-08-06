"""Unit tests for DiscordAgent (ingest-side gateway client)."""

import pytest
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


def test_parse_instruction_urls() -> None:
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction("resume https://x.com/resume.pdf portfolio https://y.com")
    assert parsed["resume_url"] == "https://x.com/resume.pdf"
    assert parsed["website"] == "https://y.com"
    assert parsed["force_all"] is False


def test_parse_instruction_bare_url_is_resume() -> None:
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction("add my resume https://x.com/resume.pdf")
    assert parsed["resume_url"] == "https://x.com/resume.pdf"


def test_parse_instruction_force_all_and_no_resume() -> None:
    from src.agent.memory_wizard import parse_instruction

    assert parse_instruction("everything")["force_all"] is True
    assert parse_instruction("re-grill all")["force_all"] is True
    assert parse_instruction("no resume")["no_resume"] is True


def test_parse_instruction_no_keywords() -> None:
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction("update this and add my resume and portfolio")
    assert parsed["resume_url"] is None
    assert parsed["website"] is None


def test_parse_instruction_portfolio_only_url() -> None:
    """A single bare portfolio URL after 'portfolio' must NOT become the resume.

    Regression: `/memory update this and add my resume and portfolio
    https://przknv.cc` used to treat przknv.cc as BOTH resume and website,
    indexing the portfolio homepage HTML into resume_embeddings.
    """
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction("update this and add my resume and portfolio https://przknv.cc")
    assert parsed["website"] == "https://przknv.cc"
    assert parsed["resume_url"] is None
    assert parsed["resume_path"] is None


def test_parse_instruction_resume_url_and_portfolio() -> None:
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction(
        "my resume https://f.przknv.cc/raw/XghaIR.pdf and portfolio https://przknv.cc"
    )
    assert parsed["resume_url"] == "https://f.przknv.cc/raw/XghaIR.pdf"
    assert parsed["website"] == "https://przknv.cc"


def test_parse_instruction_bare_url_no_keyword_is_resume() -> None:
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction("https://f.przknv.cc/raw/XghaIR.pdf")
    assert parsed["resume_url"] == "https://f.przknv.cc/raw/XghaIR.pdf"
    assert parsed["website"] is None


def test_parse_instruction_portfolio_second_of_two_urls() -> None:
    from src.agent.memory_wizard import parse_instruction

    parsed = parse_instruction("resume https://x.com/a.pdf portfolio https://y.com")
    assert parsed["resume_url"] == "https://x.com/a.pdf"
    assert parsed["website"] == "https://y.com"


def test_parse_instruction_explicit_portfolio_wins(monkeypatch) -> None:
    from src.agent.memory_wizard import parse_instruction

    monkeypatch.setenv("PORTFOLIO_URL", "https://env.example.com")
    parsed = parse_instruction("portfolio https://explicit.example.com")
    assert parsed["website"] == "https://explicit.example.com"


@pytest.mark.asyncio
async def test_resolve_website_uses_env_when_no_explicit(monkeypatch) -> None:
    from src.agent.memory_wizard import MemoryWizard

    monkeypatch.setenv("PORTFOLIO_URL", "https://env.example.com")
    w = MemoryWizard(ask=lambda q, m: _noop_ask(), log=_noop_log)

    data = {"identity": {}}
    resolved = await w._resolve_website(data, {"website": ""}, "update this")
    assert resolved == "https://env.example.com"
    assert data["identity"]["website"] == "https://env.example.com"


@pytest.mark.asyncio
async def test_resolve_website_explicit_wins_over_env(monkeypatch) -> None:
    from src.agent.memory_wizard import MemoryWizard

    monkeypatch.setenv("PORTFOLIO_URL", "https://env.example.com")
    w = MemoryWizard(ask=lambda q, m: _noop_ask(), log=_noop_log)

    data = {"identity": {}}
    resolved = await w._resolve_website(data, {"website": "https://explicit.example.com"}, "update")
    assert resolved == "https://explicit.example.com"
    assert data["identity"]["website"] == "https://explicit.example.com"


@pytest.mark.asyncio
async def test_resolve_website_keeps_existing_without_explicit(monkeypatch) -> None:
    from src.agent.memory_wizard import MemoryWizard

    monkeypatch.setenv("PORTFOLIO_URL", "https://env.example.com")
    w = MemoryWizard(ask=lambda q, m: _noop_ask(), log=_noop_log)

    data = {"identity": {"website": "https://saved.example.com"}}
    await w._resolve_website(data, {"website": ""}, "update this")
    # env is set but a saved value exists and no explicit URL -> keep saved
    assert data["identity"]["website"] == "https://saved.example.com"


@pytest.mark.asyncio
async def test_resolve_website_explicit_overwrites_existing(monkeypatch) -> None:
    from src.agent.memory_wizard import MemoryWizard

    monkeypatch.setenv("PORTFOLIO_URL", "https://env.example.com")
    w = MemoryWizard(ask=lambda q, m: _noop_ask(), log=_noop_log)

    data = {"identity": {"website": "https://saved.example.com"}}
    resolved = await w._resolve_website(data, {"website": "https://new.example.com"}, "update")
    assert resolved == "https://new.example.com"
    assert data["identity"]["website"] == "https://new.example.com"


async def _noop_ask() -> None:
    return None


async def _noop_log(_t: str) -> None:
    return None
