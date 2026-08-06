# mypy: ignore-errors
"""Comprehensive unit tests for the Discord slash commands and memory wizard.

Uses fake Message/Interaction/Thread objects (no live Discord or network),
mocking the heavy service calls at their real import sites. Covers:

  /status, /health, /help, /persona, /resend (dry-run + real + empty + fail),
  /analytics, /memory thread creation + answer routing, memory button
  routing, the extra-Q&A loop guard, and the on_message dispatcher.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from src.agent.discord_agent import (
    _MEMORY_BUTTON_VALUES,
    DiscordAgent,
    _MemoryWizardSession,
)

# ── fakes ──────────────────────────────────────────────────────────────


class FakeAuthor:
    def __init__(self, bot: bool = False) -> None:
        self.bot = bot


class FakeMessage:
    def __init__(self, content: str = "", channel: Any = None, author: Any = None) -> None:
        self.content = content
        self.channel = channel or FakeChannel()
        self.author = author or FakeAuthor()
        self.reference = None
        self.id = 1


class FakeChannel:
    def __init__(self, id: int = 100, thread: bool = False) -> None:
        self.id = id
        self._thread = thread
        self.sent: list[tuple[str | None, dict]] = []

    async def send(self, content: str | None = None, **kwargs):
        self.sent.append((content, kwargs))
        return FakeMessage(content=content or "", channel=self)


class FakeThread(FakeChannel):
    def __init__(self, id: int = 200) -> None:
        super().__init__(id=id, thread=True)


def patch_memory_threads(agent: DiscordAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make _is_memory_thread recognize FakeThread ids (skips the real
    `isinstance(discord.Thread)` gate)."""

    def is_thread(channel: Any) -> bool:
        return isinstance(channel, FakeChannel) and channel.id in agent._memory_sessions

    monkeypatch.setattr(agent, "_is_memory_thread", is_thread)


class FakeServerChannel(FakeChannel):
    """A channel that can create threads (for /memory)."""

    def __init__(self, thread: FakeThread | None = None) -> None:
        super().__init__(id=123)
        self.thread = thread or FakeThread(id=999)

    async def send(self, content: str | None = None, **kwargs):
        self.sent.append((content, kwargs))
        starter = FakeMessage(content=content or "", channel=self)
        starter.create_thread = self.create_thread  # type: ignore[attr-defined]
        return starter

    async def create_thread(self, name: str, auto_archive_duration: int) -> FakeThread:
        return self.thread


class FakeInteraction:
    def __init__(self, custom_id: str = "", message: Any = None) -> None:
        self.type = 2  # component
        self.data = {"custom_id": custom_id}
        self.message = message
        self.response = FakeResponse()
        self.followup = FakeFollowup()


class FakeFollowup:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, text: str) -> None:
        self.sent.append(text)


class FakeResponse:
    async def defer(self, *a, **k):
        pass

    async def send_message(self, *a, **k):
        pass


# ── helpers ────────────────────────────────────────────────────────────


def make_agent(ctx: Any = None) -> DiscordAgent:
    agent = DiscordAgent(bot_token="tok", channel_id="123", ctx=ctx)
    agent._channel = FakeChannel(id=123)
    return agent


# Attach a test-only dispatcher that mirrors the real on_message closure
# registered in start_polling (same routing order, no network).
async def _on_message_for_test(self: DiscordAgent, message: FakeMessage) -> None:
    if message.author.bot:
        return
    if (
        self._channel is not None
        and message.channel != self._channel
        and not self._is_memory_thread(message.channel)
    ):
        return
    if self._is_memory_answer(message):
        return
    if not message.content.startswith("/"):
        return
    cmd = message.content.split()[0].lower()
    if cmd == "/memory":
        await self._handle_memory(message)
    elif cmd == "/persona":
        await self._handle_persona(message)
    elif cmd == "/status":
        await self._handle_status(message)
    elif cmd == "/health":
        await self._handle_health(message)
    elif cmd == "/analytics":
        await self._handle_analytics(message)
    elif cmd == "/resend":
        await self._handle_resend(message)
    elif cmd == "/help":
        await self._handle_help(message)


DiscordAgent.on_message_for_test = _on_message_for_test  # type: ignore[attr-defined]


def sent_text(agent: DiscordAgent) -> str:
    return "".join(c or "" for c, _ in agent._channel.sent)


def job_row(**overrides: Any) -> dict:
    row = {
        "canonical_id": "1",
        "normalized_role": "Eng",
        "normalized_company": "Acme",
        "direct_apply_url": "https://a",
        "normalized_location": "Remote",
        "match_percent": 90,
        "shortlist_probability": 80,
        "verdict": "MATCH",
        "funding_stage": "",
        "salary_amount": None,
        "salary_currency": "",
        "salary_period": "",
        "salary_raw": "",
        "jd_summary": "",
        "company_description": "",
        "role_summary": "",
        "is_remote": True,
        "founders": [],
        "funding_info": {},
        "founder_socials": [],
        "company_news": "",
        "osint_signals": [],
        "extra": {},
    }
    row.update(overrides)
    return row


class FakePool:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    async def fetch(self, sql, *args):
        return self.rows

    def acquire(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeMemStore:
    _rows: list[dict] = []

    def __init__(self, rows: list[dict] | None = None) -> None:
        if rows is not None:
            FakeMemStore._rows = rows
        self._pool = FakePool(FakeMemStore._rows)

    @classmethod
    async def create(cls, *a, **k):
        return cls(FakeMemStore._rows)

    async def close(self) -> None:
        pass


# ── /status ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_reports_pipeline_state() -> None:
    from src.agent.discord_agent import set_pipeline_state

    set_pipeline_state(running=True, sweep=3, phase="matching", matched_total=12, scraped_count=40)
    agent = make_agent()
    await agent._handle_status(FakeMessage(content="/status", channel=agent._channel))
    text = sent_text(agent)
    assert "Pipeline Status" in text
    assert "Running: True" in text
    assert "Sweep: 3" in text
    assert "Matched total: 12" in text


@pytest.mark.asyncio
async def test_status_includes_last_error() -> None:
    from src.agent.discord_agent import set_pipeline_state

    set_pipeline_state(last_error="embed timeout")
    agent = make_agent()
    await agent._handle_status(FakeMessage(content="/status", channel=agent._channel))
    assert "embed timeout" in sent_text(agent)


# ── /health ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_reports_services(monkeypatch) -> None:
    async def fake_health() -> str:
        return "**System Health Check**\n\n✅ llama-server (Embed)\n✅ agent-memory-db (pgvector)"

    from src.agent import discord_agent

    monkeypatch.setattr(discord_agent, "run_health_checks", fake_health)
    agent = make_agent()
    await agent._handle_health(FakeMessage(content="/health", channel=agent._channel))
    text = sent_text(agent)
    assert "System Health Check" in text
    assert "agent-memory-db" in text


@pytest.mark.asyncio
async def test_health_graceful_on_failure(monkeypatch) -> None:
    async def broken() -> str:
        raise RuntimeError("boom")

    from src.agent import discord_agent

    monkeypatch.setattr(discord_agent, "run_health_checks", broken)
    agent = make_agent()
    await agent._handle_health(FakeMessage(content="/health", channel=agent._channel))
    assert "Health check failed" in sent_text(agent)


# ── /help ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_help_lists_all_commands() -> None:
    agent = make_agent()
    await agent._handle_help(FakeMessage(content="/help", channel=agent._channel))
    text = sent_text(agent)
    for cmd in ("/memory", "/persona", "/status", "/health", "/analytics", "/resend", "/help"):
        assert cmd in text, f"missing {cmd} from /help"


# ── /persona ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persona_renders_stored_persona(monkeypatch, tmp_path) -> None:
    p = tmp_path / "persona.json"
    p.write_text(json.dumps({"name": "Harsh Sahu", "identity": {"website": "https://przknv.cc"}}))
    monkeypatch.setattr("src.agent.memory_wizard.PERSONA_JSON", p)
    agent = make_agent()
    await agent._handle_persona(FakeMessage(content="/persona", channel=agent._channel))
    text = sent_text(agent)
    assert "Harsh Sahu" in text
    assert "przknv.cc" in text


@pytest.mark.asyncio
async def test_persona_chunks_long_output(monkeypatch, tmp_path) -> None:
    big = "x" * 3000
    p = tmp_path / "persona.json"
    p.write_text(
        json.dumps({"name": "Harsh", "identity": {}, "answers": [], "resume_summary": big})
    )
    monkeypatch.setattr("src.agent.memory_wizard.PERSONA_JSON", p)
    agent = make_agent()
    await agent._handle_persona(FakeMessage(content="/persona", channel=agent._channel))
    chunks = [c or "" for c, _ in agent._channel.sent if c]
    assert sum(len(c) for c in chunks) >= 3000
    assert all(len(c) <= 1900 for c in chunks)


@pytest.mark.asyncio
async def test_persona_graceful_missing(monkeypatch, tmp_path) -> None:
    p = tmp_path / "persona.json"
    p.write_text("not json")
    monkeypatch.setattr("src.agent.memory_wizard.PERSONA_JSON", p)
    agent = make_agent()
    await agent._handle_persona(FakeMessage(content="/persona", channel=agent._channel))
    text = sent_text(agent)
    assert "No persona.json yet" in text


# ── /resend ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resend_dry_run(monkeypatch) -> None:
    monkeypatch.setattr("src.memory.pgvector_store.MemoryStore", FakeMemStore([job_row()]))
    agent = make_agent()
    await agent._handle_resend(FakeMessage(content="/resend --dry 5", channel=agent._channel))
    text = sent_text(agent)
    assert "dry run" in text
    assert "would re-send 1" in text


@pytest.mark.asyncio
async def test_resend_empty(monkeypatch) -> None:
    monkeypatch.setattr("src.memory.pgvector_store.MemoryStore", FakeMemStore([]))
    agent = make_agent()
    await agent._handle_resend(FakeMessage(content="/resend", channel=agent._channel))
    assert "No accepted candidates" in sent_text(agent)


@pytest.mark.asyncio
async def test_resend_real_sends_alerts(monkeypatch) -> None:
    monkeypatch.setattr("src.memory.pgvector_store.MemoryStore", FakeMemStore([job_row()]))
    agent = make_agent()
    sent_alerts: list[tuple[str, dict]] = []

    async def fake_send(cat, job):
        sent_alerts.append((cat, job))
        return True

    agent.send_categorized_alert = fake_send  # type: ignore[method-assign]
    await agent._handle_resend(FakeMessage(content="/resend 3", channel=agent._channel))
    assert len(sent_alerts) == 1
    assert sent_alerts[0][0] == "eligible"
    assert sent_alerts[0][1]["company"] == "Acme"
    assert "Re-sent 1" in sent_text(agent)


@pytest.mark.asyncio
async def test_resend_graceful_db_failure(monkeypatch) -> None:
    class Boom:
        @classmethod
        async def create(cls, *a, **k):
            raise RuntimeError("db down")

    monkeypatch.setattr("src.memory.pgvector_store.MemoryStore", Boom)
    agent = make_agent()
    await agent._handle_resend(FakeMessage(content="/resend", channel=agent._channel))
    assert "Resend failed" in sent_text(agent)


# ── /analytics ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analytics_no_ctx_tells_user() -> None:
    agent = make_agent(ctx=None)
    await agent._handle_analytics(FakeMessage(content="/analytics", channel=agent._channel))
    assert "Analytics unavailable" in sent_text(agent)


@pytest.mark.asyncio
async def test_analytics_with_ctx_emits_sections(monkeypatch) -> None:
    agent = make_agent(ctx=object())

    async def fake_report(self):
        return ["**Market**\nJobs up 20%", "**Skills**\nRust in demand"]

    monkeypatch.setattr(
        "src.agent.analytics_agent.AnalyticsAgent",
        lambda **k: type("A", (), {"generate_resilient_report": fake_report})(),
    )
    monkeypatch.setattr("src.graph.graph_store.GraphStore", FakeMemStore)
    monkeypatch.setattr("src.memory.pgvector_store.MemoryStore", FakeMemStore)

    async def fake_queue_lines():
        return ["**Applied:** 3"]

    monkeypatch.setattr("src.agent.discord_agent.autofill_queue_lines", fake_queue_lines)

    await agent._handle_analytics(FakeMessage(content="/analytics", channel=agent._channel))
    text = sent_text(agent)
    assert "Market" in text
    assert "Skills" in text
    assert "Applied" in text


@pytest.mark.asyncio
async def test_analytics_with_ctx_error(monkeypatch) -> None:
    agent = make_agent(ctx=object())

    class Boom:
        async def generate_resilient_report(self):
            raise RuntimeError("llm down")

    monkeypatch.setattr(
        "src.agent.analytics_agent.AnalyticsAgent",
        lambda **k: type(
            "B", (), {"generate_resilient_report": Boom().generate_resilient_report}
        )(),
    )
    monkeypatch.setattr("src.graph.graph_store.GraphStore", FakeMemStore)
    monkeypatch.setattr("src.memory.pgvector_store.MemoryStore", FakeMemStore)

    await agent._handle_analytics(FakeMessage(content="/analytics", channel=agent._channel))
    assert "Analytics failed" in sent_text(agent)


# ── /memory thread creation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_creates_thread_and_runs_wizard(monkeypatch) -> None:
    agent = make_agent()
    thread = FakeThread(id=999)
    agent._channel = FakeServerChannel(thread=thread)

    async def fake_run(session, instruction):
        assert isinstance(session, _MemoryWizardSession)
        assert session.thread.id == 999
        assert instruction == "update this and add my resume"
        await session.log("wizard ran")
        return "Resume: `57` chunks"

    agent._run_memory_wizard = fake_run  # type: ignore[method-assign]
    await agent._handle_memory(
        FakeMessage(content="/memory update this and add my resume", channel=agent._channel)
    )
    # thread got the intro + the final result
    thread_texts = [c or "" for c, _ in thread.sent]
    assert any("Memory update" in t for t in thread_texts)
    assert any("57` chunks" in t for t in thread_texts)
    # the wizard session was registered and cleaned up
    assert 999 not in agent._memory_sessions


@pytest.mark.asyncio
async def test_memory_wizard_failure_reported_in_thread() -> None:
    agent = make_agent()
    thread = FakeThread(id=998)
    agent._channel = FakeServerChannel(thread=thread)

    async def boom(session, instruction):
        raise RuntimeError("embed server down")

    agent._run_memory_wizard = boom  # type: ignore[method-assign]
    await agent._handle_memory(FakeMessage(content="/memory", channel=agent._channel))
    thread_texts = [c or "" for c, _ in thread.sent]
    assert any("failed" in t.lower() for t in thread_texts)


@pytest.mark.asyncio
async def test_memory_requires_server_channel() -> None:
    agent = make_agent()
    # DM channel: FakeChannel has no create_thread
    msg = FakeMessage(content="/memory", channel=FakeChannel(id=1))
    await agent._handle_memory(msg)
    assert any("server text channel" in (c or "") for c, _ in msg.channel.sent)


# ── memory answer routing ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_answer_routes_to_pending(monkeypatch) -> None:
    agent = make_agent()
    patch_memory_threads(agent, monkeypatch)
    thread = FakeThread(id=555)
    session = _MemoryWizardSession(thread, agent)
    agent._memory_sessions[555] = session
    fut: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()
    session.pending = fut

    msg = FakeMessage(content="Rust and TypeScript", channel=thread)
    assert agent._is_memory_answer(msg) is True
    assert fut.done() and fut.result() == "Rust and TypeScript"


@pytest.mark.asyncio
async def test_memory_command_inside_thread_skips_current_question(monkeypatch) -> None:
    agent = make_agent()
    patch_memory_threads(agent, monkeypatch)
    thread = FakeThread(id=556)
    session = _MemoryWizardSession(thread, agent)
    agent._memory_sessions[556] = session
    fut: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()
    session.pending = fut

    msg = FakeMessage(content="/memory done", channel=thread)
    assert agent._is_memory_answer(msg) is True
    assert fut.done() and fut.result() == "skip"


@pytest.mark.asyncio
async def test_memory_answer_ignored_when_no_pending(monkeypatch) -> None:
    agent = make_agent()
    patch_memory_threads(agent, monkeypatch)
    thread = FakeThread(id=557)
    session = _MemoryWizardSession(thread, agent)
    agent._memory_sessions[557] = session
    session.pending = None
    msg = FakeMessage(content="random text", channel=thread)
    assert agent._is_memory_answer(msg) is False


# ── memory button routing ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_button_routes_value(monkeypatch) -> None:
    agent = make_agent()
    patch_memory_threads(agent, monkeypatch)
    thread = FakeThread(id=558)
    session = _MemoryWizardSession(thread, agent)
    agent._memory_sessions[558] = session
    fut: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()
    session.pending = fut

    inter = FakeInteraction(custom_id="mem_yes", message=FakeMessage(content="", channel=thread))
    await agent._memory_button("mem_yes", inter)
    assert fut.done() and fut.result() == "Yes"


@pytest.mark.asyncio
async def test_memory_button_no_pending_tells_user(monkeypatch) -> None:
    agent = make_agent()
    patch_memory_threads(agent, monkeypatch)
    thread = FakeThread(id=559)
    session = _MemoryWizardSession(thread, agent)
    agent._memory_sessions[559] = session
    session.pending = None
    inter = FakeInteraction(custom_id="mem_skip", message=FakeMessage(content="", channel=thread))
    await agent._memory_button("mem_skip", inter)
    assert inter.followup.sent and "No question pending" in inter.followup.sent[0]


@pytest.mark.asyncio
async def test_memory_button_stale_session(monkeypatch) -> None:
    agent = make_agent()
    patch_memory_threads(agent, monkeypatch)
    thread = FakeThread(id=560)
    inter = FakeInteraction(custom_id="mem_done", message=FakeMessage(content="", channel=thread))
    await agent._memory_button("mem_done", inter)
    assert inter.followup.sent and "already finished" in inter.followup.sent[0]


# ── dispatcher ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatcher_routes_commands(monkeypatch) -> None:
    agent = make_agent()
    handled: list[str] = []

    async def fake_reply(message, text):
        handled.append(text)

    agent._reply = fake_reply  # type: ignore[method-assign]

    for cmd in ("/status", "/health", "/help", "/persona"):
        await agent.on_message_for_test(FakeMessage(content=cmd, channel=agent._channel))
    assert any("Pipeline Status" in t for t in handled)
    assert any("System Health Check" in t for t in handled)
    assert any("Commands" in t for t in handled)
    assert any("Harsh Sahu" in t for t in handled)


@pytest.mark.asyncio
async def test_dispatcher_ignores_non_command_and_bots() -> None:
    agent = make_agent()
    await agent.on_message_for_test(FakeMessage(content="just chatting", channel=agent._channel))
    assert not agent._channel.sent

    await agent.on_message_for_test(
        FakeMessage(content="/status", channel=agent._channel, author=FakeAuthor(bot=True))
    )
    assert not agent._channel.sent


@pytest.mark.asyncio
async def test_dispatcher_ignores_messages_from_other_channels() -> None:
    agent = make_agent()
    other = FakeChannel(id=777)
    await agent.on_message_for_test(FakeMessage(content="/status", channel=other))
    assert not other.sent


# ── extra-QA loop guard ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extra_qa_loop_terminates_on_done() -> None:
    from src.agent.memory_wizard import MemoryWizard

    replies = ["question | answer", "done"]
    data = {"answers": []}

    async def ask(q, meta):
        return replies.pop(0) if replies else "done"

    async def log(t):
        pass

    w = MemoryWizard(ask=ask, log=log)
    await w._extra_qa(data)
    assert len(data["answers"]) == 1
    assert data["answers"][0]["question"] == "question"


@pytest.mark.asyncio
async def test_extra_qa_loop_breaks_on_two_unparseable() -> None:
    from src.agent.memory_wizard import MemoryWizard

    data = {"answers": []}

    async def ask(q, meta):
        return "not-a-pipe"

    async def log(t):
        pass

    w = MemoryWizard(ask=ask, log=log)
    await w._extra_qa(data)
    assert data["answers"] == []


@pytest.mark.asyncio
async def test_memory_button_value_map() -> None:
    assert _MEMORY_BUTTON_VALUES["mem_yes"] == "Yes"
    assert _MEMORY_BUTTON_VALUES["mem_no"] == "No"
    assert _MEMORY_BUTTON_VALUES["mem_pnta"] == "Prefer not to answer"
    assert _MEMORY_BUTTON_VALUES["mem_skip"] == "skip"
    assert _MEMORY_BUTTON_VALUES["mem_done"] == "done"


# ── relation graph prettify ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_relation_line_unwraps_provenance(monkeypatch) -> None:
    """Relation graph must render provenance-wrapped node names as plain text."""
    from src.agent import discord_agent

    class FakeNode:
        def __init__(self, nid: str, ntype: str, name_value):
            self.id = nid
            self.node_type = type("T", (), {"__str__": lambda s: ntype})()
            self.data = {"name": name_value}

    class FakeGraph:
        @classmethod
        async def create(cls):
            return cls()

        async def close(self):
            pass

        async def get_local_graph(self, node_id, radius=1):
            return {
                "nodes": [
                    FakeNode("company:glean", "company", "glean"),
                    FakeNode(
                        "founder:x",
                        "founder",
                        {"value": "Arvind Jain", "source": "radar", "confidence": 0.5},
                    ),
                    FakeNode("uses:neo4j", "technology", {"value": "neo4j", "source": "radar"}),
                ],
                "edges": [],
            }

    monkeypatch.setattr("src.graph.graph_store.GraphStore", FakeGraph)
    line = await discord_agent._relation_line("glean")
    assert line is not None
    assert "{'value'" not in line
    assert "Arvind Jain" in line
    assert "neo4j" in line


@pytest.mark.asyncio
async def test_relation_line_skips_self_and_returns_none(monkeypatch) -> None:
    from src.agent import discord_agent

    class FakeNode:
        def __init__(self, nid: str, ntype: str, name_value):
            self.id = nid
            self.node_type = type("T", (), {"__str__": lambda s: ntype})()
            self.data = {"name": name_value}

    class FakeGraph:
        @classmethod
        async def create(cls):
            return cls()

        async def close(self):
            pass

        async def get_local_graph(self, node_id, radius=1):
            return {"nodes": [FakeNode("company:glean", "company", "glean")], "edges": []}

    monkeypatch.setattr("src.graph.graph_store.GraphStore", FakeGraph)
    assert await discord_agent._relation_line("glean") is None


# ── persona button + thread ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persona_button_has_stable_custom_id(monkeypatch, tmp_path) -> None:
    from src.agent.discord_agent import _PersonaButton

    # The button must carry custom_id="persona" so on_interaction routes it
    # even after the view is gone from memory (bot restart).
    view = _PersonaButton(make_agent())
    persona_btn = view.children[0]
    assert persona_btn.custom_id == "persona"


@pytest.mark.asyncio
async def test_send_full_persona_creates_thread(monkeypatch, tmp_path) -> None:
    p = tmp_path / "persona.json"
    p.write_text(
        json.dumps(
            {
                "name": "Harsh Sahu",
                "identity": {"website": "https://przknv.cc"},
                "answers": [{"category": "identity", "question": "Location", "answer": "India"}],
                "resume_summary": "TypeScript/Rust engineer",
            }
        )
    )
    monkeypatch.setattr("src.agent.memory_wizard.PERSONA_JSON", p)

    agent = make_agent()
    thread_sent: list[str] = []

    class FakeThreadObj:
        async def send(self, content=None, **kw):
            thread_sent.append(content or "")

    fake_thread = FakeThreadObj()

    # The main channel's send() returns a starter message that can create a
    # thread (the thread must be created from a FRESH message, not the
    # button's message — that's the regression this guards against).
    class FakeStarterMessage:
        async def create_thread(self, name, auto_archive_duration):
            return fake_thread

    class FakePersonaChannel:
        def __init__(self):
            self.sent = []

        async def send(self, content=None, **kw):
            self.sent.append((content, kw))
            return FakeStarterMessage()

    agent._channel = FakePersonaChannel()  # type: ignore[assignment]

    class FakeInter:
        def __init__(self):
            self.message = None
            self.response = FakeResponse()
            self.type = 2
            self.data = {"custom_id": "persona"}

    await agent._send_full_persona(FakeInter())
    assert thread_sent, "persona should be posted to the created thread"
    combined = "".join(thread_sent)
    assert "Harsh Sahu" in combined
    assert "przknv.cc" in combined
    assert "TypeScript/Rust engineer" in combined


@pytest.mark.asyncio
async def test_send_full_persona_no_persona_tells_user(monkeypatch, tmp_path) -> None:
    p = tmp_path / "persona.json"
    p.write_text(json.dumps({"name": "", "identity": {}, "answers": []}))
    monkeypatch.setattr("src.agent.memory_wizard.PERSONA_JSON", p)

    agent = make_agent()

    class FakeMsgForThread:
        async def create_thread(self, name, auto_archive_duration):
            raise AssertionError("thread should not be created without persona")

    class FakeInter:
        def __init__(self):
            self.message = FakeMsgForThread()
            self.response = FakeResponse()
            self.data = {"custom_id": "persona"}

    sent = {}

    class R:
        async def send_message(self, content=None, ephemeral=None, **kw):
            sent["content"] = content or ""

    inter = FakeInter()
    inter.response = R()
    await agent._send_full_persona(inter)
    assert "No persona" in sent.get("content", "")


# ── thread precedence ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_to_thread_prefers_sweep_thread(monkeypatch) -> None:
    agent = make_agent()
    recovery = FakeChannel(id=11, thread=True)
    sweep = FakeChannel(id=22, thread=True)
    agent._run_thread = recovery  # type: ignore[assignment]
    agent._sweep_thread = sweep  # type: ignore[assignment]

    ok = await agent._send_to_thread(content="sweep msg")
    assert ok is True
    assert any((c or "") == "sweep msg" for c, _ in sweep.sent)
    assert not any((c or "") == "sweep msg" for c, _ in recovery.sent)


@pytest.mark.asyncio
async def test_send_to_thread_falls_back_to_recovery_thread() -> None:
    agent = make_agent()
    recovery = FakeChannel(id=11, thread=True)
    agent._run_thread = recovery  # type: ignore[assignment]

    ok = await agent._send_to_thread(content="recovery msg")
    assert ok is True
    assert any((c or "") == "recovery msg" for c, _ in recovery.sent)


@pytest.mark.asyncio
async def test_send_to_thread_no_thread_uses_channel() -> None:
    agent = make_agent()
    agent._run_thread = None
    agent._sweep_thread = None
    ok = await agent._send_to_thread(content="main msg")
    assert ok is True
    assert any((c or "") == "main msg" for c, _ in agent._channel.sent)


@pytest.mark.asyncio
async def test_begin_recovery_thread_sets_run_thread() -> None:
    agent = make_agent()
    thread = FakeThread(id=888)
    agent._channel = FakeServerChannel(thread=thread)
    await agent.begin_recovery_thread()
    assert agent._run_thread is thread
    # The embed title lives in kwargs, not content.
    assert any(kw.get("embed") is not None for _, kw in agent._channel.sent)


# ── native slash commands ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slash_adapter_bridges_to_handler(monkeypatch) -> None:
    """_slash_bridge must ack the interaction then run the handler."""

    agent = make_agent()
    handled = []

    async def fake_status(message):
        handled.append(message.content)
        assert message.channel == agent._channel
        await agent._reply(message, "ok")

    class FakeSlashInteraction:
        channel = agent._channel

        class Response:
            @staticmethod
            async def defer():
                pass

        async def followup_send(self, text):
            pass

    await agent._slash_bridge(FakeSlashInteraction(), "/status", fake_status)
    assert handled == ["/status"]
    assert any((c or "") == "ok" for c, _ in agent._channel.sent)


@pytest.mark.asyncio
async def test_slash_bridge_failure_notifies() -> None:
    agent = make_agent()
    msgs = []

    async def boom(message):
        raise RuntimeError("nope")

    class FakeFollowup:
        async def send(self, text):
            msgs.append(text)

    class FakeSlashInteraction:
        channel = agent._channel
        followup = FakeFollowup()

        class Response:
            @staticmethod
            async def defer():
                pass

    await agent._slash_bridge(FakeSlashInteraction(), "/x", boom)
    assert msgs and "Command failed" in msgs[0]


@pytest.mark.asyncio
async def test_sync_app_commands_builds_tree(monkeypatch) -> None:
    """_sync_app_commands registers all native commands and syncs to guild."""

    agent = make_agent()
    agent._guild_id = "12345"

    # Build a fake tree that records commands + sync call.
    class FakeTree:
        def __init__(self):
            self.commands = []
            self.synced_to = None

        def command(self, name=None, description=None):
            def deco(fn):
                self.commands.append(name)
                return fn

            return deco

        async def sync(self, guild=None):
            self.synced_to = guild
            return [object() for _ in self.commands]

    fake = FakeTree()
    agent._tree = fake  # type: ignore[assignment]
    await agent._sync_app_commands()

    for name in ("memory", "persona", "status", "health", "analytics", "resend", "help"):
        assert name in fake.commands, f"missing {name}"
    assert fake.synced_to is not None
    assert fake.synced_to.id == 12345


@pytest.mark.asyncio
async def test_sync_app_commands_global_without_guild(monkeypatch) -> None:
    agent = make_agent()
    agent._guild_id = None

    class FakeTree:
        def __init__(self):
            self.commands = []
            self.synced_to = "UNSET"

        def command(self, name=None, description=None):
            def deco(fn):
                self.commands.append(name)
                return fn

            return deco

        async def sync(self, guild=None):
            self.synced_to = guild
            return []

    fake = FakeTree()
    agent._tree = fake  # type: ignore[assignment]
    await agent._sync_app_commands()
    assert fake.synced_to is None  # global sync


@pytest.mark.asyncio
async def test_slash_memory_builds_content() -> None:

    agent = make_agent()
    captured = []

    async def fake_handle_memory(message):
        captured.append(message.content)

    agent._handle_memory = fake_handle_memory  # type: ignore[method-assign]

    class FakeSlashInteraction:
        channel = agent._channel

        class Response:
            @staticmethod
            async def defer():
                pass

    await agent._slash_memory(FakeSlashInteraction(), "update this")
    assert captured == ["/memory update this"]


@pytest.mark.asyncio
async def test_slash_persona_builds_content() -> None:
    agent = make_agent()
    captured = []

    async def fake_handle_persona(message):
        captured.append(message.content)

    agent._handle_persona = fake_handle_persona  # type: ignore[method-assign]

    class FakeSlashInteraction:
        channel = agent._channel

        class Response:
            @staticmethod
            async def defer():
                pass

    await agent._slash_persona(FakeSlashInteraction())
    assert captured == ["/persona"]


# ── sweep intro summary ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_begin_sweep_posts_intro_summary(monkeypatch) -> None:
    """begin_sweep must post a queue-state summary as the thread's second message."""
    agent = make_agent()
    thread = FakeThread(id=700)
    agent._channel = FakeServerChannel(thread=thread)

    async def fake_queue_summary():
        return {
            "applied": 3,
            "pending": 5,
            "filling": 2,
            "deferred": 1,
            "awaiting_review": 0,
            "failed": 1,
        }

    monkeypatch.setattr(agent, "_queue_summary_map", fake_queue_summary)
    await agent.begin_sweep(1)
    assert agent._sweep_thread is thread
    # starter message (embed) + intro summary embed
    assert len(thread.sent) >= 1
    # The intro summary was sent to the thread
    thread_embeds = [kw.get("embed") for _, kw in thread.sent]
    assert any(e is not None and "Starting Point" in str(e.title) for e in thread_embeds)
    assert any(e is not None and "Applied" in str(e.description) for e in thread_embeds)


@pytest.mark.asyncio
async def test_begin_sweep_skips_duplicate_thread(monkeypatch) -> None:
    """Same sweep number must not re-create the thread."""
    agent = make_agent()
    thread = FakeThread(id=701)
    agent._channel = FakeServerChannel(thread=thread)
    await agent.begin_sweep(1)
    first_count = len(agent._channel.sent)
    await agent.begin_sweep(1)
    assert len(agent._channel.sent) == first_count


@pytest.mark.asyncio
async def test_send_sweep_summary_posts_complete_embed(monkeypatch) -> None:
    agent = make_agent()
    thread = FakeThread(id=702)
    agent._channel = FakeServerChannel(thread=thread)
    await agent.begin_sweep(2)
    assert agent._sweep_thread is thread

    async def fake_queue_summary():
        return {
            "applied": 4,
            "pending": 3,
            "filling": 1,
            "deferred": 2,
            "awaiting_review": 1,
            "failed": 0,
        }

    monkeypatch.setattr(agent, "_queue_summary_map", fake_queue_summary)
    await agent.send_sweep_summary(2, matched=7, scraped=90, duration=12.5)
    embeds = [kw.get("embed") for _, kw in thread.sent]
    complete = [e for e in embeds if e is not None and "Complete" in str(e.title)]
    assert complete, "expected a Sweep Complete embed"
    desc = str(complete[0].description)
    assert "Scraped" in desc and "7" in desc and "12.5" in desc
    assert "Applied" in desc
