"""DiscordAgent: single Discord gateway for pipeline alerts, sweep summaries,
command handling, and routing autofill question answers.

One gateway per process (the ingest radar process owns it in loop mode; a
standalone CLI owns its own). The autofill bridge never opens a gateway — it
sends questions over the REST API and reads answers from the shared
``discord_question_mailbox`` table that this agent's event handlers populate,
so two gateway clients never race on the same bot token.

Commands (send in the configured channel):
    /memory    - update resume + persona memory (Q&A runs in a thread)
    /persona   - view the stored persona
    /status    - current pipeline state (sweep, matched jobs, LLM status)
    /health    - runs live health checks on all services
    /analytics - market intelligence & skill arbitrage report
    /resend    - resend accepted job matches (usage: /resend [--dry] [limit])
    /help      - lists available commands
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
from typing import TYPE_CHECKING, Any

import discord
from src.logging import get_logger

if TYPE_CHECKING:
    from src.llm.context import ContextManager

logger = get_logger("discord_agent")

_pipeline_state: dict[str, Any] = {
    "running": False,
    "sweep": 0,
    "phase": "idle",
    "matched_total": 0,
    "rejected_total": 0,
    "last_error": None,
    "started_at": None,
    "sweep_started_at": 0.0,
    "llm_tokens_used": 0,
    "scraped_count": 0,
}


def set_pipeline_state(**kwargs: Any) -> None:
    _pipeline_state.update(kwargs)


_MEMORY_BUTTON_LABELS: dict[str, str] = {
    "yes": "Yes",
    "no": "No",
    "pnta": "Prefer not to answer",
    "skip": "Skip",
    "done": "Done",
}
_MEMORY_BUTTON_VALUES: dict[str, str] = {
    "mem_yes": "Yes",
    "mem_no": "No",
    "mem_pnta": "Prefer not to answer",
    "mem_skip": "skip",
    "mem_done": "done",
}


class _SlashAdapter:
    """Minimal stand-in for a ``discord.Message`` so native slash commands can
    reuse the existing ``_handle_*`` text-command handlers.

    The handlers only touch ``content`` and ``channel`` — this adapter provides
    exactly those, plus the ``reply_to`` the mailbox capture expects (None).
    """

    def __init__(self, content: str, channel: Any) -> None:
        self.content = content
        self.channel = channel
        self.reference = None


class _MemoryWizardSession:
    """Runs one /memory Q&A session inside a thread.

    The wizard asks a question, this posts it (with quick-answer buttons) and
    blocks on a future that the gateway resolves when a message or button
    arrives in the thread. Timeout returns None so the wizard skips the
    question instead of hanging forever.
    """

    _TIMEOUT = 600.0

    def __init__(self, thread: discord.Thread, parent: DiscordAgent) -> None:
        self.thread = thread
        self.parent = parent
        self.pending: asyncio.Future[str | None] | None = None

    async def log(self, text: str) -> None:
        with contextlib.suppress(Exception):
            await self.thread.send(text)

    @staticmethod
    def _build_view(buttons: list[str]) -> discord.ui.View | None:
        if not buttons:
            return None
        view = discord.ui.View(timeout=None)
        for key in buttons:
            label = _MEMORY_BUTTON_LABELS.get(key, key)
            if key == "yes":
                style = discord.ButtonStyle.success
            elif key == "done":
                style = discord.ButtonStyle.primary
            else:
                style = discord.ButtonStyle.secondary
            view.add_item(
                discord.ui.Button(
                    label=label,
                    custom_id=f"mem_{key}",
                    style=style,
                )
            )
        return view

    async def ask(self, question: str, meta: dict[str, Any]) -> str | None:
        buttons = meta.get("buttons") or []
        hint = meta.get("hint")
        text = question
        if hint:
            text += f"\n*(current: {hint})*"
        view = self._build_view(buttons)
        kwargs: dict[str, Any] = {}
        if view is not None:
            kwargs["view"] = view
        # Arm the future BEFORE posting so a reply racing the send is never
        # dropped — the gateway resolves it as soon as it arrives.
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[str | None] = loop.create_future()
        self.pending = fut
        try:
            with contextlib.suppress(Exception):
                await self.thread.send(text, **kwargs)
            return await asyncio.wait_for(fut, timeout=self._TIMEOUT)
        except TimeoutError:
            await self.log("⏰ No answer in 10 minutes — skipping this question.")
            return None
        finally:
            self.pending = None


async def autofill_queue_lines() -> list[str]:
    """Autofill queue status lines (applied / remaining / review / failed)."""
    try:
        from autofill.src.core.db import AutofillDB

        db = await AutofillDB.create()
        try:
            s = await db.queue_summary()
        finally:
            await db.close()
    except Exception:
        return []
    need_review = s.get("deferred", 0) + s.get("awaiting_review", 0)
    remaining = s.get("pending", 0) + s.get("filling", 0)
    return [
        f"**Applied:** {s.get('applied', 0)}",
        f"**Remaining:** {remaining}",
        f"**Need Review:** {need_review}",
        f"**Failed:** {s.get('failed', 0)}",
        f"**Open:** {s.get('open', 0)}",
    ]


_CATEGORY_ICONS: dict[str, str] = {
    "urgent": "🚨",
    "startup_signal": "📡",
    "outreach": "🤝",
    "eligible": "✅",
    "review": "🔎",
    "general_accepted": "🎯",
}

_CATEGORY_LABELS: dict[str, str] = {
    "urgent": "Urgent High-Fit Verified Role",
    "startup_signal": "Startup Hiring Signal",
    "outreach": "Cold Outreach Opportunity",
    "eligible": "Eligible Role",
    "review": "Freshness Review Role",
    "general_accepted": "Matched Role",
}


async def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        return True
    except Exception:
        return False


async def _check_http(url: str, timeout: float = 3.0) -> bool:
    try:
        from src.http_client import get_client

        client = await get_client("discord_agent", timeout=timeout)
        resp = await client.get(url)
        return resp.status_code < 500
    except Exception:
        return False


def get_system_metrics() -> dict[str, str]:
    mem_str = "N/A"
    disk_str = "N/A"
    cpu_str = "N/A"
    try:
        with open("/proc/meminfo") as f:
            lines = {
                line.split(":")[0]: int(line.split(":")[1].split()[0]) for line in f if ":" in line
            }
            total = lines.get("MemTotal", 0)
            avail = lines.get("MemAvailable", 0)
            pct = round((total - avail) / total * 100, 1) if total > 0 else 0.0
            mem_str = (
                f"{round((total - avail) / (1024 * 1024), 1)}GB / "
                f"{round(total / (1024 * 1024), 1)}GB ({pct}%)"
            )
    except Exception:
        pass
    try:
        du = shutil.disk_usage("/")
        pct = round(du.used / du.total * 100, 1)
        disk_str = (
            f"{round(du.used / (1024**3), 1)}GB / {round(du.total / (1024**3), 1)}GB ({pct}%)"
        )
    except Exception:
        pass
    try:
        load1, load5, _ = os.getloadavg()
        cpu_str = f"{load1:.2f} (1m), {load5:.2f} (5m)"
    except Exception:
        pass
    return {"ram": mem_str, "disk": disk_str, "cpu_load": cpu_str}


async def run_health_checks() -> str:
    checks = [
        ("llama-server (Embed)", _check_http("http://localhost:8900/health")),
        ("SearXNG", _check_http("http://localhost:8080")),
        ("agent-memory-db (pgvector)", _check_port("localhost", 5433)),
        ("Neo4j Graph Store", _check_port("localhost", 7687)),
    ]
    results = await asyncio.gather(*[coro for _, coro in checks])
    lines = ["**System Health Check**", ""]
    all_ok = True
    for (name, _), ok in zip(checks, results, strict=True):
        tag = "✅" if ok else "❌"
        lines.append(f"{tag} {name}")
        if not ok:
            all_ok = False
    lines.append("")
    lines.append(
        "All infrastructure services healthy." if all_ok else "[WARNING] Some services down."
    )
    return "\n".join(lines)


def _color_for(category: str) -> int:
    colors = {
        "urgent": 0xEF5350,
        "startup_signal": 0x42A5F5,
        "outreach": 0xAB47BC,
        "eligible": 0x66BB6A,
        "review": 0xFFA726,
        "general_accepted": 0x26C6DA,
    }
    return colors.get(category, 0x607D8B)


def _job_salary_line(job: dict[str, Any]) -> str:
    """Salary string with a confirmed/estimated tag + source."""
    raw = str(job.get("salary") or "").strip()
    if raw and raw not in ("-", "Not specified", "N/A", "Flexible", "Competitive"):
        sal = raw
    elif job.get("salary_annual_usd"):
        sal = f"${job['salary_annual_usd']:,.0f}/yr"
    else:
        sal = "-"
    if job.get("salary_estimated"):
        src = str(job.get("salary_source") or "").strip()
        tag = f"· est. ({src})" if src else "· est."
    else:
        tag = "· confirmed"
    return f"{sal} {tag}"


def _funding_line(job: dict[str, Any]) -> str | None:
    """'Recently raised $X (Round, Date) - led by ...' from funding_info."""
    fi = job.get("funding_info") or {}
    if not isinstance(fi, dict):
        fi = {}
    round_ = str(fi.get("round") or job.get("funding_stage") or "").strip()
    amount = str(fi.get("amount_raised") or "").strip()
    date = str(fi.get("date_announced") or "").strip()
    leads = fi.get("lead_investors") or []
    if isinstance(leads, list):
        leads = [str(x) for x in leads if x]
    parts = []
    if amount:
        parts.append(f"raised {amount}")
    if round_:
        parts.append(round_)
    if date:
        parts.append(date)
    if not parts:
        return None
    line = "Recently " + " ".join(parts)
    if leads:
        line += f" — led by {', '.join(leads[:3])}"
    return line


async def _relation_line(company: str | None) -> str | None:
    """One-line relation graph for the job's company from Neo4j."""
    if not company:
        return None
    try:
        from src.graph.entity import _unwrap_prov, make_company_id
        from src.graph.graph_store import GraphStore

        graph = await GraphStore.create()
        try:
            local = await graph.get_local_graph(make_company_id(company), radius=1)
            parts: list[str] = []
            for node in local.get("nodes", []):
                nd = node.data or {}
                name = str(_unwrap_prov(nd.get("name")) or node.id or "").strip()
                if not name or name == company:
                    continue
                kind = str(node.node_type)
                if kind == "founder":
                    parts.append(f"founder: {name}")
                elif kind == "career_site":
                    parts.append(f"site: {name}")
                elif kind == "technology":
                    parts.append(f"uses: {name}")
                elif kind == "hiring_post" or kind == "job":
                    parts.append(f"hiring: {name}")
                elif kind == "investor":
                    parts.append(f"investor: {name}")
                else:
                    parts.append(f"{kind}: {name}")
            return " | ".join(parts[:6]) if parts else None
        finally:
            await graph.close()
    except Exception:
        return None


async def _build_job_embed(category: str, job: dict[str, Any]) -> discord.Embed:
    role = str(job.get("role") or "Software Engineer").strip()
    company = str(job.get("company") or "Company").strip()
    icon = _CATEGORY_ICONS.get(category, "📌")
    label = _CATEGORY_LABELS.get(category, category)

    embed = discord.Embed(
        title=f"{icon} {label}: {role}",
        color=_color_for(category),
        description=str(job.get("company_description") or job.get("jd_summary") or "")[:400]
        or None,
    )

    match = job.get("match_percent")
    shortlist = job.get("shortlist_probability")
    underdog = job.get("underdog_score")
    embed.add_field(name="Company", value=company, inline=True)
    embed.add_field(name="Location", value=str(job.get("location") or "Remote"), inline=True)
    fit = f"{match}% match · {shortlist}% shortlist" if match is not None else "-"
    if underdog:
        fit += f" · underdog {underdog}"
    embed.add_field(name="Fit", value=fit, inline=True)
    if category == "outreach":
        funding_line = _funding_line(job)
        if funding_line:
            embed.add_field(name="Funding", value=funding_line, inline=False)
        elif _job_salary_line(job) != "- confirmed":
            embed.add_field(name="Salary", value=_job_salary_line(job), inline=False)
    else:
        embed.add_field(name="Salary", value=_job_salary_line(job), inline=False)

    warnings: list[str] = []
    from src.radar.core.signals import is_us_location

    loc_raw = str(job.get("location") or "Remote").strip()
    is_remote_role = bool(job.get("is_remote")) or "remote" in loc_raw.lower()
    is_us = is_us_location(loc_raw)
    if not is_remote_role:
        warnings.append("⚠️ Onsite role - requires visa/relocation, may be rejected")
    if is_us and not job.get("sponsors_visa"):
        warnings.append("⚠️ US role - visa sponsorship not confirmed")
    if job.get("osint_signals"):
        sigs = job["osint_signals"]
        if isinstance(sigs, list):
            for sig in sigs[:2]:
                txt = sig.get("text") if isinstance(sig, dict) else str(sig)
                if txt:
                    warnings.append(f"📡 {txt[:120]}")
    if warnings:
        embed.add_field(name="Warnings", value="\n".join(warnings), inline=False)

    badges: list[str] = []
    if job.get("sponsors_visa"):
        badges.append("visa sponsor")
    if job.get("is_remote") or "remote" in loc_raw.lower():
        badges.append("remote")
    if job.get("funding_stage"):
        badges.append(str(job["funding_stage"]))
    if badges:
        embed.add_field(name="Tags", value=", ".join(badges), inline=False)

    founders = job.get("founders") or []
    if isinstance(founders, list) and founders:
        names = []
        for f in founders[:3]:
            if isinstance(f, dict):
                nm = f.get("name") or ""
                url = f.get("linkedin_url") or ""
                names.append(f"[{nm}]({url})" if url and nm else nm)
        if names:
            embed.add_field(name="Founders", value=", ".join(names), inline=False)

    skills = job.get("matching_skills") or []
    if isinstance(skills, list) and skills:
        embed.add_field(
            name="Matching skills", value=", ".join(str(s) for s in skills[:8]), inline=False
        )

    link = job.get("apply_link") or job.get("direct_apply_url") or job.get("url") or ""
    if str(link).startswith("http"):
        embed.add_field(name="Apply", value=str(link), inline=False)

    relation = await _relation_line(company)
    if relation:
        embed.add_field(name="Relation graph", value=relation, inline=False)
    return embed


class _PersonaButton(discord.ui.View):
    """Button on the startup message that posts the full candidate persona.

    The button press posts the complete persona (identity + answers + resume
    summary) as a message in the main channel — outside any sweep thread.
    """

    def __init__(self, agent: DiscordAgent) -> None:
        super().__init__(timeout=None)
        self.agent = agent

    @discord.ui.button(
        label="📄 View Full Persona",
        style=discord.ButtonStyle.secondary,
        custom_id="persona",
    )
    async def persona(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        await self.agent._send_full_persona(interaction)

    @discord.ui.button(label="Analytics", style=discord.ButtonStyle.primary, custom_id="analytics")
    async def analytics(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        await self.agent._send_analytics_report(interaction)

    @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger, custom_id="stop_pipeline")
    async def stop_pipeline(
        self, interaction: discord.Interaction, _button: discord.ui.Button[Any]
    ) -> None:
        await self.agent._stop_pipeline(interaction)


class DiscordAgent:
    """Discord gateway client for pipeline notifications and commands."""

    def __init__(
        self,
        bot_token: str | None = None,
        channel_id: str | None = None,
        ctx: ContextManager | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token if bot_token is not None else os.getenv("DISCORD_BOT_TOKEN", "")
        self.channel_id = (
            (channel_id or chat_id)
            if (channel_id or chat_id) is not None
            else os.getenv("DISCORD_CHANNEL_ID", "")
        )
        self.ctx = ctx
        self._notified_keys: set[str] = set()
        self._seen_errors: set[str] = set()
        self._mailbox_db: Any | None = None
        self._client: discord.Client | None = None
        self._channel: discord.abc.Messageable | None = None
        self._run_thread: discord.Thread | None = None
        self._sweep_thread: discord.Thread | None = None
        self._last_sweep: int = 0
        self._guild_id: str | None = os.getenv("DISCORD_GUILD_ID") or None
        self._poll_task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._shutdown_callback: Any | None = None
        self._memory_sessions: dict[int, _MemoryWizardSession] = {}
        self._tree: discord.app_commands.CommandTree | None = None

    def set_shutdown_callback(self, cb: Any) -> None:
        """Register a callable invoked when the /stop button is pressed."""
        self._shutdown_callback = cb

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.channel_id)

    # ── gateway lifecycle ──────────────────────────────────────────────

    async def start_polling(self) -> None:
        if not self.is_configured or self._client is not None:
            return
        intents = discord.Intents.default()
        intents.message_content = True
        import aiohttp

        connector = aiohttp.TCPConnector(ssl=False)
        client = discord.Client(
            intents=intents,
            max_messages=2000,
            connector=connector,
        )
        self._client = client
        self._tree = discord.app_commands.CommandTree(client)

        async def on_ready() -> None:
            logger.info("DiscordAgent connected", bot=client.user)
            self._ready.set()
            # Fetch the channel lazily in the background — never block on_ready
            # on a REST call that could hang and delay every first message.
            with contextlib.suppress(Exception):
                if self.channel_id:
                    self._channel = await client.fetch_channel(int(self.channel_id))  # type: ignore[assignment]
            await self._sync_app_commands()

        async def on_message(message: discord.Message) -> None:
            if message.author.bot:
                return
            # Only the main channel AND active memory threads are watched.
            if (
                self._channel is not None
                and message.channel != self._channel
                and not self._is_memory_thread(message.channel)
            ):
                return
            await self._heartbeat_poller()
            if self._is_memory_answer(message):
                return
            await self._capture_mailbox_answer(message)
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

        async def on_interaction(interaction: discord.Interaction) -> None:
            await self._heartbeat_poller()
            if interaction.type != discord.InteractionType.component:
                return
            custom_id = ""
            if interaction.data:
                custom_id = str(interaction.data.get("custom_id") or "")
            message_id = interaction.message.id if interaction.message else None
            if custom_id == "persona" and message_id is not None:
                with contextlib.suppress(Exception):
                    await self._send_full_persona(interaction)
                return
            if custom_id == "analytics" and message_id is not None:
                with contextlib.suppress(Exception):
                    await self._send_analytics_report(interaction)
                return
            if custom_id == "stop_pipeline" and message_id is not None:
                with contextlib.suppress(Exception):
                    await self._stop_pipeline(interaction)
                return
            if custom_id.startswith("mem_") and message_id is not None:
                with contextlib.suppress(Exception):
                    await self._memory_button(custom_id, interaction)
                return
            if message_id is not None:
                await self._capture_mailbox_button(message_id, custom_id, interaction)

        client.event(on_ready)
        client.event(on_message)
        client.event(on_interaction)
        self._poll_task = asyncio.create_task(self._client.start(self.bot_token))

        # Wait for readiness WITHOUT the Event (which proved unreliable here):
        # poll client.is_ready() until the gateway reports connected.
        for _ in range(60):
            if self._client.is_ready():
                break
            await asyncio.sleep(1)
        if not self._client.is_ready():
            logger.warning("Discord gateway not ready after 60s")

    async def stop_polling(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._channel = None

    async def _wait_channel(self) -> discord.abc.Messageable | None:
        if self._channel is not None:
            return self._channel
        if self._client is not None and self._client.is_ready():
            with contextlib.suppress(Exception):
                if self.channel_id:
                    self._channel = await self._client.fetch_channel(int(self.channel_id))  # type: ignore[assignment]
        return self._channel

    # ── sending ────────────────────────────────────────────────────────

    async def _send(
        self,
        content: str = "",
        embed: discord.Embed | None = None,
        view: discord.ui.View | None = None,
    ) -> bool:
        channel = await self._wait_channel()
        if channel is None:
            return False
        try:
            kwargs: dict[str, Any] = {}
            if embed is not None:
                kwargs["embed"] = embed
            if view is not None:
                kwargs["view"] = view
            await channel.send(content=content or None, **kwargs)
            return True
        except Exception as e:
            logger.warning("Discord send failed", error=str(e))
            return False

    async def _send_to_thread(self, content: str = "", embed: discord.Embed | None = None) -> bool:
        # Prefer the sweep thread once a sweep is live; fall back to the
        # recovery/run thread so pre-sweep re-deliveries still land in a thread.
        thread = self._sweep_thread or self._run_thread
        if thread is None:
            return await self._send(content=content, embed=embed)
        try:
            kwargs: dict[str, Any] = {}
            if embed is not None:
                kwargs["embed"] = embed
            await thread.send(content=content or None, **kwargs)
            return True
        except Exception as e:
            logger.warning("Discord thread send failed", error=str(e))
            return False

    async def send_error(self, message: str, dedup_key: str = "") -> None:
        if dedup_key and dedup_key in self._seen_errors:
            return
        if dedup_key:
            self._seen_errors.add(dedup_key)
        embed = discord.Embed(title="⚠ Pipeline Error", color=0xEF5350, description=message[:3900])
        await self._send(embed=embed)

    async def send_startup(self, sweep_count: int = 0) -> None:
        if not self.is_configured:
            return

        # Pipeline-wide stats: memory (pgvector), sources, candidates, queue.
        resume_chunks = sweep_count
        persona_chunks = 0
        sources = 0
        candidates = 0
        queue_stats: dict[str, int] = {}
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                persona_chunks = await store.persona_chunk_count()
                async with store._pool.acquire() as conn:
                    sources = await conn.fetchval("SELECT COUNT(*) FROM source_checkpoints") or 0
                    candidates = await conn.fetchval("SELECT COUNT(*) FROM radar_candidates") or 0
            finally:
                await store.close()
        except Exception:
            pass
        try:
            from autofill.src.core.db import AutofillDB

            db = await AutofillDB.create()
            try:
                queue_stats = await db.unapplied_stats()
            finally:
                await db.close()
        except Exception:
            pass

        summary = await self._queue_summary_map()
        applied = summary.get("applied", 0)
        remaining = summary.get("pending", 0) + summary.get("filling", 0)
        review = summary.get("deferred", 0) + summary.get("awaiting_review", 0)
        failed = summary.get("failed", 0)
        unapplied = queue_stats.get("unapplied", 0)
        stale = queue_stats.get("stale", 0)

        total_mem = resume_chunks + persona_chunks
        description = (
            f"**Memory** — Resume `{resume_chunks}` · Persona `{persona_chunks}` · "
            f"Total `{total_mem}`\n"
            f"**Corpus** — `{candidates:,}` candidates · `{sources:,}` sources\n"
            f"**Queue** — Applied `{applied}` · Remaining `{remaining}` · "
            f"Review `{review}` · Failed `{failed}`\n"
            f"**Unapplied** — `{unapplied}` (`{stale}` stale > 48h)\n"
            f"**Scheduler** — 8 async workers"
        )
        embed = discord.Embed(title="🚀 Pipeline Started", color=0x66BB6A, description=description)
        await self._send(embed=embed, view=_PersonaButton(self))

    async def _queue_summary_map(self) -> dict[str, int]:
        """Autofill queue summary counts (empty on error)."""
        try:
            from autofill.src.core.db import AutofillDB

            db = await AutofillDB.create()
            try:
                return await db.queue_summary()
            finally:
                await db.close()
        except Exception:
            return {}

    async def _send_analytics_report(self, interaction: discord.Interaction) -> None:
        with contextlib.suppress(Exception):
            await interaction.response.defer()
        channel = await self._wait_channel()
        if channel is None:
            return
        await channel.send("Crunching market data and calculating skill arbitrage...")
        queue_lines = await autofill_queue_lines()
        if queue_lines:
            await channel.send("**[QUEUE] Autofill Status**\n" + "\n".join(queue_lines))
        if self.ctx is None:
            await channel.send("Analytics unavailable (no LLM context).")
            return
        try:
            from src.agent.analytics_agent import AnalyticsAgent
            from src.graph.graph_store import GraphStore
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            graph = await GraphStore.create()
            try:
                agent = AnalyticsAgent(store=store, graph=graph, ctx=self.ctx)
                sections = await agent.generate_resilient_report()
            finally:
                await graph.close()
                await store.close()
            for section in sections:
                if section.strip():
                    await channel.send(section[:1900])
                    await asyncio.sleep(0.5)
        except Exception as e:
            await channel.send(f"Analytics failed: {e}")

    async def _stop_pipeline(self, interaction: discord.Interaction) -> None:
        with contextlib.suppress(Exception):
            await interaction.response.defer()
        set_pipeline_state(running=False, phase="stopping")
        channel = await self._wait_channel()
        if channel is not None:
            await channel.send("Stopping pipeline...")
        if self._shutdown_callback is not None:
            with contextlib.suppress(Exception):
                self._shutdown_callback()

    async def _send_full_persona(self, interaction: discord.Interaction) -> None:
        try:
            from src.agent.memory_wizard import format_persona

            persona = format_persona().strip()
        except Exception:
            persona = ""
        if (
            not persona
            or persona.startswith("No persona.json yet")
            or persona.startswith("_persona.json is empty")
        ):
            with contextlib.suppress(Exception):
                await interaction.response.send_message(
                    "No persona built yet — run `/memory` to set one up.", ephemeral=True
                )
            return
        # Ack the interaction, then post a fresh starter message and spin a
        # thread from it — creating a thread from the button's own message
        # fails with "A thread has already been created for this message".
        with contextlib.suppress(Exception):
            await interaction.response.defer()
        channel = await self._wait_channel()
        if channel is None:
            return
        try:
            starter = await channel.send("📄 **Full Persona** — identity, Q&A and resume summary.")
            thread = await starter.create_thread(
                name="Full Persona",
                auto_archive_duration=1440,
            )
        except Exception as e:
            logger.warning("Persona thread creation failed", error=str(e))
            with contextlib.suppress(Exception):
                await channel.send(f"```{persona[:3900]}```")
            return
        for chunk in re.findall(r".{1,1900}", persona, re.DOTALL) or [persona]:
            with contextlib.suppress(Exception):
                await thread.send(f"```{chunk}```")

    async def begin_sweep(self, sweep: int) -> None:
        """Create the per-sweep thread before alerts fire, so every message of
        a sweep (alerts + summary) is concentrated in one thread."""
        if not self.is_configured:
            return
        # If the gateway hasn't finished connecting yet, wait a bounded window
        # so the very first sweep's thread isn't silently dropped.
        if self._client is not None and not self._client.is_ready():
            try:
                await asyncio.wait_for(self._wait_gateway_ready(), timeout=20.0)
            except Exception:
                logger.warning("Discord gateway not ready; sweep thread skipped")
        channel = await self._wait_channel()
        if channel is None:
            logger.warning("Discord channel unavailable; sweep thread skipped")
            return
        try:
            if sweep != self._last_sweep:
                embed = discord.Embed(
                    title=f"🔄 Sweep #{sweep} Started",
                    color=0x42A5F5,
                    description=(
                        "Scanning sources, matching jobs, and filing applications — "
                        "this thread collects everything from this sweep."
                    ),
                )
                starter = await channel.send(embed=embed)
                self._sweep_thread = await starter.create_thread(
                    name=f"Sweep #{sweep}",
                    auto_archive_duration=1440,
                )
                # Record the thread id so the autofill bridge (a separate
                # process) can send deferred/captcha/queue notifications into
                # this thread instead of the main channel.
                with contextlib.suppress(Exception):
                    from autofill.src.core.db import AutofillDB

                    db = await AutofillDB.create()
                    try:
                        await db.set_active_thread(str(self._sweep_thread.id))
                    finally:
                        await db.close()
                # Second message: an opening summary of the current queue
                # state, so the thread opens with context before alerts land.
                try:
                    summary = await self._queue_summary_map()
                    applied = summary.get("applied", 0)
                    remaining = summary.get("pending", 0) + summary.get("filling", 0)
                    review = summary.get("deferred", 0) + summary.get("awaiting_review", 0)
                    failed = summary.get("failed", 0)
                    ml_line = await self._ml_status_line()
                    intro = discord.Embed(
                        title=f"📊 Sweep #{sweep} Starting Point",
                        color=0x42A5F5,
                        description=(
                            f"**Queue** — Applied `{applied}` · Remaining `{remaining}` · "
                            f"Review `{review}` · Failed `{failed}`\n"
                            f"{ml_line}\n"
                            "Watching for new matches — alerts land below as they're found."
                        ),
                    )
                    await self._sweep_thread.send(embed=intro)
                except Exception as e:
                    logger.debug("Sweep intro summary skipped", error=str(e))
                self._last_sweep = sweep
        except Exception as e:
            logger.warning("Sweep thread creation failed", error=str(e))

    async def _wait_gateway_ready(self) -> None:
        while self._client is not None and not self._client.is_ready():
            await asyncio.sleep(0.5)

    async def _ml_status_line(self) -> str:
        """One-line summary of the learning system state (bandits, Gmail push,
        LTR mode, event volume) pulled from Postgres. Best-effort."""
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                async with store._pool.acquire() as conn:
                    events = await conn.fetchval("SELECT COUNT(*) FROM decision_events")
                    impressions = await conn.fetchval("SELECT COUNT(*) FROM impressions")
                    outcomes = await conn.fetchval(
                        "SELECT COUNT(*) FROM decision_events WHERE reward IS NOT NULL"
                    )
                    unattributed = await conn.fetchval("SELECT COUNT(*) FROM unattributed_outcomes")
                    model = await conn.fetchrow(
                        "SELECT version, model_type FROM model_registry "
                        "WHERE status='active' ORDER BY created_at DESC LIMIT 1"
                    )
                    push = await conn.fetchval(
                        "SELECT 1 FROM gmail_push_state WHERE history_id != '' LIMIT 1"
                    )
                parts = [
                    f"Events `{events or 0}`",
                    f"Impressions `{impressions or 0}`",
                    f"Rewards `{outcomes or 0}`",
                ]
                if push:
                    parts.append("Gmail push `ON`")
                else:
                    parts.append("Gmail push `off`")
                if model:
                    parts.append(f"Model `{model['version']}` ({model['model_type']})")
                else:
                    parts.append("Model `none` (shadow)")
                if unattributed:
                    parts.append(f"Unattributed `{unattributed}`")
                return "🧠 **Learning system** — " + " · ".join(parts)
            finally:
                await store.close()
        except Exception:
            return "🧠 **Learning system** — unavailable"

    async def begin_recovery_thread(self, title: str = "Startup Re-delivery") -> None:
        """Create a thread for the startup re-delivery of previously-accepted
        roles, so those alerts live in a thread instead of the main channel."""
        channel = await self._wait_channel()
        if channel is None:
            return
        try:
            embed = discord.Embed(
                title=f"🔁 {title}",
                color=0x42A5F5,
                description=(
                    "Catching up on accepted roles you haven't been alerted "
                    "about yet — everything here lives in this thread."
                ),
            )
            starter = await channel.send(embed=embed)
            self._run_thread = await starter.create_thread(
                name=title,
                auto_archive_duration=1440,
            )
        except Exception as e:
            logger.warning("Recovery thread creation failed", error=str(e))

    async def send_sweep_summary(
        self, sweep: int, matched: int, scraped: int, duration: float
    ) -> None:
        await self.begin_sweep(sweep)
        thread = self._sweep_thread
        if thread is None:
            return
        try:
            summary = await self._queue_summary_map()
            unapplied_stats = {}
            try:
                from autofill.src.core.db import AutofillDB

                db = await AutofillDB.create()
                try:
                    unapplied_stats = await db.unapplied_stats()
                finally:
                    await db.close()
            except Exception:
                pass
            applied = summary.get("applied", 0)
            remaining = summary.get("pending", 0) + summary.get("filling", 0)
            review = summary.get("deferred", 0) + summary.get("awaiting_review", 0)
            failed = summary.get("failed", 0)
            unapplied = unapplied_stats.get("unapplied", 0)
            stale = unapplied_stats.get("stale", 0)

            ml_line = await self._ml_status_line()
            description = (
                f"**Scraped** `{scraped}` · **Matched** `{matched}` · "
                f"**Took** `{duration:.1f}s`\n"
                f"**Queue** — Applied `{applied}` · Remaining `{remaining}` · "
                f"Review `{review}` · Failed `{failed}`\n"
                f"**Unapplied** — `{unapplied}` (`{stale}` stale > 48h)\n"
                f"{ml_line}"
            )
            embed = discord.Embed(
                title=f"✅ Sweep #{sweep} Complete", color=0x66BB6A, description=description
            )
            await thread.send(embed=embed)
        except Exception as e:
            logger.warning("Sweep summary send failed", error=str(e))

    async def send_stage_progress(
        self, stage: str, summary: str, extra_metrics: dict[str, Any] | None = None
    ) -> None:
        lines = [f"**{stage}**", summary]
        if extra_metrics:
            lines.extend(f"{k}: {v}" for k, v in extra_metrics.items())
        await self._send_to_thread("\n".join(lines))

    async def send_categorized_alert(
        self, category: str, job: dict[str, Any], dedup_key: str = ""
    ) -> bool:
        if not self.is_configured:
            return False
        if dedup_key and dedup_key in self._notified_keys:
            return False
        embed = await _build_job_embed(category, job)
        ok = await self._send_to_thread(embed=embed)
        if ok and dedup_key:
            self._notified_keys.add(dedup_key)
        return ok

    async def send_notification(self, job: dict[str, Any]) -> bool:
        return await self.send_categorized_alert("general_accepted", job)

    async def notify_verified_jobs(
        self,
        jobs: list[dict[str, Any]],
        store: Any,
        dedup_key: str | None = None,
        limit: int = 10,
    ) -> int:
        sent = 0
        for job in jobs[:limit]:
            success = await self.send_categorized_alert("eligible", job, dedup_key="")
            if success:
                sent += 1
                await asyncio.sleep(0.3)
        return sent

    # ── commands ───────────────────────────────────────────────────────

    async def _reply(self, message: discord.Message, text: str) -> None:
        try:
            await message.channel.send(text[:1900])
        except Exception as e:
            logger.warning("Discord command reply failed", error=str(e))

    async def _handle_status(self, message: discord.Message) -> None:
        s = _pipeline_state
        lines = [
            "**Pipeline Status**",
            f"Running: {s.get('running')}",
            f"Sweep: {s.get('sweep')}",
            f"Phase: {s.get('phase')}",
            f"Matched total: {s.get('matched_total')}",
            f"Scraped count: {s.get('scraped_count')}",
        ]
        if s.get("last_error"):
            lines.append(f"Last error: {s.get('last_error')}")
        ml_line = await self._ml_status_line()
        lines.append(ml_line)
        await self._reply(message, "\n".join(lines))

    async def _handle_health(self, message: discord.Message) -> None:
        try:
            report = await run_health_checks()
        except Exception as e:
            report = f"Health check failed: {e}"
        await self._reply(message, report[:1900])

    async def _handle_analytics(self, message: discord.Message) -> None:
        await self._reply(message, "Crunching market data and calculating skill arbitrage...")
        queue_lines = await autofill_queue_lines()
        if queue_lines:
            await message.channel.send("**[QUEUE] Autofill Status**\n" + "\n".join(queue_lines))
        if self.ctx is None:
            await message.channel.send("Analytics unavailable (no LLM context).")
            return
        try:
            from src.agent.analytics_agent import AnalyticsAgent
            from src.graph.graph_store import GraphStore
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            graph = await GraphStore.create()
            try:
                agent = AnalyticsAgent(store=store, graph=graph, ctx=self.ctx)
                sections = await agent.generate_resilient_report()
            finally:
                await graph.close()
                await store.close()
            for section in sections:
                if section.strip():
                    await message.channel.send(section[:1900])
                    await asyncio.sleep(0.5)
        except Exception as e:
            await message.channel.send(f"Analytics failed: {e}")

    async def _handle_resend(self, message: discord.Message) -> None:
        dry = "--dry" in message.content
        rest = message.content.replace("--dry", "")
        limit = 10
        m = re.search(r"(\d+)\s*$", rest)
        if m:
            limit = int(m.group(1))
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                async with store._pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT canonical_id, normalized_role, normalized_company, "
                        "direct_apply_url, normalized_location, match_percent, "
                        "shortlist_probability, verdict, funding_stage, salary_amount, "
                        "salary_currency, salary_period, salary_raw, jd_summary, "
                        "company_description, role_summary, is_remote, founders, "
                        "funding_info, founder_socials, company_news, osint_signals, "
                        "extra FROM radar_candidates "
                        "WHERE eligibility = 'accepted' "
                        "ORDER BY match_percent DESC LIMIT $1",
                        limit,
                    )
            finally:
                await store.close()
            if not rows:
                await self._reply(message, "No accepted candidates in memory right now.")
                return
            if dry:
                await self._reply(
                    message,
                    f"**Resend dry run** — would re-send {len(rows)} accepted candidate alert(s).",
                )
                return
            sent = 0
            for r in rows:
                job = {
                    "role": r["normalized_role"] or "Software Engineer",
                    "company": r["normalized_company"] or "Company",
                    "location": r.get("normalized_location") or "Remote",
                    "match_percent": r["match_percent"],
                    "shortlist_probability": r.get("shortlist_probability") or 0,
                    "verdict": r["verdict"],
                    "salary": r.get("salary_raw") or "",
                    "apply_link": r.get("direct_apply_url") or "",
                    "jd_summary": r.get("jd_summary") or "",
                    "company_description": r.get("company_description") or "",
                    "role_summary": r.get("role_summary") or "",
                    "is_remote": bool(r.get("is_remote")),
                    "founders": r.get("founders") or [],
                    "funding_stage": r.get("funding_stage") or "",
                    "funding_info": r.get("funding_info") or {},
                    "founder_socials": r.get("founder_socials") or [],
                    "company_news": r.get("company_news") or "",
                    "osint_signals": r.get("osint_signals") or [],
                }
                if await self.send_categorized_alert("eligible", job):
                    sent += 1
                    await asyncio.sleep(0.3)
            await self._reply(message, f"Re-sent {sent} accepted candidate alert(s).")
        except Exception as e:
            await self._reply(message, f"Resend failed: {e}")

    async def _handle_help(self, message: discord.Message) -> None:
        await self._reply(
            message,
            "**Commands**\n/memory – update resume + persona (Q&A runs in a thread)\n"
            "/persona – view stored persona\n/status – pipeline state\n"
            "/health – service health\n/analytics – market report\n"
            "/resend – resend matches\n/help – this message",
        )

    # ── memory wizard ─────────────────────────────────────────────────

    async def _memory_starter(self, message: discord.Message, command: str, rest: str) -> None:
        thread = getattr(message.channel, "create_thread", None)
        if thread is None:
            await self._reply(
                message,
                "Run /memory from a server text channel — I'll spin up a thread there.",
            )
            return
        starter = await message.channel.send(
            "**Memory update** — I'll walk through your resume and persona "
            "Q&A here. Reply with answers in this thread, or use the buttons."
        )
        try:
            t = await starter.create_thread(name=command, auto_archive_duration=1440)
            await t.send(
                "Use this thread for this update. Paste any text or links "
                "as replies, and use the quick buttons where they appear.\n"
                "Type `/memory done` or just wait to finish each step."
            )
        except Exception as e:
            logger.warning("Memory thread creation failed", error=str(e))
            await self._reply(message, "Couldn't create the thread for this update.")
            return

        if rest:
            await t.send(f"`/memory {rest}`")

        session = _MemoryWizardSession(t, self)
        self._memory_sessions[t.id] = session
        try:
            result = await self._run_memory_wizard(session, rest)
            await t.send(f"✅ **Memory updated** — {result}")
        except Exception as e:
            await t.send(f"❌ **Memory update failed** — {e}")
        finally:
            self._memory_sessions.pop(t.id, None)

    def _is_memory_thread(self, channel: discord.abc.Messageable | None) -> bool:
        if not isinstance(channel, discord.Thread):
            return False
        return channel.id in self._memory_sessions

    def _is_memory_answer(self, message: discord.Message) -> bool:
        if not self._is_memory_thread(message.channel):
            return False
        session = self._memory_sessions.get(message.channel.id)
        if session is None or session.pending is None:
            return False
        # A command inside the thread (e.g. `/memory done`) just moves the
        # current question along — never start a nested wizard.
        if message.content.startswith("/"):
            session.pending.set_result("skip")
        else:
            session.pending.set_result(message.content.strip())
        return True

    async def _memory_button(self, custom_id: str, interaction: discord.Interaction) -> None:
        key = custom_id[len("mem_") :]
        value = _MEMORY_BUTTON_VALUES.get(custom_id, key)
        try:
            with contextlib.suppress(Exception):
                await interaction.response.defer()
            if interaction.message is not None:
                channel = interaction.message.channel
            else:
                return
            session = self._memory_sessions.get(channel.id)
            if session is None:
                with contextlib.suppress(Exception):
                    await interaction.followup.send("That memory update already finished.")
                return
            if session.pending is None:
                with contextlib.suppress(Exception):
                    await interaction.followup.send("No question pending right now.")
                return
            session.pending.set_result(value)
        except Exception as e:
            logger.debug("Memory button routing failed", source="discord", error=str(e))

    async def _run_memory_wizard(self, session: _MemoryWizardSession, instruction: str) -> str:
        from src.agent.memory_wizard import MemoryWizard

        wizard = MemoryWizard(ask=session.ask, log=session.log, ctx=self.ctx)
        return await wizard.run(instruction)

    async def _handle_memory(self, message: discord.Message) -> None:
        rest = message.content[len("/memory") :].strip()
        await self._memory_starter(message, "Memory update", rest)

    async def _handle_persona(self, message: discord.Message) -> None:
        try:
            from src.agent.memory_wizard import format_persona

            text = format_persona()
        except Exception as e:
            await self._reply(message, f"Couldn't read persona: {e}")
            return
        for chunk in re.findall(r".{1,1900}", text, re.DOTALL) or [text]:
            await self._reply(message, chunk)

    # ── native slash commands ─────────────────────────────────────────

    async def _sync_app_commands(self) -> None:
        """Register/refresh the bot's native slash commands in Discord.

        Runs once on gateway-ready. With DISCORD_GUILD_ID set the commands
        sync instantly to that guild; otherwise they register globally (can
        take up to an hour to propagate). Fails silently — the text-command
        handlers still work as a fallback.
        """
        tree = self._tree
        if tree is None:
            return
        try:

            @tree.command(
                name="memory", description="Update resume + persona memory (Q&A in a thread)"
            )
            async def _memory(interaction: discord.Interaction, note: str | None = None) -> None:
                await self._slash_memory(interaction, note or "")

            @tree.command(name="persona", description="View the full stored persona")
            async def _persona(interaction: discord.Interaction) -> None:
                await self._slash_persona(interaction)

            @tree.command(name="status", description="Current pipeline status")
            async def _status(interaction: discord.Interaction) -> None:
                await self._slash_bridge(interaction, "/status", self._handle_status)

            @tree.command(name="health", description="Live health checks for all services")
            async def _health(interaction: discord.Interaction) -> None:
                await self._slash_bridge(interaction, "/health", self._handle_health)

            @tree.command(
                name="analytics", description="Market intelligence & skill arbitrage report"
            )
            async def _analytics(interaction: discord.Interaction) -> None:
                await self._slash_bridge(interaction, "/analytics", self._handle_analytics)

            @tree.command(name="resend", description="Re-send accepted job matches")
            async def _resend(
                interaction: discord.Interaction, dry: bool = False, limit: int = 10
            ) -> None:
                content = f"/resend{' --dry' if dry else ''} {limit}"
                await self._slash_bridge(interaction, content, self._handle_resend)

            @tree.command(name="help", description="List all available commands")
            async def _help(interaction: discord.Interaction) -> None:
                await self._slash_bridge(interaction, "/help", self._handle_help)

        except Exception as e:
            logger.warning("Slash command registration failed", error=str(e))
            return

        try:
            target: discord.abc.Snowflake | None = None
            if self._guild_id:
                target = discord.Object(id=int(self._guild_id))
            synced = await tree.sync(guild=target)
            logger.info(
                "Slash commands synced",
                count=len(synced),
                guild=self._guild_id or "global",
            )
        except Exception as e:
            logger.warning("Slash command sync failed", error=str(e))

    async def _slash_bridge(
        self,
        interaction: discord.Interaction,
        content: str,
        handler: Any,
    ) -> None:
        """Route a native slash command into the existing message handler.

        Acks the interaction, then hands it a ``_SlashAdapter`` so the shared
        ``_handle_*`` logic runs unchanged (it posts via channel.send).
        """
        with contextlib.suppress(Exception):
            await interaction.response.defer()
        adapter = _SlashAdapter(content, interaction.channel)
        try:
            await handler(adapter)
        except Exception as e:
            logger.debug("Slash bridge handler failed", error=str(e))
            with contextlib.suppress(Exception):
                await interaction.followup.send(f"Command failed: {e}")

    async def _slash_memory(self, interaction: discord.Interaction, note: str) -> None:
        with contextlib.suppress(Exception):
            await interaction.response.defer()
        adapter = _SlashAdapter(f"/memory {note}".strip(), interaction.channel)
        await self._handle_memory(adapter)  # type: ignore[arg-type]

    async def _slash_persona(self, interaction: discord.Interaction) -> None:
        with contextlib.suppress(Exception):
            await interaction.response.defer()
        adapter = _SlashAdapter("/persona", interaction.channel)
        await self._handle_persona(adapter)  # type: ignore[arg-type]

    # ── autofill question mailbox ──────────────────────────────────────

    async def _question_mailbox_db(self) -> Any:
        if self._mailbox_db is None:
            from autofill.src.core.db import AutofillDB

            self._mailbox_db = await AutofillDB.create()
        return self._mailbox_db

    async def _heartbeat_poller(self) -> None:
        try:
            db = await self._question_mailbox_db()
            await db.heartbeat_poller()
        except Exception:
            pass

    async def _capture_mailbox_answer(self, message: discord.Message) -> None:
        """Route a reply to a pending autofill question into the mailbox.

        The autofill bridge sends a question with buttons; this agent (the
        single gateway) captures the user's reply (text reply_to, or a button
        press) and stores it for the bridge to poll.
        """
        answer: str | None = None
        target_id: int | None = None
        if message.reference is not None and message.reference.message_id is not None:
            target_id = message.reference.message_id
            answer = (message.content or "").strip() or None
        if answer is None:
            return
        try:
            db = await self._question_mailbox_db()
            if await db.answer_mailbox_message(target_id, answer):
                logger.info(
                    "Routed Discord answer to pending autofill question", message_id=target_id
                )
        except Exception as e:
            logger.debug("Mailbox answer routing failed", source="discord", error=str(e))

    async def _capture_mailbox_button(
        self, message_id: int, custom_id: str, interaction: discord.Interaction
    ) -> None:
        """Route a component button press to a pending autofill question, then
        confirm to the user (answer recorded + updated memory count)."""
        label = custom_id
        try:
            if interaction.message and interaction.message.components:
                for row in interaction.message.components:
                    for comp in getattr(row, "children", []) or []:
                        if getattr(comp, "custom_id", None) == custom_id:
                            label = getattr(comp, "label", "") or label
        except Exception:
            pass
        try:
            db = await self._question_mailbox_db()
            if await db.answer_mailbox_message(message_id, custom_id):
                logger.info(
                    "Routed Discord button to pending autofill question", message_id=message_id
                )
                persona_count = await self._persona_chunk_count()
                msg = (
                    f"✅ **{label}** recorded — memory updated "
                    f"(persona_embeddings: {persona_count})"
                )
            else:
                # Stale click on an old question: report its actual state.
                row = await db.mailbox_lookup_by_message(message_id)
                if row:
                    q = str(row.get("question") or "this question")
                    state = str(row.get("state") or "unknown")
                    if row.get("answer"):
                        msg = f"ℹ️ `{q}` was already answered **{row['answer']}** ({state})."
                    else:
                        msg = f"ℹ️ `{q}` is no longer active ({state}); it wasn't re-asked."
                else:
                    msg = "Hmm, I couldn't match that question."
            with contextlib.suppress(Exception):
                await interaction.response.send_message(msg)
        except Exception as e:
            logger.debug("Mailbox button routing failed", source="discord", error=str(e))

    async def _persona_chunk_count(self) -> int:
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                return await store.persona_chunk_count()
            finally:
                await store.close()
        except Exception:
            return 0
