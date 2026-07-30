"""TelegramAgent: Delivers real-time job alerts with inline keyboards,
responds to bot commands, pushes proactive stealth/warm-intro signals,
and notifies on pipeline errors.

Commands (send to bot in Telegram DMs):
    /status    – current pipeline state (sweep, matched jobs, LLM status)
    /health    – runs live health checks on all services
    /analytics – generate market intelligence & skill arbitrage report
    /help      – lists available commands
"""  # noqa: E501

from __future__ import annotations

import asyncio
import contextlib
import html
import os
import re
import time
from typing import TYPE_CHECKING, Any

import httpx

from src.logging import get_logger

if TYPE_CHECKING:
    from src.llm.context import ContextManager

logger = get_logger("telegram_agent")

TELEGRAM_BASE = "https://api.telegram.org/bot{token}"
TELEGRAM_SEND = f"{TELEGRAM_BASE}/sendMessage"
TELEGRAM_UPDATES = f"{TELEGRAM_BASE}/getUpdates"

_TG_MAX_LEN = 4000
_HTML_TAG_RX = re.compile(r"<[^>]+>")

_pipeline_state: dict[str, Any] = {
    "running": False,
    "sweep": 0,
    "phase": "idle",
    "matched_total": 0,
    "last_error": None,
    "started_at": None,
    "llm_tokens_used": 0,
    "scraped_count": 0,
}


def set_pipeline_state(**kwargs: Any) -> None:
    _pipeline_state.update(kwargs)


async def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        return True
    except Exception:
        return False


async def _check_http(url: str, timeout: float = 3.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            return resp.status_code < 500
    except Exception:
        return False


async def run_health_checks() -> str:
    results: list[tuple[str, bool]] = []
    results.append(("llama-server :8900", await _check_http("http://localhost:8900/health")))
    results.append(("Firecrawl :3002", await _check_http("http://localhost:3002")))
    results.append(("SearXNG :8080", await _check_http("http://localhost:8080")))
    results.append(("pgvector :5433", await _check_port("localhost", 5433)))

    lines = ["<b>Health Check</b>", ""]
    all_ok = True
    for name, ok in results:
        icon = "✅" if ok else "❌"
        lines.append(f"{icon} {name}")
        if not ok:
            all_ok = False
    if all_ok:
        lines.extend(["", "All services healthy."])
    else:
        lines.extend(["", "One or more services are DOWN."])
    return "\n".join(lines)


class TelegramAgent:
    """Agent responsible for Telegram alerts, command handling, and error notifications."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        ctx: ContextManager | None = None,
    ) -> None:
        self.bot_token = bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID", "")
        self.ctx = ctx
        self._notified_keys: set[str] = set()
        self._update_id: int = 0
        self._poll_task: asyncio.Task[None] | None = None
        self._seen_errors: set[str] = set()
        self._stealth_notified: set[str] = set()

    @property
    def _chat_ids(self) -> list[str]:
        raw = (self.chat_id or "").strip()
        if not raw:
            return []
        return [cid.strip() for cid in raw.split(",") if cid.strip()]

    @property
    def _primary_chat_id(self) -> str:
        ids = self._chat_ids
        return ids[0] if ids else ""

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # ── low-level send ──────────────────────────────────────────────

    async def _send_raw(
        self,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: dict | None = None,
    ) -> bool:
        if not self.is_configured:
            return False
        recipients = self._chat_ids
        if not recipients:
            return False

        all_ok = True
        chunks = [text[i : i + _TG_MAX_LEN] for i in range(0, len(text), _TG_MAX_LEN)]
        for i, chunk in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(0.5)
            for cid in recipients:
                ok = await self._send_to_chat(cid, chunk, parse_mode, reply_markup)
                all_ok = all_ok and ok
        return all_ok

    async def _send_to_chat(
        self,
        cid: str,
        text: str,
        parse_mode: str,
        reply_markup: dict | None = None,
    ) -> bool:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    payload: dict[str, Any] = {
                        "chat_id": cid,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    }
                    if reply_markup:
                        payload["reply_markup"] = reply_markup

                    resp = await client.post(
                        TELEGRAM_SEND.format(token=self.bot_token), json=payload
                    )
                    if resp.status_code == 200:
                        return True

                    body = resp.text[:200]
                    logger.warning(
                        f"Telegram send {resp.status_code}",
                        source="telegram",
                        extra={"body": body},
                    )
                    if resp.status_code == 400 and parse_mode == "HTML":
                        text = _HTML_TAG_RX.sub("", text)
                        parse_mode = ""
                        continue
                    if resp.status_code == 429:
                        await asyncio.sleep(2 << attempt)
                    elif resp.status_code >= 500:
                        await asyncio.sleep(1 << attempt)
                    else:
                        return False
            except Exception as e:
                logger.warning(
                    f"Telegram send attempt {attempt + 1} failed",
                    source="telegram",
                    exception=str(e),
                )
                if attempt < 2:
                    await asyncio.sleep(1 << attempt)
        return False

    # ── polling / commands ──────────────────────────────────────────

    async def start_polling(self) -> None:
        if not self.is_configured:
            return
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_loop())
            logger.info("TelegramAgent command bot polling started")

    async def stop_polling(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

    async def _poll_loop(self) -> None:
        self._update_id = 0
        while True:
            try:
                await self._process_updates()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug("Telegram poll error", source="telegram", exception=str(e))
            await asyncio.sleep(5)

    async def _process_updates(self) -> None:
        params: dict[str, Any] = {"timeout": 3, "allowed_updates": ["message"]}
        if self._update_id > 0:
            params["offset"] = self._update_id + 1

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(TELEGRAM_UPDATES.format(token=self.bot_token), params=params)
            if resp.status_code != 200:
                return
            data = resp.json()
            if not data.get("ok"):
                return

        for upd in data.get("result", []):
            self._update_id = max(self._update_id, upd.get("update_id", 0))
            msg = upd.get("message", {})
            chat = msg.get("chat", {})
            sender_id = str(chat.get("id", ""))
            if sender_id not in self._chat_ids:
                continue
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue

            cmd = text.split()[0].lower().split("@")[0]
            if cmd == "/status":
                await self._handle_status()
            elif cmd == "/health":
                await self._handle_health()
            elif cmd == "/analytics":
                await self._handle_analytics()
            elif cmd == "/help":
                await self._handle_help()

    async def _handle_analytics(self) -> None:
        await self._send_raw("⏳ <i>Crunching market data and calculating skill arbitrage...</i>")
        try:
            from src.agent.analytics_agent import AnalyticsAgent
            from src.graph.graph_store import GraphStore
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            graph = await GraphStore.create()
            try:
                agent = AnalyticsAgent(store=store, graph=graph, ctx=self.ctx)
                report = await agent.generate_market_report()
                await self._send_raw(report)
            finally:
                await graph.close()
                await store.close()
        except Exception as e:
            logger.exception("Analytics report generation failed", exc=e)
            await self._send_raw(
                "❌ <b>Analytics Report Failed</b>\n\n"
                f"<code>{str(e)[:400]}</code>\n\n"
                "Try again in a moment."
            )

    async def _handle_status(self) -> None:
        s = _pipeline_state
        uptime = ""
        if s.get("started_at"):
            delta = int(time.time() - s["started_at"])
            h, m = divmod(delta, 3600)
            mm, ss = divmod(m, 60)
            uptime = f"{h}h {mm}m {ss}s"

        status = "🟢 running" if s["running"] else "🔴 idle"
        lines = [
            "<b>Pipeline Status</b>",
            "",
            f"State: {status}",
            f"Phase: {s['phase']}",
            f"Sweep: {s['sweep']}",
            f"Matched total: {s['matched_total']}",
            f"Scraped this sweep: {s['scraped_count']}",
        ]
        if uptime:
            lines.append(f"Uptime: {uptime}")
        if s.get("last_error"):
            lines.extend(["", "<b>Last error:</b>", f"<code>{s['last_error'][:200]}</code>"])
        lines.extend(["", "Send /health to check services."])
        await self._send_raw("\n".join(lines))

    async def _handle_health(self) -> None:
        report = await run_health_checks()
        await self._send_raw(report)

    async def _handle_help(self) -> None:
        lines = [
            "<b>Commands</b>",
            "",
            "/status    – pipeline state + match count",
            "/health    – live service health check",
            "/analytics – market intelligence & skill arbitrage report",
            "/help      – this message",
            "",
            "I'll also notify you on pipeline errors, new matches,",
            "stealth hiring signals, and warm-intro paths.",
        ]
        await self._send_raw("\n".join(lines))

    # ── notifications ───────────────────────────────────────────────

    async def send_error(self, message: str, dedup_key: str = "") -> None:
        if dedup_key and dedup_key in self._seen_errors:
            return
        if dedup_key:
            self._seen_errors.add(dedup_key)
        await self._send_raw(f"🚨 <b>Pipeline Error</b>\n\n<code>{message[:800]}</code>")
        logger.info("TelegramAgent sent error alert")

    async def send_startup(self, sweep_count: int = 0) -> None:
        await self._send_raw(
            f"🟢 <b>Pipeline Started</b>\n\n"
            f"Resume loaded, {sweep_count} existing jobs.\n"
            f"Beginning sweeps..."
        )

    async def send_sweep_summary(
        self, sweep: int, matched: int, scraped: int, duration: float
    ) -> None:
        await self._send_raw(
            f"✅ <b>Sweep {sweep} Complete</b>\n\n"
            f"Scraped: {scraped}\n"
            f"Matched: {matched}\n"
            f"Duration: {duration:.1f}s"
        )

    # ── job card + inline keyboards ─────────────────────────────────

    def format_job_card(self, job: dict[str, Any]) -> str:
        role = html.escape(str(job.get("role") or "Software Engineer").strip())
        company = html.escape(str(job.get("company") or "Company").strip())
        match_pct = job.get("match_percent", 0)
        shortlist_pct = job.get("shortlist_probability", 0)
        salary = html.escape(str(job.get("salary") or "Not specified").strip())
        location = html.escape(str(job.get("location") or "Remote").strip())

        comp_desc = html.escape(
            str(
                job.get("company_description")
                or job.get("jd_summary")
                or job.get("role_summary")
                or ""
            ).strip()
        )
        if len(comp_desc) > 200:
            comp_desc = comp_desc[:197] + "..."

        lines = [
            f"<b>{role.upper()}</b> • <b>{company.upper()}</b>",
            "<code>───────────────────────────</code>",
            f"<b>JD Match:</b> {match_pct}%  |  <b>Shortlist:</b> {shortlist_pct}%",
            f"<b>Location:</b> {location}",
        ]

        if salary and salary != "-":
            lines.append(f"<b>Salary:</b> {salary}")

        if comp_desc:
            lines.extend(["", f"<blockquote>{comp_desc}</blockquote>"])

        funding_info = job.get("funding_info") or {}
        funding_stage = job.get("funding_stage", "")
        founders = job.get("founders", [])
        osint_signals = job.get("osint_signals", [])

        has_osint = bool(funding_info or funding_stage or founders or osint_signals)
        if has_osint:
            lines.extend(["", "<b>🕵️ OSINT &amp; Outreach</b>", ""])

        if isinstance(funding_info, dict) and any(funding_info.values()):
            fi = funding_info
            parts = []
            if fi.get("round"):
                parts.append(f"💰 <b>{fi['round']}</b>")
            if fi.get("amount_raised"):
                parts.append(f"({fi['amount_raised']})")
            if fi.get("lead_investors"):
                parts.append(f"led by {', '.join(fi['lead_investors'])}")
            if fi.get("date_announced"):
                parts.append(f"[{fi['date_announced']}]")
            if parts:
                lines.append(" ".join(parts))
        elif funding_stage and funding_stage not in ("N/A", "-"):
            lines.append(f"💰 Funding: {funding_stage}")

        if founders:
            if isinstance(founders[0], dict):
                for f in founders:
                    name = html.escape(str(f.get("name", "?")))
                    title = html.escape(str(f.get("title", "")))
                    title_str = f" ({title})" if title else ""
                    badges = []
                    if f.get("email"):
                        badges.append(f'<a href="mailto:{html.escape(f["email"])}">Email</a>')
                    if f.get("linkedin_url"):
                        badges.append(f'<a href="{html.escape(f["linkedin_url"])}">LinkedIn</a>')
                    if f.get("github_url"):
                        badges.append(f'<a href="{html.escape(f["github_url"])}">GitHub</a>')
                    badge_str = f" — {' | '.join(badges)}" if badges else ""
                    lines.append(f"👤 {name}{title_str}{badge_str}")
            else:
                lines.append(f"👤 Founders: {', '.join(html.escape(str(f)) for f in founders)}")
                socials = job.get("founder_socials", [])
                if socials:
                    sl = []
                    for s in socials[:2]:
                        if isinstance(s, str) and s.startswith("http"):
                            sl.append(
                                f'<a href="{html.escape(s)}">{html.escape(s.split("//")[-1])}</a>'
                            )
                        else:
                            sl.append(str(s))
                    lines.append(f"   Links: {', '.join(sl)}")

        if osint_signals:
            for sig in osint_signals:
                lines.append(f"📡 {html.escape(str(sig))}")

        founder_posts = job.get("founder_posts", [])
        if founder_posts and isinstance(founder_posts, list):
            lines.extend(["", "<b>🚨 ACTIVE FOUNDER POST:</b>"])
            for fp in founder_posts[:2]:
                if not isinstance(fp, dict):
                    continue
                name = html.escape(str(fp.get("founder_name", "Unknown")))
                intent = html.escape(str(fp.get("intent", "")))
                post_url = fp.get("post_url", "")
                line = f"👤 <b>{name}</b>"
                if intent:
                    line += f" — {intent}"
                lines.append(line)
                if post_url.startswith("http"):
                    lines.append(
                        f'└ <a href="{html.escape(post_url)}"><b>DM them on LinkedIn →</b></a>'
                    )
                lines.append("")

        return "\n".join(lines)

    async def send_notification(self, job: dict[str, Any]) -> bool:
        if not self.is_configured:
            return False

        text = self.format_job_card(job)

        buttons: list[list[dict[str, str]]] = []
        link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""
        if link and str(link).startswith("http"):
            buttons.append([{"text": "🚀 Apply Direct", "url": link}])

        founders = job.get("founders", [])
        if founders and isinstance(founders[0], dict):
            for f in founders[:2]:
                if f.get("linkedin_url"):
                    name = html.escape(str(f.get("name", "Founder")))
                    buttons.append([{"text": f"👤 {name} LinkedIn", "url": f["linkedin_url"]}])

        reply_markup = {"inline_keyboard": buttons} if buttons else None
        return await self._send_to_chat(self._primary_chat_id, text, "HTML", reply_markup)

    async def notify_verified_jobs(
        self,
        jobs: list[dict[str, Any]],
        min_match_pct: int = 40,
        store: Any | None = None,
    ) -> int:
        if not self.is_configured:
            return 0

        sent_count = 0
        for j in jobs:
            role = str(j.get("role") or "").strip()
            company = str(j.get("company") or "").strip()
            match_pct = int(j.get("match_percent", 0))

            if (
                not role
                or not company
                or role in ("N/A", "Unknown")
                or company in ("N/A", "Unknown")
            ):
                continue
            if match_pct < min_match_pct:
                continue

            dedup_key = f"{company.lower()}:{role.lower()}"
            if dedup_key in self._notified_keys:
                continue

            if store is not None:
                try:
                    if await store.is_telegram_notified(dedup_key):
                        self._notified_keys.add(dedup_key)
                        continue
                except Exception:
                    pass

            success = await self.send_notification(j)
            if success:
                self._notified_keys.add(dedup_key)
                if store is not None:
                    with contextlib.suppress(Exception):
                        await store.mark_telegram_notified(dedup_key, role, company)
                sent_count += 1
                logger.info(f"Telegram alert sent for {role} @ {company}")
                await asyncio.sleep(1.2)

        return sent_count

    # ── proactive stealth & warm-intro signals ──────────────────────

    async def notify_stealth_startup(self, startup: dict[str, Any]) -> None:
        """Proactive push alert when a funded company has zero job postings."""
        dedup = startup.get("company_name", "")
        if dedup in self._stealth_notified:
            return
        self._stealth_notified.add(dedup)

        name = html.escape(startup.get("company_name", "Unknown"))
        stage = html.escape(startup.get("funding_stage", "Unknown"))
        url = startup.get("url", "")

        text = (
            f"🕵️‍♂️ <b>STEALTH HIRING SIGNAL</b>\n\n"
            f"<b>{name}</b> just surfaced with <b>{stage}</b> funding, "
            f"but has ZERO job postings.\n"
            f"This is your chance to bypass the ATS entirely."
        )

        buttons: list[list[dict[str, str]]] = []
        if url:
            buttons.append([{"text": "🌐 Website", "url": url}])
        buttons.append(
            [
                {
                    "text": f"🔍 Search '{name}' on LinkedIn",
                    "url": (
                        f"https://www.linkedin.com/search/results/people/?keywords={name}%20founder"
                    ),
                }
            ]
        )

        await self._send_to_chat(
            self._primary_chat_id,
            text,
            "HTML",
            {"inline_keyboard": buttons},
        )
        logger.info("Stealth signal pushed", entity=name)

    async def notify_warm_intro(
        self,
        paths: list[dict[str, Any]],
        target_company: str,
    ) -> None:
        """Push warm-intro paths with an LLM-generated cold-DM draft."""
        if not paths or not self.is_configured:
            return

        name = html.escape(target_company)
        lines = [
            f"🔗 <b>2-HOP WARM INTRO: {name}</b>",
            "",
        ]
        for idx, p in enumerate(paths[:3], 1):
            founder = html.escape(str(p.get("founder_name", "?")))
            common = html.escape(str(p.get("common_ground", "?")))
            linkedin = p.get("linkedin_url", "")
            lines.append(f"<b>{idx}.</b> {founder} — shared: {common}")
            if linkedin:
                lines.append(f'    <a href="{html.escape(linkedin)}">LinkedIn →</a>')

        buttons: list[list[dict[str, str]]] = []
        for p in paths[:3]:
            linkedin = p.get("linkedin_url", "")
            if linkedin:
                founder = html.escape(str(p.get("founder_name", "Founder")))
                buttons.append([{"text": f"👤 DM {founder}", "url": linkedin}])

        draft = ""
        if self.ctx is not None and paths:
            draft = await self._generate_cold_dm_draft(paths, target_company)

        if draft:
            lines.extend(["", f"<blockquote expandable>{html.escape(draft)}</blockquote>"])

        await self._send_to_chat(
            self._primary_chat_id,
            "\n".join(lines),
            "HTML",
            {"inline_keyboard": buttons} if buttons else None,
        )
        logger.info("Warm-intro signal pushed", entity=target_company)

    async def _generate_cold_dm_draft(self, paths: list[dict[str, Any]], company_name: str) -> str:
        common_grounds = [p.get("common_ground", "") for p in paths if p.get("common_ground")]
        founder_names = [p.get("founder_name", "") for p in paths if p.get("founder_name")]

        prompt = (
            f"You are helping a candidate write a cold DM to a startup founder.\n\n"
            f"Company: {company_name}\n"
            f"Shared common ground: {', '.join(common_grounds[:3])}\n"
            f"Founder names: {', '.join(founder_names[:2])}\n\n"
            "Write a single, punchy, 3-sentence LinkedIn DM the candidate "
            "can send. Mention the shared common ground naturally. "
            "Be warm, direct, and ask about engineering roles. "
            "Return ONLY the draft text, no quotes, no explanations."
        )
        try:
            draft = await self.ctx.chat(prompt[:3000])
            return draft.strip().strip('"')
        except Exception:
            return ""

    async def push_stealth_and_warm_intro_batch(
        self,
        stealth: list[dict[str, Any]],
        warm_intro_targets: list[dict[str, Any]] | None = None,
    ) -> int:
        """Called by the orchestrator to push stealth + warm-intro alerts.

        Returns number of alerts sent.
        """
        sent = 0
        for s in stealth:
            try:
                await self.notify_stealth_startup(s)
                sent += 1
            except Exception:
                pass

        if warm_intro_targets:
            for wi in warm_intro_targets:
                try:
                    company = wi.get("company", "")
                    paths = wi.get("paths", [])
                    if paths:
                        await self.notify_warm_intro(paths, company)
                        sent += 1
                except Exception:
                    pass
        return sent
