"""TelegramAgent: Delivers real-time job alerts, responds to bot commands,
and notifies on pipeline errors.

Commands (send to bot in Telegram DMs):
    /status   – current pipeline state (sweep, matched jobs, LLM status)
    /health   – runs live health checks on all services
    /help     – lists available commands
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import os
import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from src.llm.context import ContextManager

TELEGRAM_BASE = "https://api.telegram.org/bot{token}"
TELEGRAM_SEND = f"{TELEGRAM_BASE}/sendMessage"
TELEGRAM_UPDATES = f"{TELEGRAM_BASE}/getUpdates"

# ---------------------------------------------------------------------------
# Shared pipeline state (written by orchestrator, read by /status handler)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Async health checks (non-blocking, runs in-process)
# ---------------------------------------------------------------------------
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
    """Return a formatted health-check report as plain text."""
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
        lines.append("")
        lines.append("All services healthy.")
    else:
        lines.append("")
        lines.append("One or more services are DOWN.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TelegramAgent
# ---------------------------------------------------------------------------


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
        self._seen_errors: set[str] = set()  # dedupe repeated errors

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

    # ---- helpers ----------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def _send_raw(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.is_configured:
            return False
        recipients = self._chat_ids
        if not recipients:
            return False

        all_ok = True
        for cid in recipients:
            ok = await self._send_to_chat(cid, text, parse_mode)
            all_ok = all_ok and ok
        return all_ok

    async def _send_to_chat(self, cid: str, text: str, parse_mode: str) -> bool:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        TELEGRAM_SEND.format(token=self.bot_token),
                        json={
                            "chat_id": cid,
                            "text": text,
                            "parse_mode": parse_mode,
                            "disable_web_page_preview": True,
                        },
                    )
                    if resp.status_code == 200:
                        return True
                    body = resp.text[:200]
                    print(f"  [yellow]Telegram {resp.status_code} -> chat {cid}: {body}[/yellow]")
                    if resp.status_code == 400 and parse_mode == "HTML":
                        fixed = await self._llm_fix_html(text, body)
                        if fixed and fixed != text:
                            print("  [dim]Telegram: retrying with LLM-fixed HTML[/dim]")
                            text = fixed
                            continue
                        print("  [dim]Telegram: falling back to plain text[/dim]")
                        parse_mode = ""
                        continue
                    if resp.status_code == 429:
                        await asyncio.sleep(2 << attempt)
                    elif resp.status_code >= 500:
                        await asyncio.sleep(1 << attempt)
                    else:
                        return False
            except Exception as e:
                print(f"  [yellow]Telegram send failed (attempt {attempt + 1}): {e}[/yellow]")
                if attempt < 2:
                    await asyncio.sleep(1 << attempt)
        return False

    async def _llm_fix_html(self, text: str, error_body: str) -> str | None:
        """Ask the LLM to fix Telegram parse_mode HTML violations."""
        if self.ctx is None:
            return None
        prompt = (
            "Telegram rejected this HTML message with error:\n"
            f"{error_body}\n\n"
            "Fix ALL parse_mode violations (unescaped &, unclosed tags, "
            "invalid nesting, forbidden entities) and return ONLY the "
            "corrected HTML with no explanation or code fences.\n\n"
            f"Original message:\n{text}"
        )
        try:
            fixed = await self.ctx.chat(prompt[:4000])
            if fixed and "<b>" in fixed:
                return fixed.strip()
        except Exception as e:
            print(f"  [dim]LLM HTML fix failed: {e}[/dim]")
        return None

    # ---- command bot ------------------------------------------------------

    async def start_polling(self) -> None:
        """Start background command polling. Safe to call when not configured."""
        if not self.is_configured:
            return
        if self._poll_task is None:
            self._poll_task = asyncio.create_task(self._poll_loop())
            print("  📱 [TelegramAgent] Command bot polling started")

    async def stop_polling(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None

    async def _poll_loop(self) -> None:
        """Poll getUpdates every 5s, process commands only from configured chat."""
        self._update_id = 0
        while True:
            try:
                await self._process_updates()
            except Exception as e:
                print(f"  [dim]Telegram poll error: {e}[/dim]")
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
            if sender_id != str(self._primary_chat_id):
                continue
            text = (msg.get("text") or "").strip()
            if not text.startswith("/"):
                continue

            cmd = text.split()[0].lower().split("@")[0]
            if cmd == "/status":
                await self._handle_status()
            elif cmd == "/health":
                await self._handle_health()
            elif cmd == "/help":
                await self._handle_help()

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
            "/status – pipeline state + match count",
            "/health – live service health check",
            "/help   – this message",
            "",
            "I'll also notify you instantly on pipeline errors and new job matches.",
        ]
        await self._send_raw("\n".join(lines))

    # ---- error notifications ----------------------------------------------

    async def send_error(self, message: str, dedup_key: str = "") -> None:
        """Send an error alert. Deduplicates repeated errors by key."""
        if dedup_key and dedup_key in self._seen_errors:
            return
        if dedup_key:
            self._seen_errors.add(dedup_key)
        await self._send_raw(f"🚨 <b>Pipeline Error</b>\n\n<code>{message[:800]}</code>")
        print("  📱 [TelegramAgent] Sent error alert")

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

    # ---- job notifications (existing) -------------------------------------

    def format_job_card(self, job: dict[str, Any]) -> str:
        role = html.escape(str(job.get("role") or "Software Engineer").strip())
        company = html.escape(str(job.get("company") or "Company").strip())
        match_pct = job.get("match_percent", 0)
        shortlist_pct = job.get("shortlist_probability", 0)
        salary = html.escape(str(job.get("salary") or "Not specified").strip())
        location = html.escape(str(job.get("location") or "Remote").strip())
        link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""

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

        # ── OSINT & Outreach ──────────────────────────────────────────

        funding_info = job.get("funding_info") or {}
        funding_stage = job.get("funding_stage", "")
        founders = job.get("founders", [])
        osint_signals = job.get("osint_signals", [])

        has_osint = bool(funding_info or funding_stage or founders or osint_signals)
        if has_osint:
            lines.extend(["", "<b>🕵️ OSINT &amp; Outreach</b>", ""])

        # Funding line
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

        # Founder rows with clickable links
        founders = job.get("founders", [])
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

        # OSINT signals
        if osint_signals:
            for sig in osint_signals:
                lines.append(f"📡 {html.escape(str(sig))}")

        # 🚨 Active Founder Posts
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

        # Apply link
        if link and str(link).startswith("http"):
            lines.extend(["", f'<a href="{html.escape(link)}"><b>Apply Direct →</b></a>'])

        return "\n".join(lines)

    async def send_notification(self, job: dict[str, Any]) -> bool:
        if not self.is_configured:
            return False
        return await self._send_raw(self.format_job_card(job))

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
                print(f"  📱 [TelegramAgent] Sent alert for {role} @ {company}")
                await asyncio.sleep(1.2)

        return sent_count
