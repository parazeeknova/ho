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
import os
import time
from typing import Any

import httpx

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
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
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
    results.append(("Qdrant :6333", await _check_port("localhost", 6333)))

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
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._notified_keys: set[str] = set()
        self._update_id: int = 0
        self._poll_task: asyncio.Task[None] | None = None
        self._seen_errors: set[str] = set()  # dedupe repeated errors

    # ---- helpers ----------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def _send_raw(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.is_configured:
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    TELEGRAM_SEND.format(token=self.bot_token),
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    },
                )
                return resp.status_code == 200
        except Exception as e:
            print(f"  [dim]Telegram send failed: {e}[/dim]")
            return False

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
            resp = await client.get(
                TELEGRAM_UPDATES.format(token=self.bot_token), params=params
            )
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
            if sender_id != str(self.chat_id):
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
        await self._send_raw(
            f"🚨 <b>Pipeline Error</b>\n\n<code>{message[:800]}</code>"
        )
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
        role = str(job.get("role") or "Software Engineer").strip()
        company = str(job.get("company") or "Company").strip()
        match_pct = job.get("match_percent", 0)
        shortlist_pct = job.get("shortlist_probability", 0)
        salary = str(job.get("salary") or "Not specified").strip()
        location = str(job.get("location") or "Remote").strip()
        link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""

        comp_desc = str(
            job.get("company_description")
            or job.get("jd_summary")
            or job.get("role_summary")
            or ""
        ).strip()
        if len(comp_desc) > 200:
            comp_desc = comp_desc[:197] + "..."

        founders = job.get("founders", [])
        funding = job.get("funding_stage")
        socials = job.get("founder_socials", [])

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

        if founders:
            lines.append(f"<b>Founders:</b> {', '.join(founders)}")
        if funding:
            lines.append(f"<b>Funding:</b> {funding}")
        if socials:
            social_links = [
                f'<a href="{s}">{s.split("//")[-1]}</a>'
                if s.startswith("http")
                else s
                for s in socials[:2]
            ]
            lines.append(f"<b>Outreach:</b> {', '.join(social_links)}")

        if link and str(link).startswith("http"):
            lines.extend(["", f'<a href="{link}"><b>Apply Direct →</b></a>'])

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
