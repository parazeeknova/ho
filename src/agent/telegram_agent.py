"""TelegramAgent: Delivers real-time job alerts with inline keyboards,
responds to bot commands, pushes proactive stealth/warm-intro signals,
and notifies on pipeline errors.

Commands (send to bot in Telegram DMs):
    /status    – current pipeline state (sweep, matched jobs, LLM status)
    /health    – runs live health checks on all services
    /analytics – generate market intelligence & skill arbitrage report
    /resend    – resend accepted job matches (usage: /resend [--dry] [limit])
    /help      – lists available commands
"""  # noqa: E501

from __future__ import annotations

import asyncio
import contextlib
import html
import os
import re
import shutil
import time
from typing import TYPE_CHECKING, Any

from src.http_client import get_client
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
    "rejected_total": 0,
    "last_error": None,
    "started_at": None,
    "sweep_started_at": 0.0,
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
        client = await get_client("telegram_agent", timeout=timeout)
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
            mem_total_gb = round(total / (1024 * 1024), 1)
            mem_used_gb = round((total - avail) / (1024 * 1024), 1)
            pct = round((total - avail) / total * 100, 1) if total > 0 else 0.0
            mem_str = f"{mem_used_gb}GB / {mem_total_gb}GB ({pct}%)"
    except Exception:
        pass

    try:
        du = shutil.disk_usage("/")
        total_gb = round(du.total / (1024**3), 1)
        used_gb = round(du.used / (1024**3), 1)
        pct = round(du.used / du.total * 100, 1)
        disk_str = f"{used_gb}GB / {total_gb}GB ({pct}%)"
    except Exception:
        pass

    try:
        load1, load5, _ = os.getloadavg()
        cpu_str = f"{load1:.2f} (1m), {load5:.2f} (5m)"
    except Exception:
        pass

    return {
        "ram": mem_str,
        "disk": disk_str,
        "cpu_load": cpu_str,
    }


async def run_health_checks() -> str:
    checks = [
        ("llama-server (Embed)", _check_http("http://localhost:8900/health")),
        ("Firecrawl API", _check_port("localhost", 3002)),
        ("SearXNG", _check_http("http://localhost:8080")),
        ("agent-memory-db (pgvector)", _check_port("localhost", 5433)),
        ("Neo4j Graph Store", _check_port("localhost", 7687)),
        ("Redis", _check_port("localhost", 6379)),
        ("RabbitMQ", _check_port("localhost", 5672)),
        ("Playwright Service", _check_port("localhost", 3000)),
        ("NuQ Postgres", _check_port("localhost", 5432)),
    ]

    results = await asyncio.gather(*[coro for _, coro in checks])

    lines = ["<b>System Health Check</b>", ""]
    all_ok = True
    for (name, _), ok in zip(checks, results, strict=True):
        tag = "<b>[OK]</b>" if ok else "<b>[DOWN]</b>"
        lines.append(f"{tag} {name}")
        if not ok:
            all_ok = False

    if all_ok:
        lines.extend(["", "All 9 infrastructure services are healthy."])
    else:
        lines.extend(["", "[WARNING] One or more services are down."])
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

    # low-level send

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
        chunks: list[str] = []
        current = ""
        for paragraph in text.split("\n\n"):
            if len(current) + len(paragraph) + 2 > _TG_MAX_LEN:
                if current:
                    chunks.append(current)
                current = paragraph
            else:
                current += ("\n\n" + paragraph) if current else paragraph
        if current:
            chunks.append(current)

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
                client = await get_client("telegram_agent", timeout=10.0)
                payload: dict[str, Any] = {
                    "chat_id": cid,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                }
                if reply_markup:
                    payload["reply_markup"] = reply_markup

                resp = await client.post(TELEGRAM_SEND.format(token=self.bot_token), json=payload)
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

    # polling / commands

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

        client = await get_client("telegram_agent", timeout=8.0)
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
            elif cmd == "/resend":
                await self._handle_resend(text)
            elif cmd == "/help":
                await self._handle_help()

    async def _handle_analytics(self) -> None:
        await self._send_raw("<i>Crunching market data and calculating skill arbitrage...</i>")
        sections: list[str] = []
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
        except Exception as e:
            logger.exception("Analytics report generation failed", exc=e)
            sections.append(
                "<b>[ERROR] Analytics Report Failed</b>\n\n"
                f"<code>{str(e)[:400]}</code>\n\n"
                "Try again in a moment."
            )

        for section in sections:
            if section.strip():
                await self._send_raw(section)
                await asyncio.sleep(0.5)

    async def _handle_status(self) -> None:
        s = _pipeline_state
        uptime = ""
        if s.get("started_at"):
            delta = int(time.time() - s["started_at"])
            h, m = divmod(delta, 3600)
            mm, ss = divmod(m, 60)
            uptime = f"{h}h {mm}m {ss}s"

        status = "<b>[RUNNING]</b>" if s["running"] else "<b>[IDLE]</b>"
        lines = [
            "<b>Pipeline Status</b>",
            "",
            f"State: {status}",
            f"Phase: <code>{html.escape(str(s['phase']))}</code>",
            f"Sweep: #{s['sweep']}",
            "▪ Process Workers: <b>4 (1 Master + 3 Workers)</b>",
        ]
        if uptime:
            lines.append(f"Uptime: {uptime}")

        # Search Persona
        try:
            from src.configuration import get_config

            cfg_persona = get_config().candidate.persona.strip()
            if cfg_persona:
                persona_lines = [
                    line_str.strip()
                    for line_str in cfg_persona.split("\n")
                    if line_str.strip()
                    and not line_str.strip().lower().startswith("candidate profile")
                ]
                short_persona = "\n".join(persona_lines[:3]) if persona_lines else cfg_persona[:200]
                lines.extend(
                    [
                        "",
                        "<b>Search Persona Focus</b>",
                        f"<code>{html.escape(short_persona)}</code>",
                    ]
                )
        except Exception:
            pass

        # Data & Volume Stats
        try:
            from src.graph.graph_store import GraphStore
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            graph = await GraphStore.create()
            try:
                resume_chunks = await store.chunk_count()
                async with store._pool.acquire() as conn:
                    obs_cnt = await conn.fetchval("SELECT COUNT(*) FROM job_observations") or 0
                    cand_cnt = await conn.fetchval("SELECT COUNT(*) FROM radar_candidates") or 0
                    src_cnt = await conn.fetchval("SELECT COUNT(*) FROM source_checkpoints") or 0
                    accepted_cnt = (
                        await conn.fetchval(
                            "SELECT COUNT(*) FROM radar_candidates WHERE eligibility = 'accepted'"
                        )
                        or 0
                    )

                node_rows = await graph._run("MATCH (n:GraphNode) RETURN count(n) AS cnt")
                graph_nodes = node_rows[0]["cnt"] if node_rows else 0

                lines.extend(
                    [
                        "",
                        "<b>Storage & Volume Stats</b>",
                        f"▪ Resume Chunks: <b>{resume_chunks}</b>",
                        f"▪ Active Sources: <b>{src_cnt}</b>",
                        f"▪ Job Observations: <b>{obs_cnt}</b>",
                        f"▪ Saved Candidates: <b>{cand_cnt}</b> (<b>{accepted_cnt}</b> accepted)",
                        f"▪ Neo4j Graph Nodes: <b>{graph_nodes}</b>",
                    ]
                )
            finally:
                await graph.close()
                await store.close()
        except Exception as e:
            logger.debug("Failed fetching storage stats for status", exc=e)

        queue_status = s.get("llm_queue", {})
        if queue_status:
            cooldown = "Active" if queue_status.get("cooldown_active") else "Clear"
            lines.extend(
                [
                    "",
                    "<b>LLM Queue</b>",
                    f"Pending: {queue_status.get('pending', 0)}",
                    f"In-flight: {queue_status.get('in_flight', 0)}",
                    f"RPM used: {queue_status.get('requests_this_minute', 0)}",
                    f"TPM used: {queue_status.get('tokens_this_minute', 0)}",
                    f"Cooldown: {cooldown}",
                    f"Total 429s: {queue_status.get('total_429s', 0)}",
                ]
            )

        rejection_counts = s.get("rejection_counts", [])
        if rejection_counts:
            lines.extend(["", "<b>Top Rejection Reasons</b>"])
            for rc in rejection_counts[:5]:
                reason = rc.get("reason", "?").replace("_", " ")
                lines.append(f"  ▪ {reason}: {rc.get('count', 0)}")

        source_health = s.get("source_health", {})
        if source_health:
            disabled = [sid for sid, sh in source_health.items() if not sh.get("active")]
            if disabled:
                lines.extend(["", f"Disabled sources: {len(disabled)}"])

        lines.extend(
            [
                "",
                f"Matched total: <b>{s['matched_total']}</b> | "
                f"Rejected total: <b>{s['rejected_total']}</b> | "
                f"Scraped this sweep: <b>{s['scraped_count']}</b>",
            ]
        )

        # Recent Logs (Last 5 clean non-noise log entries)
        try:
            from pathlib import Path

            log_path = Path("logs/run.log")
            if log_path.exists():
                recent_logs: list[str] = []
                with open(log_path, encoding="utf-8", errors="ignore") as f:
                    all_lines = [line.strip() for line in f if line.strip()]
                    for line_str in reversed(all_lines):
                        if len(recent_logs) >= 5:
                            break
                        if "firecrawl_tail" in line_str:
                            continue
                        if line_str.startswith("{") and "message" in line_str:
                            try:
                                import json

                                d = json.loads(line_str)
                                ts = d.get("timestamp", "").split("T")[-1][:8]
                                lvl = d.get("level", "INFO")
                                logger_n = d.get("logger", "sys")
                                msg = str(d.get("message", line_str))
                                tag = f"[{ts}] [{lvl}] {logger_n}"
                                recent_logs.insert(
                                    0,
                                    f"▪ <code>{html.escape(tag)}</code>: {html.escape(msg[:100])}",
                                )
                            except Exception:
                                recent_logs.insert(0, f"▪ {html.escape(line_str[:120])}")
                        else:
                            recent_logs.insert(0, f"▪ {html.escape(line_str[:120])}")
                if recent_logs:
                    lines.extend(["", "<b>Recent Activity Logs</b>"] + recent_logs)
        except Exception:
            pass

        sweep_start = s.get("sweep_started_at", 0)
        if sweep_start:
            interval = s.get("sweep_interval", 300)
            elapsed = time.time() - sweep_start
            if s["phase"].startswith("idle") and s["running"]:
                remaining = max(0, interval - elapsed)
                m, sec = divmod(int(remaining), 60)
                lines.append(f"\nNext sweep in: <b>{m}m {sec}s</b>")

        if s.get("last_error"):
            lines.extend(
                ["", "<b>Last error:</b>", f"<code>{html.escape(s['last_error'][:200])}</code>"]
            )
        lines.extend(["", "Send /health to check services."])
        await self._send_raw("\n".join(lines))

    async def _handle_health(self) -> None:
        report = await run_health_checks()
        await self._send_raw(report)

    async def _handle_resend(self, text: str) -> None:
        parts = text.split()
        args = [p.lower() for p in parts[1:]]
        is_dry = "--dry" in args or "dry" in args

        limit = 10
        for p in parts[1:]:
            if p.isdigit():
                limit = min(50, max(1, int(p)))
                break

        from src.memory.pgvector_store import MemoryStore

        try:
            store = await MemoryStore.create()
            try:
                async with store._pool.acquire() as conn:
                    total_row = await conn.fetchrow(
                        "SELECT COUNT(*) as cnt FROM radar_candidates "
                        "WHERE eligibility = 'accepted'"
                    )
                    total_count = total_row["cnt"] if total_row else 0

                    rows = await conn.fetch(
                        """
                        SELECT canonical_id, normalized_role, normalized_company,
                               normalized_location, direct_apply_url, match_percent,
                               shortlist_probability, verdict, funding_stage,
                               funding_info, salary_raw, company_description,
                               jd_summary, founders, founder_socials, osint_signals, extra
                        FROM radar_candidates
                        WHERE eligibility = 'accepted'
                        ORDER BY match_percent DESC, created_at DESC
                        LIMIT $1
                        """,
                        limit,
                    )

                if not rows:
                    await self._send_raw("No accepted job matches found in database to resend.")
                    return

                if is_dry:
                    lines = [
                        "<b>[DRY RUN] Available Accepted Jobs</b>",
                        f"<i>Total in DB: {total_count} accepted | Showing top {len(rows)}:</i>",
                        "",
                    ]
                    for idx, r in enumerate(rows, 1):
                        role = html.escape(str(r["normalized_role"] or "Position"))
                        company = html.escape(str(r["normalized_company"] or "Company"))
                        match_pct = r["match_percent"] or 0
                        loc = html.escape(str(r["normalized_location"] or "Remote"))
                        link = r["direct_apply_url"] or ""
                        line = f"<b>{idx}. {company}</b> — {role} ({match_pct}% match, {loc})"
                        if link and link.startswith("http"):
                            line += f' | <a href="{html.escape(link)}">Apply →</a>'
                        lines.append(line)

                    lines.extend(
                        [
                            "",
                            "<i>Run <code>/resend</code> to send cards for top 10, "
                            "or <code>/resend N</code> for top N.</i>",
                        ]
                    )
                    await self._send_raw("\n".join(lines))
                    return

                await self._send_raw(f"<i>Resending top {len(rows)} accepted job matches...</i>")

                count = 0
                for r in rows:
                    import json

                    def _parse_json(val: Any, default: Any) -> Any:
                        if isinstance(val, (dict, list)):
                            return val
                        if isinstance(val, str) and val.strip():
                            try:
                                return json.loads(val)
                            except Exception:
                                pass
                        return default

                    job_card = {
                        "role": r["normalized_role"],
                        "company": r["normalized_company"],
                        "match_percent": r["match_percent"],
                        "shortlist_probability": r["shortlist_probability"],
                        "salary": r["salary_raw"],
                        "location": r["normalized_location"],
                        "apply_link": r["direct_apply_url"],
                        "jd_summary": r["jd_summary"],
                        "company_description": r["company_description"],
                        "founders": _parse_json(r["founders"], []),
                        "funding_stage": r["funding_stage"],
                        "funding_info": _parse_json(r["funding_info"], {}),
                        "osint_signals": _parse_json(r["osint_signals"], []),
                        "founder_socials": _parse_json(r["founder_socials"], []),
                    }
                    ok = await self.send_categorized_alert("eligible", job_card, dedup_key="")
                    if ok:
                        count += 1
                        await asyncio.sleep(0.8)
            finally:
                await store.close()
        except Exception as e:
            logger.exception("Resend command failed", exc=e)
            await self._send_raw(f"<b>[ERROR] Resend failed:</b> <code>{str(e)[:200]}</code>")
            return

        await self._send_raw(f"<b>[RESEND] Sent {count} job alerts to Telegram.</b>")

    async def _handle_help(self) -> None:
        lines = [
            "<b>Commands</b>",
            "",
            "/status    – pipeline state + match count",
            "/health    – live service health check",
            "/analytics – market intelligence & skill arbitrage report",
            "/resend    – resend top accepted job alerts (e.g. /resend 10)",
            "/help      – this message",
            "",
            "I'll also notify you on pipeline errors, new matches,",
            "stealth hiring signals, and warm-intro paths.",
        ]
        await self._send_raw("\n".join(lines))

    # notifications

    async def send_error(self, message: str, dedup_key: str = "") -> None:
        if dedup_key and dedup_key in self._seen_errors:
            return
        if dedup_key:
            self._seen_errors.add(dedup_key)
        await self._send_raw(f"<b>[ERROR] Pipeline Error</b>\n\n<code>{message[:800]}</code>")
        logger.info("TelegramAgent sent error alert")

    async def send_startup(self, sweep_count: int = 0) -> None:
        metrics = get_system_metrics()

        src_cnt = 0
        cand_cnt = 0
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                async with store._pool.acquire() as conn:
                    src_cnt = await conn.fetchval("SELECT COUNT(*) FROM source_checkpoints") or 0
                    cand_cnt = await conn.fetchval("SELECT COUNT(*) FROM radar_candidates") or 0
            finally:
                await store.close()
        except Exception:
            pass

        short_persona = ""
        try:
            from src.configuration import get_config

            cfg_persona = get_config().candidate.persona.strip()
            if cfg_persona:
                persona_lines = [
                    line_str.strip()
                    for line_str in cfg_persona.split("\n")
                    if line_str.strip()
                    and not line_str.strip().lower().startswith("candidate profile")
                ]
                short_persona = "\n".join(persona_lines[:3]) if persona_lines else cfg_persona[:200]
        except Exception:
            pass

        lines = [
            "<b>[SYSTEM] Pipeline Started</b>",
            "",
            "<b>System Resources</b>",
            f"▪ RAM Usage: <b>{metrics['ram']}</b>",
            f"▪ Disk Space: <b>{metrics['disk']}</b>",
            f"▪ CPU Load: <b>{metrics['cpu_load']}</b>",
            "",
            "<b>Environment & Storage</b>",
            f"▪ Resume Chunks: <b>{sweep_count} vectors loaded</b>",
            f"▪ Registered Sources: <b>{src_cnt} active</b>",
            f"▪ Saved Candidates: <b>{cand_cnt} in database</b>",
            "▪ Scheduler: <b>8 async workers active</b>",
        ]

        if short_persona:
            lines.extend(
                [
                    "",
                    "<b>Search Persona Focus</b>",
                    f"<code>{html.escape(short_persona)}</code>",
                ]
            )

        lines.extend(["", "<i>Beginning discovery and polling sweeps...</i>"])
        await self._send_raw("\n".join(lines))

    async def send_stage_progress(
        self,
        stage: str,
        summary: str,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.is_configured:
            return

        metrics = get_system_metrics()
        lines = [
            f"<b>[PROGRESS] {html.escape(stage)}</b>",
            f"▪ {html.escape(summary)}",
            f"▪ RAM: <b>{metrics['ram']}</b> | CPU Load: <b>{metrics['cpu_load']}</b>",
        ]
        if extra_metrics:
            for k, v in extra_metrics.items():
                lines.append(f"▪ {html.escape(str(k))}: <b>{html.escape(str(v))}</b>")

        await self._send_raw("\n".join(lines))

    async def send_sweep_summary(
        self, sweep: int, matched: int, scraped: int, duration: float
    ) -> None:
        await self._send_raw(
            f"<b>[SWEEP] Sweep {sweep} Complete</b>\n\n"
            f"Scraped: {scraped}\n"
            f"Matched: {matched}\n"
            f"Duration: {duration:.1f}s"
        )

    # job card + inline keyboards

    def format_job_card(self, job: dict[str, Any]) -> str:
        role_raw = str(job.get("role") or "Software Engineer").strip()
        role = html.escape(" ".join(w[:1].upper() + w[1:] for w in role_raw.split()))
        company = html.escape(str(job.get("company") or "Company").strip())
        match_pct = job.get("match_percent", 0)
        shortlist_pct = job.get("shortlist_probability", 0)

        raw_sal = str(job.get("salary") or "").strip()
        salary_annual = job.get("salary_annual_usd")
        salary_estimated = bool(job.get("salary_estimated"))
        if raw_sal and raw_sal not in ("-", "Not specified", "N/A", "Flexible", "Competitive"):
            salary_str = html.escape(raw_sal)
            salary_estimated = salary_estimated and "est" not in raw_sal.lower()
        elif salary_annual:
            salary_str = f"${salary_annual:,.0f}/yr"
            salary_estimated = True
        else:
            salary_str = ""
        if salary_str and salary_estimated:
            src = str(job.get("salary_source") or "").strip()
            salary_str += f"  <i>(est. {html.escape(src)})</i>" if src else "  <i>(est.)</i>"

        location = html.escape(str(job.get("location") or "Remote").strip())
        raw_link = job.get("apply_link") or job.get("direct_apply_url") or job.get("url") or ""
        apply_link = str(raw_link).strip()

        # Location-eligibility warnings: this candidate needs visa sponsorship
        # to attend onsite roles, so flag them loudly on the card itself.
        from src.radar.core.signals import is_us_location

        loc_raw = str(job.get("location") or "Remote").strip()
        is_remote_role = bool(job.get("is_remote")) or "remote" in loc_raw.lower()
        is_us = is_us_location(loc_raw)
        warnings: list[str] = []
        if not is_remote_role:
            warnings.append("⚠ Onsite role - requires visa/relocation, may be rejected")
        if is_us and not job.get("sponsors_visa"):
            warnings.append("⚠ US role - visa sponsorship not confirmed")
        warning_block = ""
        if warnings:
            warning_block = "\n" + "\n".join(f"<b>{w}</b>" for w in warnings)

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

        badges: list[str] = []
        if job.get("sponsors_visa"):
            badges.append("visa sponsor")
        if job.get("is_remote") or "remote" in location.lower():
            badges.append("remote")
        if job.get("underdog_score") and float(job.get("underdog_score", 0)) > 0:
            badges.append("underdog")
        badges_str = f" · {' · '.join(badges)}" if badges else ""

        lines = [
            f"<b>{role}</b>",
            f"<b>{company}</b>",
            "<code>───────────────────────────</code>",
        ]
        if match_pct > 0 or shortlist_pct > 0:
            lines.append(
                f"Match <b>{match_pct}%</b> · Shortlist <b>{shortlist_pct}%</b>{badges_str}"
            )
        else:
            lines.append(f"<i>Newly surfaced</i>{badges_str}")
        lines.append(f"📍 {location}")
        if warning_block:
            lines.append(warning_block)
        if salary_str:
            lines.append(f"💰 {salary_str}")

        if apply_link and apply_link.startswith("http"):
            esc_link = html.escape(apply_link)
            lines.append(f'🔗 <a href="{esc_link}"><b>Apply →</b></a>')

        skills = job.get("matching_skills")
        if skills and isinstance(skills, list) and skills:
            lines.append(f"<b>Skills:</b> {html.escape(', '.join(str(s) for s in skills[:6]))}")

        if comp_desc:
            lines.extend(["", f"<blockquote>{comp_desc}</blockquote>"])

        funding_info = job.get("funding_info") or {}
        funding_stage = job.get("funding_stage", "")
        founders = job.get("founders", [])
        if isinstance(founders, str):
            founders = []
        osint_signals = job.get("osint_signals", [])
        if isinstance(osint_signals, str):
            osint_signals = []

        has_osint = bool(funding_info or funding_stage or founders or osint_signals)
        if has_osint:
            lines.extend(["", "<b>OSINT &amp; Outreach</b>", ""])

        if isinstance(funding_info, dict) and any(funding_info.values()):
            fi = funding_info
            parts = []
            if fi.get("round"):
                parts.append(f"▪ <b>{fi['round']}</b>")
            if fi.get("amount_raised"):
                parts.append(f"({fi['amount_raised']})")
            if fi.get("lead_investors"):
                parts.append(f"led by {', '.join(fi['lead_investors'])}")
            if fi.get("date_announced"):
                parts.append(f"[{fi['date_announced']}]")
            if parts:
                lines.append(" ".join(parts))
        elif funding_stage and funding_stage not in ("N/A", "-"):
            lines.append(f"▪ Funding: {funding_stage}")

        if founders:
            if isinstance(founders[0], dict):
                for f in founders:
                    name = html.escape(str(f.get("name", "?")))
                    title = html.escape(str(f.get("title", "")))
                    title_str = f" ({title})" if title else ""
                    badges = []
                    if f.get("email"):
                        email = str(f["email"])
                        badges.append(
                            f'<a href="mailto:{html.escape(email)}">✉️ {html.escape(email)}</a>'
                        )
                    if f.get("linkedin_url"):
                        url = str(f["linkedin_url"])
                        handle = url.rstrip("/").rsplit("/", 1)[-1] or "LinkedIn"
                        badges.append(f'<a href="{html.escape(url)}">in/{html.escape(handle)}</a>')
                    if f.get("github_url"):
                        url = str(f["github_url"])
                        handle = url.rstrip("/").rsplit("/", 1)[-1] or "GitHub"
                        badges.append(f'<a href="{html.escape(url)}">{html.escape(handle)}</a>')
                    badge_str = f" — {' | '.join(badges)}" if badges else ""
                    lines.append(f"▪ Founder: {name}{title_str}{badge_str}")
            else:
                lines.append(f"▪ Founders: {', '.join(html.escape(str(f)) for f in founders)}")
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
                lines.append(f"▪ Signal: {html.escape(str(sig))}")

        founder_posts = job.get("founder_posts", [])
        if founder_posts and isinstance(founder_posts, list):
            lines.extend(["", "<b>ACTIVE FOUNDER POST:</b>"])
            for fp in founder_posts[:2]:
                if not isinstance(fp, dict):
                    continue
                name = html.escape(str(fp.get("founder_name", "Unknown")))
                intent = html.escape(str(fp.get("intent", "")))
                post_url = fp.get("post_url", "")
                line = f"▪ <b>{name}</b>"
                if intent:
                    line += f" — {intent}"
                lines.append(line)
                if post_url.startswith("http"):
                    lines.append(f'└ <a href="{html.escape(post_url)}"><b>DM on LinkedIn →</b></a>')
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
        return await self._send_raw(text, "HTML", reply_markup)

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

    # categorized alerts (radar v2)

    _CATEGORY_ICONS: dict[str, str] = {
        "urgent": "[URGENT]",
        "startup_signal": "[SIGNAL]",
        "outreach": "[OUTREACH]",
        "eligible": "[ELIGIBLE]",
        "review": "[REVIEW]",
        "general_accepted": "[MATCH]",
    }

    _CATEGORY_LABELS: dict[str, str] = {
        "urgent": "Urgent High-Fit Verified Role",
        "startup_signal": "Startup Hiring Signal",
        "outreach": "Cold Outreach Opportunity",
        "eligible": "Eligible Role",
        "review": "Freshness Review Role",
        "general_accepted": "Matched Role",
    }

    async def send_categorized_alert(
        self,
        category: str,
        job: dict[str, Any],
        dedup_key: str = "",
    ) -> bool:
        if not self.is_configured:
            return False

        if dedup_key and dedup_key in self._notified_keys:
            return False

        icon = self._CATEGORY_ICONS.get(category, "[ALERT]")
        label = self._CATEGORY_LABELS.get(category, category)

        text = self.format_job_card(job)
        header = f"<b>{icon} {label}</b>\n\n"
        text = header + text

        buttons: list[list[dict[str, str]]] = []
        link = job.get("apply_link") or job.get("direct_apply_url") or job.get("url") or ""
        if link and str(link).startswith("http"):
            buttons.append([{"text": "Apply Direct", "url": link}])

        founders = job.get("founders", [])
        if founders and isinstance(founders[0], dict):
            for f in founders[:2]:
                if f.get("linkedin_url"):
                    name = html.escape(str(f.get("name", "Founder")))
                    buttons.append([{"text": f"LinkedIn: {name}", "url": f["linkedin_url"]}])

        reply_markup = {"inline_keyboard": buttons} if buttons else None
        success = await self._send_raw(text, "HTML", reply_markup)

        if success and dedup_key:
            self._notified_keys.add(dedup_key)

        return success

    async def send_category_digest(
        self,
        category: str,
        jobs: list[dict[str, Any]],
        max_jobs: int = 5,
    ) -> int:
        if not self.is_configured or not jobs:
            return 0

        icon = self._CATEGORY_ICONS.get(category, "[DIGEST]")
        label = self._CATEGORY_LABELS.get(category, category)

        lines = [f"<b>{icon} {label} Digest</b>", f"  <i>{len(jobs)} roles found</i>", ""]

        sent = 0
        for j in jobs[:max_jobs]:
            role = html.escape(str(j.get("role") or j.get("normalized_role") or "Position"))
            company = html.escape(str(j.get("company") or j.get("normalized_company") or "Company"))
            match_pct = j.get("match_percent", 0)
            location = html.escape(
                str(j.get("location") or j.get("normalized_location") or "Remote")
            )
            link = j.get("apply_link") or j.get("direct_apply_url", "")

            line = f"<b>{company}</b> — {role} ({match_pct}% match, {location})"
            if link and link.startswith("http"):
                line += f'\n  <a href="{html.escape(link)}">Apply →</a>'
            lines.append(line)
            lines.append("")
            sent += 1

        await self._send_raw("\n".join(lines))
        return sent

    # proactive stealth & warm-intro signals

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
            f"<b>[STEALTH SIGNAL] {name}</b>\n\n"
            f"<b>{name}</b> just surfaced with <b>{stage}</b> funding, "
            f"but has zero job postings.\n"
            f"Opportunity to bypass ATS entirely via direct outreach."
        )

        buttons: list[list[dict[str, str]]] = []
        if url:
            buttons.append([{"text": "Website", "url": url}])
        buttons.append(
            [
                {
                    "text": f"Search '{name}' on LinkedIn",
                    "url": (
                        f"https://www.linkedin.com/search/results/people/?keywords={name}%20founder"
                    ),
                }
            ]
        )

        await self._send_raw(
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
            f"<b>[WARM INTRO] {name}</b>",
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
                buttons.append([{"text": f"DM {founder}", "url": linkedin}])

        draft = ""
        if self.ctx is not None and paths:
            draft = await self._generate_cold_dm_draft(paths, target_company)

        if draft:
            lines.extend(["", f"<blockquote expandable>{html.escape(draft)}</blockquote>"])

        await self._send_raw(
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
