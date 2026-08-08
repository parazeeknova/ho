"""Per-sweep submission summary email (the review's ask: ONE email per sweep).

After a sweep/epoch ends, send a single email listing every confirmed
submission and the fields that were filled for it, using the Gmail app
password (GMAIL_EMAIL + GMAIL_APP_PASSWORD). One thread per sweep: we use a
fixed subject prefix so Gmail threads replies together, and we do not send a
separate mail per job.
"""

from __future__ import annotations

import asyncio
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

from src.logging import get_logger

logger = get_logger("autofill.src.outcomes.email_summary")

_SUBJECT_PREFIX = "[ho] Sweep submission summary"


def _smtp_config() -> dict[str, str]:
    return {
        "host": os.getenv("GMAIL_SMTP_HOST", "smtp.gmail.com"),
        "port": os.getenv("GMAIL_SMTP_PORT", "587"),
        "user": os.getenv("GMAIL_EMAIL", ""),
        "password": os.getenv("GMAIL_APP_PASSWORD", ""),
        "to": os.getenv("GMAIL_EMAIL", ""),
    }


def is_configured() -> bool:
    cfg = _smtp_config()
    return bool(cfg["user"] and cfg["password"] and cfg["to"])


def _format_fields(fills: Any) -> str:
    """Render a job's filled fields as a readable list."""
    if not fills:
        return "  (no field-level fills recorded)"
    # asyncpg returns jsonb aggregates as one JSON string — parse it once.
    if isinstance(fills, str):
        import json

        try:
            fills = json.loads(fills)
        except Exception:
            return "  (no field-level fills recorded)"
    if not isinstance(fills, list):
        return "  (no field-level fills recorded)"
    lines = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        q = (f.get("question") or "").strip()[:80]
        a = (f.get("answer") or "").strip()
        if not q:
            continue
        src = (f.get("source") or "?").strip()
        if len(a) > 120:
            a = a[:117] + "..."
        lines.append(f"  - {q}: {a or '<blank>'}  [{src}]")
    return "\n".join(lines) if lines else "  (no field-level fills recorded)"


def build_summary_body(
    sweep_label: str,
    submissions: list[dict[str, Any]],
    epoch_id: str | None = None,
    extra: str = "",
) -> str:
    """Assemble the full plain-text summary for one sweep."""
    lines = [
        f"Sweep: {sweep_label}",
        f"Confirmed submissions: {len(submissions)}",
    ]
    if epoch_id:
        lines.append(f"Learning epoch: {epoch_id}")
    lines.append("")
    if not submissions:
        lines.append("No applications were confirmed submitted in this sweep.")
    for i, s in enumerate(submissions, 1):
        company = (s.get("company") or "?").strip()
        role = (s.get("role") or "").strip()
        url = (s.get("apply_link") or s.get("url") or "").strip()
        lines.append(f"{i}. {company} — {role}")
        if url:
            lines.append(f"   {url}")
        fills = s.get("fills") or []
        lines.append(_format_fields(fills))
        lines.append("")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def _send_sync(body: str, sweep_label: str) -> bool:
    """Blocking SMTP send (called via asyncio.to_thread)."""
    cfg = _smtp_config()
    if not (cfg["user"] and cfg["password"]):
        logger.warning("email summary skipped: GMAIL_EMAIL/GMAIL_APP_PASSWORD not set")
        return False
    msg = EmailMessage()
    msg["Subject"] = f"{_SUBJECT_PREFIX} — {sweep_label}"
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(body)
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=30) as server:
            server.starttls(context=ctx)
            server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
        return True
    except Exception as e:
        logger.warning("email summary send failed", error=str(e))
        return False


async def send_sweep_summary(
    sweep_label: str,
    submissions: list[dict[str, Any]],
    epoch_id: str | None = None,
    extra: str = "",
) -> bool:
    """Send ONE summary email for a sweep's confirmed submissions."""
    if not is_configured():
        logger.info("email summary disabled (GMAIL_EMAIL/GMAIL_APP_PASSWORD unset)")
        return False
    body = build_summary_body(sweep_label, submissions, epoch_id=epoch_id, extra=extra)
    ok = await asyncio.to_thread(_send_sync, body, sweep_label)
    if ok:
        logger.info("sweep summary email sent", sweep=sweep_label, count=len(submissions))
    return ok
