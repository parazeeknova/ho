"""TelegramAgent: Delivers real-time, beautifully formatted job match notifications
to Telegram once all agent verifications and enrichment steps complete.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAgent:
    """Agent responsible for dispatching complete, verified job alerts to Telegram."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._notified_keys: set[str] = set()

    @property
    def is_configured(self) -> bool:
        """Return True if Telegram Bot token and Chat ID are configured."""
        return bool(self.bot_token and self.chat_id)

    def format_job_card(self, job: dict[str, Any]) -> str:
        """Format a single complete job listing into Telegram HTML markup."""
        role = str(job.get("role") or "Software Engineer").strip()
        company = str(job.get("company") or "Company").strip()
        match_pct = job.get("match_percent", 0)
        shortlist_pct = job.get("shortlist_probability", 0)
        salary = str(job.get("salary") or "Not specified").strip()
        location = str(job.get("location") or "Remote").strip()
        link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""

        comp_desc = str(
            job.get("company_description") or job.get("jd_summary") or job.get("role_summary") or ""
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
                f'<a href="{s}">{s.split("//")[-1]}</a>' if s.startswith("http") else s
                for s in socials[:2]
            ]
            lines.append(f"<b>Outreach:</b> {', '.join(social_links)}")

        if link and str(link).startswith("http"):
            lines.extend(["", f'<a href="{link}"><b>Apply Direct →</b></a>'])

        return "\n".join(lines)

    async def send_notification(self, job: dict[str, Any]) -> bool:
        """Send a single verified job card to the configured Telegram chat."""
        if not self.is_configured:
            return False

        card_html = self.format_job_card(job)
        url = TELEGRAM_API_URL.format(token=self.bot_token)

        payload = {
            "chat_id": self.chat_id,
            "text": card_html,
            "parse_mode": "HTML",
            "disable_web_page_preview": False,
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(url, json=payload)
                return resp.status_code == 200
        except Exception as e:
            print(f"  [dim]Telegram dispatch failed: {e}[/dim]")
            return False

    async def notify_verified_jobs(
        self, jobs: list[dict[str, Any]], min_match_pct: int = 40
    ) -> int:
        """Iterate over verified jobs one by one and dispatch notifications."""
        if not self.is_configured:
            return 0

        sent_count = 0
        for j in jobs:
            role = str(j.get("role") or "").strip()
            company = str(j.get("company") or "").strip()
            match_pct = int(j.get("match_percent", 0))

            # Strictly verify complete fields before sending
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

            success = await self.send_notification(j)
            if success:
                self._notified_keys.add(dedup_key)
                sent_count += 1
                print(f"  📱 [TelegramAgent] Sent alert for {role} @ {company}")
                await asyncio.sleep(1.2)  # Rate limiting between Telegram dispatches

        return sent_count
