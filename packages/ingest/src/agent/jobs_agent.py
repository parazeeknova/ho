"""JobsAgent: Persistent job ledger backed by pgvector (MemoryStore).

Features:
- Server-side upsert with smart merge (highest match_percent wins)
- Deduplication by company:role key
- Atomic safe file replacement for jobs.md
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from src.output.writer import compute_days_ago

if TYPE_CHECKING:
    from src.memory.pgvector_store import MemoryStore


def _normalize_key(company: str, role: str, location: str = "Remote") -> str:
    c = "".join(ch for ch in company.lower() if ch.isalnum())
    r = "".join(ch for ch in role.lower() if ch.isalnum())
    loc = "".join(ch for ch in location.lower() if ch.isalnum())
    return f"{c}:{r}:{loc}"


class JobsAgent:
    def __init__(
        self,
        output_path: str = "jobs.md",
        store: MemoryStore | None = None,
    ) -> None:
        self.output_path = output_path
        self.store = store

    async def add_or_merge_jobs(
        self,
        new_jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Intelligently merge new jobs into the pgvector ledger."""
        if not self.store or not new_jobs:
            return []

        for job in new_jobs:
            company = str(job.get("company") or "Unknown")
            role = str(job.get("role") or "Position")
            key = _normalize_key(company, role, job.get("location", "Remote"))

            apply_link = job.get("apply_link") or job.get("source_url") or job.get("url") or ""
            if not apply_link or not str(apply_link).startswith("http"):
                apply_link = str(job.get("url", ""))

            payload = {
                "role": role,
                "company": company,
                "company_description": job.get("company_description", ""),
                "role_summary": job.get("role_summary", ""),
                "is_startup": job.get("is_startup", False),
                "founders": job.get("founders", []),
                "funding_stage": job.get("funding_stage", ""),
                "funding_info": job.get("funding_info", {}),
                "founder_socials": job.get("founder_socials", []),
                "company_news": job.get("company_news", ""),
                "osint_signals": job.get("osint_signals", []),
                "match_percent": int(job.get("match_percent", 0)),
                "shortlist_probability": int(job.get("shortlist_probability", 0)),
                "salary": job.get("salary"),
                "posted_date": job.get("posted_date"),
                "location": job.get("location") or "Remote",
                "apply_link": apply_link,
                "jd_summary": job.get("jd_summary", ""),
                "verdict": job.get("verdict", "NO_MATCH"),
                "source_url": job.get("source_url", job.get("url", "")),
            }
            await self.store.upsert_job_ledger(key, payload)

        all_jobs = await self.get_all_jobs()
        self._atomic_write_md(all_jobs)
        return all_jobs

    async def get_all_jobs(self) -> list[dict[str, Any]]:
        if not self.store:
            return []
        return await self.store.get_all_jobs_ledger()

    def _atomic_write_md(self, jobs: list[dict[str, Any]]) -> None:
        """Atomically update jobs.md."""
        from datetime import UTC, datetime

        now_str = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "# Job Matches",
            "",
            f"Generated: {now_str}",
            "",
            "| # | Role | Company | Company Info | JD Match | Shortlist% | Salary | Posted | Location | Apply |",  # noqa: E501
            "|---|------|---------|--------------|----------|------------|--------|--------|----------|-------|",
        ]

        valid_jobs = []
        for j in jobs:
            role = str(j.get("role") or "").strip()
            company = str(j.get("company") or "").strip()
            verdict = str(j.get("verdict") or "").upper()
            match_pct = int(j.get("match_percent", 0))

            if role in ("", "N/A", "Unknown", "-") or company in ("", "N/A", "Unknown", "-"):
                continue
            if verdict == "NO_MATCH" and match_pct < 30:
                continue

            valid_jobs.append(j)

        valid_jobs.sort(key=lambda x: int(x.get("match_percent", 0)), reverse=True)

        for i, j in enumerate(valid_jobs, start=1):
            role = (
                str(j.get("role") or "-").replace("|", "\\|").replace("\n", " ").replace("\r", "")
            )
            company = (
                str(j.get("company") or "-")
                .replace("|", "\\|")
                .replace("\n", " ")
                .replace("\r", "")
            )
            comp_info = (
                str(
                    j.get("company_description")
                    or j.get("jd_summary")
                    or j.get("role_summary")
                    or "-"
                )
                .strip()
                .replace("|", "\\|")
                .replace("\n", " ")
                .replace("\r", "")
            )
            if len(comp_info) > 60:
                comp_info = comp_info[:57] + "..."

            match_str = f"{j.get('match_percent', 0)}%"
            shortlist_str = f"{j.get('shortlist_probability', 0)}%"
            salary = (
                str(j.get("salary") or "-").replace("|", "\\|").replace("\n", " ").replace("\r", "")
            )

            posted_raw = j.get("posted_date")
            posted = compute_days_ago(posted_raw) if posted_raw else "-"

            location = (
                str(j.get("location") or "Remote")
                .replace("|", "\\|")
                .replace("\n", " ")
                .replace("\r", "")
            )

            link = j.get("apply_link") or j.get("source_url") or j.get("url") or ""
            if link and isinstance(link, str) and link.startswith("http"):
                link_md = f"[Apply]({link})"
            else:
                link_md = "-"

            row_str = (
                f"| {i} | {role} | {company} | {comp_info} | {match_str} | "
                f"{shortlist_str} | {salary} | {posted} | {location} | {link_md} |"
            )
            lines.append(row_str)

        lines.extend(["", f"*{len(valid_jobs)} positions matched*", ""])

        lines.extend(["---", "## Detailed Position Insights", ""])
        for i, j in enumerate(valid_jobs, start=1):
            role = str(j.get("role") or "Position")
            company = str(j.get("company") or "Company")
            comp_desc = str(j.get("company_description") or j.get("jd_summary") or "").strip()
            role_desc = str(j.get("role_summary") or j.get("jd_summary") or "").strip()
            link = j.get("apply_link") or j.get("source_url") or j.get("url") or ""

            founders = j.get("founders", [])
            funding = j.get("funding_stage")
            socials = j.get("founder_socials", [])
            news = j.get("company_news")

            if isinstance(founders, str):
                try:
                    import json as _json

                    parsed = _json.loads(founders)
                    founders = parsed if isinstance(parsed, list) else []
                except Exception:
                    founders = []
            founder_names: list[str] = []
            for f in founders:
                n = f.get("name") if isinstance(f, dict) else f
                if n:
                    founder_names.append(str(n))

            lines.append(f"### {i}. {role} @ {company}")
            if comp_desc:
                lines.append(f"**Company Overview**: {comp_desc}")
            if role_desc:
                lines.append(f"**Role Focus**: {role_desc}")
            if founder_names:
                lines.append(f"**Founders / Leadership**: {', '.join(founder_names)}")
            if funding:
                lines.append(f"**Funding Stage**: {funding}")
            if news:
                lines.append(f"**Recent News**: {news}")
            if socials:
                social_links = [f"[{s}]({s})" if s.startswith("http") else s for s in socials]
                lines.append(f"**Outreach Links**: {', '.join(social_links)}")
            if link and str(link).startswith("http"):
                lines.append(f"**Apply Direct**: [{link}]({link})")
            lines.append("")

        content = "\n".join(lines)

        tmp_file = f"{self.output_path}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(content)

        os.replace(tmp_file, self.output_path)
