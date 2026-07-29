"""CleanupAgent: Periodic sanitization and deduplication engine for jobs.md.

Filters out non-undergrad roles (PhD/Senior), removes duplicates, normalizes table formatting,
and ensures jobs.md remains clean, crisp, and beautifully structured.
"""

from __future__ import annotations

import re
from typing import Any

from src.agent.jobs_agent import JobsAgent, _normalize_key

NON_UNDERGRAD_KEYWORDS = [
    "phd",
    "ph.d",
    "doctorate",
    "postdoc",
    "senior",
    "sr.",
    "staff",
    "principal",
    "architect",
    "director",
    "vice president",
]


class CleanupAgent:
    """Agent that cleans up and formats jobs.md into a pristine Markdown table."""

    def __init__(self, jobs_agent: JobsAgent | None = None) -> None:
        self.jobs_agent = jobs_agent or JobsAgent()

    def is_valid_undergrad_role(self, job: dict[str, Any]) -> bool:
        """Check if role matches undergrad candidate constraints."""
        role = str(job.get("role") or "").lower()
        company = str(job.get("company") or "").lower()
        jd_summary = str(job.get("jd_summary") or "").lower()
        comp_desc = str(job.get("company_description") or "").lower()
        combined = f"{role} {company} {jd_summary} {comp_desc}"

        if role in ("", "n/a", "unknown", "-") or company in ("", "n/a", "unknown", "-"):
            return False

        # Reject PhD, Doctorate, Senior, or Staff roles
        for kw in NON_UNDERGRAD_KEYWORDS:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined):
                return False

        # Reject 0% match NO_MATCH roles
        return not (job.get("verdict") == "NO_MATCH" and int(job.get("match_percent", 0)) < 30)

    async def clean_and_format_ledger(self) -> list[dict[str, Any]]:
        """Retrieve Qdrant ledger, deduplicate, filter non-undergrad roles, and format."""
        all_jobs = await self.jobs_agent.get_all_jobs()
        seen_keys: set[str] = set()
        clean_jobs: list[dict[str, Any]] = []

        for j in all_jobs:
            if not self.is_valid_undergrad_role(j):
                continue

            key = _normalize_key(j.get("company", ""), j.get("role", ""))
            if key and key not in seen_keys:
                seen_keys.add(key)
                clean_jobs.append(j)

        # Sort by match_percent descending
        clean_jobs.sort(key=lambda x: int(x.get("match_percent", 0)), reverse=True)

        # Atomically rewrite jobs.md with clean table formatting
        self.jobs_agent._atomic_write_md(clean_jobs)
        print(
            f"  🧹 [CleanupAgent] Sanitized & formatted jobs.md ({len(clean_jobs)} clean positions)"
        )
        return clean_jobs
