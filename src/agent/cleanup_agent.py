"""CleanupAgent: Periodic sanitization and deduplication engine for jobs.md.

Filters out non-undergrad roles (PhD/Senior), removes duplicates, normalizes table
formatting, and ensures jobs.md remains clean, crisp, and beautifully structured.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from src.agent.jobs_agent import JobsAgent, _normalize_key

if TYPE_CHECKING:
    from src.memory.pgvector_store import MemoryStore

NON_UNDERGRAD_KEYWORDS = [
    "phd", "ph.d", "doctorate", "postdoc",
    "senior", "sr\\.?", "staff",
    "principal", "architect", "director",
    "vice president", "vp(?=\\b|$)",
    "manager", "lead(?! engineer| developer| dev| tester| analyst| designer)",
    "head of",
    "content creator", "host live", "sales provider", "sales executive",
    "property development", "account executive",
    "marketing", "recruiter", "customer service", "customer support",
    "telemarketing", "social media", "administrative assistant",
    "store manager", "cashier", "driver",
]


def _role_has_senior_kw(role: str) -> bool:
    """Check if the job role/title contains senior/lead/manager keywords.
    Uses regex word boundaries so 'reports to the Engineering Manager'
    in the JD text does NOT match — we check the role title only."""
    title_kws = (
        r"\bsenior\b", r"\bsr\.?\b", r"\bstaff\b", r"\bmanager\b",
        r"\bdirector\b", r"\bvp\b", r"\bvice\s+president\b",
        r"\bhead\s+of\b", r"\barchitect\b", r"\bprincipal\b",
        r"\bph\.?d\b", r"\bdoctorate\b", r"\bpostdoc\b",
    )
    return any(re.search(pat, role) for pat in title_kws)


class CleanupAgent:
    """Agent that cleans up and formats jobs.md into a pristine Markdown table."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store
        self.jobs_agent = JobsAgent(store=store)

    def is_valid_undergrad_role(self, job: dict[str, Any]) -> bool:
        """Check if role matches undergrad candidate constraints."""
        role = str(job.get("role") or "").lower()
        company = str(job.get("company") or "").lower()

        if role in ("", "n/a", "unknown", "-") or company in ("", "n/a", "unknown", "-"):
            return False

        if _role_has_senior_kw(role):
            return False

        # Broader non-tech check against role + company_description
        jd_summary = str(job.get("jd_summary") or "").lower()
        comp_desc = str(job.get("company_description") or "").lower()
        combined = f"{role} {company} {jd_summary} {comp_desc}"

        for kw in NON_UNDERGRAD_KEYWORDS:
            if re.search(r"\b" + kw + r"\b", combined):
                return False

        return not (
            job.get("verdict") == "NO_MATCH"
            and int(job.get("match_percent", 0)) < 30
        )

    async def clean_and_format_ledger(self) -> list[dict[str, Any]]:
        """Retrieve ledger, deduplicate, filter non-undergrad roles, and format."""
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

        clean_jobs.sort(key=lambda x: int(x.get("match_percent", 0)), reverse=True)
        self.jobs_agent._atomic_write_md(clean_jobs)
        print(
            f"  🧹 [CleanupAgent] Sanitized & formatted jobs.md "
            f"({len(clean_jobs)} clean positions)"
        )
        return clean_jobs
