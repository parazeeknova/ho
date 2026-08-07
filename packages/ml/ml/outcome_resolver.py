"""OutcomeResolver — email → job_id with confidence.

An ATS email rarely contains a job_id. We resolve by sender domain →
company, subject/body → role, then match against recent applied jobs
(application_outcomes / autofill_queue). Low-confidence matches go to
unattributed_outcomes and are resolved later — never fabricated.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _fuzzy(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# Known ATS sender → company extraction
SENDER_COMPANY_RE = re.compile(r"@([a-z0-9.-]+)")


def company_from_sender(sender: str) -> str | None:
    m = SENDER_COMPANY_RE.search(sender or "")
    if not m:
        return None
    host = m.group(1).lower()
    # Map known ATS relay hosts to None (they mask the company)
    if any(h in host for h in ("greenhouse", "lever.co", "ashbyhq", "workable", "smartrecruiters")):
        return None
    # Otherwise the host itself hints the company
    return host.split(".")[0]


def extract_role_from_subject(subject: str) -> str | None:
    # Common patterns: "Re: Backend Engineer at Acme", "Update on ... Engineer"
    m = re.search(
        r"(?:re:|update on|regarding|for)\s+(.+?)(?:\s+at\s+|\s+—|\s+-|$)", subject or "", re.IGNORECASE
    )
    if m:
        return m.group(1).strip()[:80]
    return None


async def resolve_outcome(
    store: Any,
    email: dict[str, Any],
) -> dict[str, Any]:
    """Resolve an ATS email to a job. Returns {job_id, confidence, ...}."""
    subject = email.get("subject", "")
    sender = email.get("from", "") or email.get("sender", "")
    body = email.get("snippet", "") or email.get("body", "")

    # Fetch recent applied jobs as candidates
    try:
        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id, company, role, board
                FROM application_outcomes
                WHERE created_at > NOW() - INTERVAL '60 days'
                ORDER BY created_at DESC LIMIT 50
                """
            )
    except Exception:
        return {"job_id": None, "confidence": 0.0, "reason": "db_unavailable"}

    if not rows:
        return {"job_id": None, "confidence": 0.0, "reason": "no_recent_applications"}

    sender_company = company_from_sender(sender)
    email_role = extract_role_from_subject(subject)

    best: dict[str, Any] | None = None
    best_score = 0.0
    for r in rows:
        score = 0.0
        company = r["company"] or ""
        role = r["role"] or ""
        # Company signal
        if sender_company and company and _fuzzy(sender_company, company) > 0.7:
            score += 0.5
        elif company and company.lower() in subject.lower():
            score += 0.4
        # Role signal
        if email_role and role and _fuzzy(email_role, role) > 0.6:
            score += 0.4
        elif subject and role and _fuzzy(subject, role) > 0.5:
            score += 0.2
        if score > best_score:
            best_score = score
            best = dict(r)

    if best and best_score >= 0.6:
        return {"job_id": best["job_id"], "confidence": round(best_score, 2), "matched": best}
    if best and best_score >= 0.35:
        return {
            "job_id": best["job_id"],
            "confidence": round(best_score, 2),
            "matched": best,
            "needs_review": True,
        }
    return {"job_id": None, "confidence": round(best_score, 2), "reason": "low_confidence"}
