#!/usr/bin/env python3
"""Bridge accepted radar candidates into the autofill queue.

The ingest pipeline scores jobs and persists the accepted ones in
``radar_candidates`` (eligibility = 'accepted', with a direct apply URL) but
nothing fed the autofill.src.core.worker — jobs only entered ``autofill_queue`` via the
CLI. This bridge closes that gap: accepted candidates are enqueued unless the
link is already known to the queue (any status), so the stored corpus is
consumed exactly once per link. Idempotent, safe to call from the radar sweep
and from the loop driver.
"""

from __future__ import annotations

import re
from typing import Any

from autofill.src.core.db import AutofillDB

from src.logging import get_logger

logger = get_logger("autofill_bridge")

DRAIN_QUERY = """
SELECT canonical_id, normalized_role, normalized_company, direct_apply_url
FROM radar_candidates
WHERE eligibility = 'accepted'
  AND direct_apply_url IS NOT NULL
  AND direct_apply_url != ''
ORDER BY updated_at DESC
LIMIT $1
"""

# URLs that are clearly NOT a direct job application form. The matcher can
# mark a company careers page / solutions page as a STRONG_MATCH for the
# company, but there is no application form to fill there — enqueuing them
# makes the runner hang on a non-form and burns the queue. These are skipped.
# Job-SPECIFIC pages (careers/jobs/{id}, careers/opportunity/{id}, job
# detail URLs with a job id) are KEPT — only bare career/index/solutions
# landing pages are rejected.
_NON_APPLY_URL_RE = re.compile(
    r"(?:/careers?/?$|/careers?/(?:overview|index|all|list|search|browse|home)"
    r"|/careers?/[a-z]{2}/?$"  # locale-only careers page (/careers/de/)
    r"|/careers/open-positions"
    r"|/solutions|/workato|/company/?$|/about/?$|/it/?$|/fr/?$|/nl/?$|/es/?$)"
    r"|linkedin\.com/(?:jobs|company)/[^/]*?\?(?:trk|ref)="  # LinkedIn search/browse URLs
    r"|linkedin\.com/jobs/research|linkedin\.com/jobs/[^/]+-jobs"
    r"|(?:careers?\b.*(?:overview|index|landing))",
    re.I,
)


def print_summary(summary: dict[str, int]) -> None:
    print(
        f"[bridge] queue: {summary['applied']} applied, {summary['open']} open, "
        f"{summary['deferred']} deferred, {summary['skipped']} skipped, "
        f"{summary['errored']} errored ({summary['failed']} failed) "
        f"[pending {summary['pending']}, filling {summary['filling']}, "
        f"awaiting_review {summary['awaiting_review']}]"
    )


async def drain_once(db: AutofillDB, limit: int) -> int:
    """Enqueue accepted radar candidates not already known to the queue.

    Idempotent per link. Also honors a per-company application cap: once a
    company already has ``AUTOFILL_MAX_PER_COMPANY`` (default 3) submitted or
    in-flight applications, further accepted candidates from that company are
    NOT enqueued — applying to a dozen roles at one company looks spammy to
    the recruiter and trips the ATS's one-application-per-candidate filter.
    """
    async with db._pool.acquire() as conn:
        rows = await conn.fetch(DRAIN_QUERY, limit)
    import os as _os

    max_per_company = int(_os.environ.get("AUTOFILL_MAX_PER_COMPANY", "3"))
    # Count of submitted/in-flight (non-terminal) jobs per company — the number
    # of applications that would be (or already are) live at that company.
    active_per_company: dict[str, int] = {}
    async with db._pool.acquire() as conn:
        actives = await conn.fetch(
            """
            SELECT company, COUNT(*) AS n
            FROM autofill_queue
            WHERE status IN ('pending', 'filling', 'awaiting_review', 'submitted')
              AND company IS NOT NULL
            GROUP BY company
            """
        )
        for r in actives:
            active_per_company[str(r["company"] or "").strip().lower()] = r["n"]
    enqueued = 0
    for r in rows:
        url = r["direct_apply_url"]
        if not url:
            continue
        if _NON_APPLY_URL_RE.search(url):
            logger.info(
                "Skipping enqueue: not a direct job application URL",
                url=url,
                company=r["normalized_company"],
            )
            continue
        if await db.link_known(url):
            continue
        company = str(r["normalized_company"] or "").strip()
        if company and active_per_company.get(company.lower(), 0) >= max_per_company:
            logger.info(
                "Skipping enqueue: company at application cap",
                company=company,
                active=active_per_company.get(company.lower()),
                cap=max_per_company,
            )
            continue
        await db.enqueue_job(
            apply_link=url,
            role=r["normalized_role"] or None,
            company=company or None,
            apply_mode="auto",
            source="radar",
        )
        enqueued += 1
        if company:
            active_per_company[company.lower()] = active_per_company.get(company.lower(), 0) + 1
    if enqueued:
        logger.info("Drained accepted candidates into autofill queue", count=enqueued)
    return enqueued


async def queue_balance(db: AutofillDB) -> dict[str, Any]:
    """Queue summary plus a live count of drainable accepted candidates."""
    summary = await db.queue_summary()
    async with db._pool.acquire() as conn:
        summary["drainable_accepted"] = await conn.fetchval(
            """
            SELECT COUNT(*) FROM radar_candidates
            WHERE eligibility = 'accepted'
              AND direct_apply_url IS NOT NULL
              AND direct_apply_url != ''
            """
        )
    return summary
