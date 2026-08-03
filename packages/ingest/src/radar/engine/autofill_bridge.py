#!/usr/bin/env python3
"""Bridge accepted radar candidates into the autofill queue.

The ingest pipeline scores jobs and persists the accepted ones in
``radar_candidates`` (eligibility = 'accepted', with a direct apply URL) but
nothing fed the autofill worker — jobs only entered ``autofill_queue`` via the
CLI. This bridge closes that gap: accepted candidates are enqueued unless the
link is already known to the queue (any status), so the stored corpus is
consumed exactly once per link. Idempotent, safe to call from the radar sweep
and from the loop driver.
"""

from __future__ import annotations

from typing import Any

from autofill.db import AutofillDB

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


def print_summary(summary: dict[str, int]) -> None:
    print(
        f"[bridge] queue: {summary['applied']} applied, {summary['open']} open, "
        f"{summary['deferred']} deferred, {summary['skipped']} skipped, "
        f"{summary['errored']} errored ({summary['failed']} failed) "
        f"[pending {summary['pending']}, filling {summary['filling']}, "
        f"awaiting_review {summary['awaiting_review']}]"
    )


async def drain_once(db: AutofillDB, limit: int) -> int:
    """Enqueue accepted radar candidates not already known to the queue."""
    async with db._pool.acquire() as conn:
        rows = await conn.fetch(DRAIN_QUERY, limit)
    enqueued = 0
    for r in rows:
        url = r["direct_apply_url"]
        if not url:
            continue
        if await db.link_known(url):
            continue
        await db.enqueue_job(
            apply_link=url,
            role=r["normalized_role"] or None,
            company=r["normalized_company"] or None,
            apply_mode="auto",
            source="radar",
        )
        enqueued += 1
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
