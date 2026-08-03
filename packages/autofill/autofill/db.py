"""PostgreSQL queue and state persistence for autofill service."""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Any

import asyncpg
from src.configuration import PostgresConfig, get_config
from src.logging import get_logger

logger = get_logger("autofill.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS autofill_queue (
    job_id          TEXT PRIMARY KEY,
    apply_link      TEXT NOT NULL,
    role            TEXT,
    company         TEXT,
    ats_platform    TEXT DEFAULT '',
    apply_mode      TEXT DEFAULT 'review',
    status          TEXT DEFAULT 'pending',
    lease_expires   TIMESTAMP WITH TIME ZONE,
    retries         INTEGER DEFAULT 0,
    filled_payload  JSONB DEFAULT '{}'::jsonb,
    screenshot_path TEXT DEFAULT '',
    error           TEXT DEFAULT '',
    created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_autofill_queue_poll ON autofill_queue(status, lease_expires);
CREATE INDEX IF NOT EXISTS idx_autofill_queue_link ON autofill_queue(apply_link);

-- Idempotent migrations for overnight defer + morning digest support.
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS pending_questions JSONB DEFAULT '[]'::jsonb;
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS summary_sent BOOLEAN DEFAULT FALSE;

-- Per-job Q&A audit trail: every screener question the autofill answered for
-- a posting, so we can always reconstruct what was filled and from where.
-- job_id is deliberately NOT a foreign key (a single CLI run may fill without
-- a queue row).
CREATE TABLE IF NOT EXISTS autofill_fills (
    id          BIGSERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT,
    source      TEXT,
    options     JSONB DEFAULT '[]'::jsonb,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '2 days'
);
CREATE INDEX IF NOT EXISTS idx_autofill_fills_job ON autofill_fills(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_autofill_fills_expiry ON autofill_fills(expires_at);
"""


class AutofillDB:
    """Async connection-pool-backed queue store for job autofill state."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, config: PostgresConfig | None = None) -> AutofillDB:
        """Initialise pool and create table if not exists."""
        cfg = config or get_config().postgres
        pool = await asyncpg.create_pool(cfg.dsn, min_size=1, max_size=cfg.max_pool)
        async with pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("AutofillDB initialized", dsn=cfg.dsn.split("@")[-1])
        return cls(pool)

    async def close(self) -> None:
        """Close connection pool."""
        await self._pool.close()
        logger.info("AutofillDB closed")

    async def enqueue_job(
        self,
        apply_link: str,
        role: str | None = None,
        company: str | None = None,
        ats_platform: str = "",
        apply_mode: str = "review",
    ) -> str:
        """Enqueue a new job application. Returns job_id.

        Deduplicates against active rows (pending / filling / awaiting_review /
        deferred) for the same apply_link, returning the existing job_id.
        """
        async with self._pool.acquire() as conn:
            existing = await conn.fetchval(
                """
                SELECT job_id FROM autofill_queue
                WHERE apply_link = $1
                  AND status IN ('pending', 'filling', 'awaiting_review', 'deferred')
                ORDER BY created_at ASC
                LIMIT 1
                """,
                apply_link,
            )
            if existing:
                logger.info(
                    "Enqueue skipped: active row exists for link",
                    job_id=existing,
                    apply_link=apply_link,
                )
                return existing

            job_id = f"job-{uuid.uuid4().hex[:8]}"
            await conn.execute(
                """
                INSERT INTO autofill_queue (
                    job_id, apply_link, role, company, ats_platform, apply_mode, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, 'pending')
                """,
                job_id,
                apply_link,
                role,
                company,
                ats_platform,
                apply_mode,
            )
        logger.info("Job enqueued", job_id=job_id, apply_link=apply_link, mode=apply_mode)
        return job_id

    async def claim_next_job(self, lease_seconds: int = 600) -> dict[str, Any] | None:
        """Atomically claim the next pending job or expired lease using FOR UPDATE SKIP LOCKED."""
        query = """
        UPDATE autofill_queue
        SET status = 'filling',
            lease_expires = NOW() + ($1::int * INTERVAL '1 second'),
            retries = retries + 1,
            updated_at = NOW()
        WHERE job_id = (
            SELECT job_id FROM autofill_queue
            WHERE status = 'pending'
               OR (status IN ('filling', 'awaiting_review') AND lease_expires < NOW())
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING job_id, apply_link, role, company, ats_platform, apply_mode,
                  status, retries, filled_payload, screenshot_path;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, lease_seconds)
            if not row:
                return None
            result = dict(row)
            if isinstance(result.get("filled_payload"), str):
                result["filled_payload"] = json.loads(result["filled_payload"])
            return result

    async def update_status(
        self,
        job_id: str,
        status: str,
        filled_payload: dict[str, Any] | None = None,
        screenshot_path: str | None = None,
        error: str | None = None,
        override_terminal: bool = False,
    ) -> bool:
        """Update status and payload of a job.

        Guarded: a terminal status (``deferred``, ``submitted``) is never
        overwritten by a later non-terminal transition, so a deferred job that
        still reaches the review step keeps its deferred status and stays in
        the morning digest. The resume flow (``run_resume``) passes
        ``override_terminal=True``: after the user answers the deferred
        questions, clearing ``deferred`` to ``skipped``/``failed`` is exactly
        what is wanted, not a downgrade.
        """
        if not override_terminal and status not in ("deferred", "submitted"):
            current = await self.get_job(job_id)
            if current and current.get("status") in ("deferred", "submitted"):
                logger.info(
                    "Skipping status update: job is terminal",
                    job_id=job_id,
                    current=current.get("status"),
                    requested=status,
                )
                return False

        updates = ["status = $2", "updated_at = NOW()"]
        args: list[Any] = [job_id, status]

        if filled_payload is not None:
            args.append(json.dumps(filled_payload))
            updates.append(f"filled_payload = ${len(args)}::jsonb")

        if screenshot_path is not None:
            args.append(screenshot_path)
            updates.append(f"screenshot_path = ${len(args)}")

        if error is not None:
            args.append(error)
            updates.append(f"error = ${len(args)}")

        query = f"UPDATE autofill_queue SET {', '.join(updates)} WHERE job_id = $1"

        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *args)
            updated = "UPDATE 1" in result
            if updated:
                logger.info("Job status updated", job_id=job_id, status=status)
            return updated

    async def get_all_jobs(self) -> list[dict[str, Any]]:
        """Every queue row (oldest first), for batch tracking / reconciliation."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id, apply_link, role, company, apply_mode, status,
                       retries, error, filled_payload, screenshot_path,
                       created_at, updated_at
                FROM autofill_queue
                ORDER BY created_at ASC
                """
            )
        result = []
        for row in rows:
            res = dict(row)
            if isinstance(res.get("filled_payload"), str):
                res["filled_payload"] = json.loads(res["filled_payload"])
            result.append(res)
        return result

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch job details by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM autofill_queue WHERE job_id = $1", job_id)
            if not row:
                return None
            res = dict(row)
            if isinstance(res.get("filled_payload"), str):
                res["filled_payload"] = json.loads(res["filled_payload"])
            if isinstance(res.get("pending_questions"), str):
                res["pending_questions"] = json.loads(res["pending_questions"])
            return res

    async def mark_deferred(
        self, job_id: str, questions: list[str] | None = None, reason: str = ""
    ) -> bool:
        """Mark a job as deferred: it needs user input before it can be completed.

        ``deferred`` is a terminal status for the claim loop, so the job is
        never re-processed automatically; it is picked up again via the
        morning digest and ``resume`` flow.

        Pending questions are ACCUMULATED, never replaced: a form can raise
        several unknown questions, and each deferral must keep every question
        so the digest and resume flow see the full list.
        """
        async with self._pool.acquire() as conn:
            if questions:
                existing = await conn.fetchval(
                    "SELECT pending_questions FROM autofill_queue WHERE job_id = $1",
                    job_id,
                )
                existing_list: list[Any] = []
                if isinstance(existing, str) and existing.strip():
                    existing_list = json.loads(existing)
                elif isinstance(existing, list):
                    existing_list = existing
                merged = list(existing_list)
                seen_q = {str(q.get("question") if isinstance(q, dict) else q) for q in merged}
                for q in questions:
                    qtext = str(q.get("question") if isinstance(q, dict) else q)
                    if qtext and qtext not in seen_q:
                        merged.append(q)
                        seen_q.add(qtext)

            updates = ["status = 'deferred'", "updated_at = NOW()"]
            args: list[Any] = [job_id]
            if questions:
                args.append(json.dumps(merged))
                updates.append(f"pending_questions = ${len(args)}::jsonb")
            if reason:
                args.append(reason)
                updates.append(f"error = ${len(args)}")
            query = f"UPDATE autofill_queue SET {', '.join(updates)} WHERE job_id = $1"
            result = await conn.execute(query, *args)
            updated = "UPDATE 1" in result
            if updated:
                logger.info(
                    "Job deferred",
                    job_id=job_id,
                    questions=questions or [],
                    reason=reason,
                )
            return updated

    async def clear_pending_questions(self, job_id: str) -> bool:
        """Reset a job's pending questions after a successful resume."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE autofill_queue
                SET pending_questions = '[]'::jsonb, updated_at = NOW()
                WHERE job_id = $1
                """,
                job_id,
            )
        return "UPDATE 1" in result

    async def get_deferred_jobs(self) -> list[dict[str, Any]]:
        """All deferred jobs (newest first), for the CLI list and resume flow."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id, apply_link, role, company, apply_mode, status,
                       pending_questions, error, created_at, updated_at
                FROM autofill_queue
                WHERE status = 'deferred'
                ORDER BY updated_at DESC
                """
            )
        result = []
        for row in rows:
            res = dict(row)
            if isinstance(res.get("pending_questions"), str):
                res["pending_questions"] = json.loads(res["pending_questions"])
            result.append(res)
        return result

    async def get_pending_summary_jobs(self) -> list[dict[str, Any]]:
        """Deferred jobs not yet included in a morning digest."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT job_id, apply_link, role, company, pending_questions, updated_at
                FROM autofill_queue
                WHERE status = 'deferred'
                  AND COALESCE(summary_sent, FALSE) = FALSE
                  AND jsonb_array_length(COALESCE(pending_questions, '[]'::jsonb)) > 0
                ORDER BY updated_at DESC
                """
            )
        result = []
        for row in rows:
            res = dict(row)
            if isinstance(res.get("pending_questions"), str):
                res["pending_questions"] = json.loads(res["pending_questions"])
            result.append(res)
        return result

    async def mark_summary_sent(self, job_ids: list[str]) -> int:
        """Mark jobs as included in a morning digest. Returns rows updated."""
        if not job_ids:
            return 0
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE autofill_queue
                SET summary_sent = TRUE, updated_at = NOW()
                WHERE job_id = ANY($1::text[])
                """,
                job_ids,
            )
        updated = int(result.split()[-1]) if result else 0
        if updated:
            logger.info("Jobs included in morning digest", count=updated, job_ids=job_ids)
        return updated

    async def purge_expired_fills(self) -> int:
        """Delete autofill_fills rows older than their 2-day TTL.

        Returns the number of rows removed. Called opportunistically on every
        fill insert and lookup so the audit table never grows unbounded.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM autofill_fills WHERE expires_at IS NOT NULL AND expires_at < NOW()"
            )
        removed = int(result.split()[-1]) if result else 0
        if removed:
            logger.info("Purged expired autofill fills", count=removed)
        return removed

    async def record_fill(
        self,
        job_id: str,
        question: str,
        answer: str | None,
        source: str | None = None,
        options: list[str] | None = None,
    ) -> bool:
        """Persist one screener question + the answer autofill committed for it.

        Best-effort audit trail: raises are left to the caller (the worker
        swallows them so a store failure never aborts a fill).
        """
        if not job_id or not question:
            return False
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO autofill_fills (job_id, question, answer, source, options)
                VALUES ($1, $2, $3, $4, $5)
                """,
                job_id,
                question,
                answer,
                source,
                json.dumps(list(options or [])),
            )
        with contextlib.suppress(Exception):
            await self.purge_expired_fills()
        return True

    async def get_fills(self, job_id: str) -> list[dict[str, Any]]:
        """All recorded question/answer rows for one job (oldest first)."""
        with contextlib.suppress(Exception):
            await self.purge_expired_fills()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, job_id, question, answer, source, options, created_at
                FROM autofill_fills
                WHERE job_id = $1
                  AND (expires_at IS NULL OR expires_at >= NOW())
                ORDER BY id ASC
                """,
                job_id,
            )
        result = []
        for row in rows:
            res = dict(row)
            if isinstance(res.get("options"), str):
                res["options"] = json.loads(res["options"])
            result.append(res)
        return result
