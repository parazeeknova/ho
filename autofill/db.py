"""PostgreSQL queue and state persistence for autofill service."""

from __future__ import annotations

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
    ) -> bool:
        """Update status and payload of a job."""
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
        """
        updates = ["status = 'deferred'", "updated_at = NOW()"]
        args: list[Any] = [job_id]
        if questions is not None:
            args.append(json.dumps(questions))
            updates.append(f"pending_questions = ${len(args)}::jsonb")
        if reason:
            args.append(reason)
            updates.append(f"error = ${len(args)}")
        query = f"UPDATE autofill_queue SET {', '.join(updates)} WHERE job_id = $1"
        async with self._pool.acquire() as conn:
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
                WHERE status = 'deferred' AND COALESCE(summary_sent, FALSE) = FALSE
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
