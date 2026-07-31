"""PostgreSQL queue and state persistence for autofill service."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional
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
"""


class AutofillDB:
    """Async connection-pool-backed queue store for job autofill state."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, config: Optional[PostgresConfig] = None) -> AutofillDB:
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
        role: Optional[str] = None,
        company: Optional[str] = None,
        ats_platform: str = "",
        apply_mode: str = "review",
    ) -> str:
        """Enqueue a new job application. Returns job_id."""
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO autofill_queue (job_id, apply_link, role, company, ats_platform, apply_mode, status)
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

    async def claim_next_job(self, lease_seconds: int = 600) -> Optional[dict[str, Any]]:
        """Atomically claim the next pending job or expired lease using FOR UPDATE SKIP LOCKED."""
        query = """
        UPDATE autofill_queue
        SET status = 'filling',
            lease_expires = NOW() + ($1 || ' seconds')::INTERVAL,
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
        RETURNING job_id, apply_link, role, company, ats_platform, apply_mode, status, retries, filled_payload, screenshot_path;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, str(lease_seconds))
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
        filled_payload: Optional[dict[str, Any]] = None,
        screenshot_path: Optional[str] = None,
        error: Optional[str] = None,
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
            updated = result.endswith("1")
            if updated:
                logger.info("Job status updated", job_id=job_id, status=status)
            return updated

    async def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """Fetch job details by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM autofill_queue WHERE job_id = $1", job_id)
            if not row:
                return None
            res = dict(row)
            if isinstance(res.get("filled_payload"), str):
                res["filled_payload"] = json.loads(res["filled_payload"])
            return res
