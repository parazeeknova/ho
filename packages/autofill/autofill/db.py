"""PostgreSQL queue and state persistence for autofill service."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Sequence
from typing import Any

import asyncpg
from src.configuration import PostgresConfig, get_config
from src.logging import get_logger


def now_utc() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


_FAILURE_TAXONOMY: dict[str, str] = {
    "ban": "ban",
    "blocked": "ban",
    "selector": "selector_drift",
    "captcha": "captcha",
    "challenge": "captcha",
    "bot": "captcha",
    "network": "network",
    "timeout": "network",
    "econn": "network",
    "127": "infra",
}


def _failure_label(error: str) -> str:
    """Map a raw error message onto the failure taxonomy.

    selector_drift | captcha | ban | network | infra — used by the circuit
    breaker so the same-class error can be counted together.
    """
    low = (error or "").lower()
    for token, label in _FAILURE_TAXONOMY.items():
        if token in low:
            return label
    return "unknown"


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

-- Applied-vs-not tracking: applied_at is set the moment a job is submitted,
-- error_count/last_error/last_error_at record fill failures so the loop and
-- reports can separate applied, open and errored jobs. source records where
-- the job entered the queue ('radar' pipeline bridge, 'cli', ...).
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS applied_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS error_count INTEGER DEFAULT 0;
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS last_error TEXT DEFAULT '';
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE autofill_queue ADD COLUMN IF NOT EXISTS source TEXT DEFAULT '';

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

-- Discord question mailbox: single-gateway answer routing. The ingest
-- DiscordAgent is the ONLY gateway consumer; the autofill bridge sends a
-- question, drops its message ids here, and the agent writes the user's reply
-- (or button press) back into the row. The bridge polls this table, never
-- Discord, so two gateway clients never race on the same bot token.
CREATE TABLE IF NOT EXISTS discord_question_mailbox (
    question_id TEXT PRIMARY KEY,
    chat_id     TEXT NOT NULL,
    message_ids BIGINT[] NOT NULL,
    question    TEXT NOT NULL,
    state       TEXT NOT NULL DEFAULT 'pending',
    answer      TEXT,
    asked_at    TIMESTAMPTZ DEFAULT NOW(),
    answered_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_discord_mailbox_msgs
    ON discord_question_mailbox USING GIN (message_ids);

-- Which process currently owns the Discord gateway (the ingest DiscordAgent
-- heartbeats here on every event). The bridge uses the row's freshness to
-- decide whether answers can be routed through the mailbox.
CREATE TABLE IF NOT EXISTS discord_poller_state (
    poller_id TEXT PRIMARY KEY,
    last_seen TIMESTAMPTZ DEFAULT NOW()
);

-- Active Discord sweep thread. The ingest DiscordAgent records the id of the
-- thread it creates for the current sweep; the autofill bridge reads it so
-- deferred/captcha/queue notifications land INSIDE that thread instead of
-- the main channel.
CREATE TABLE IF NOT EXISTS discord_thread_state (
    state_id   TEXT PRIMARY KEY DEFAULT 'active_sweep',
    thread_id  TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Procedural memory: resolved selectors / flow classification / per-field
-- strategies learned from successful fills, keyed by host + form signature.
-- Consulted before the LLM; written on success, decayed on failure. This is
-- what turns the five adapters into one generic adapter plus learned overrides.
CREATE TABLE IF NOT EXISTS autofill_site_knowledge (
    host            TEXT NOT NULL,
    form_signature  TEXT NOT NULL,
    platform        TEXT NOT NULL DEFAULT 'generic',
    selectors       JSONB NOT NULL DEFAULT '{}'::jsonb,
    flow            TEXT NOT NULL DEFAULT '',
    strategies      JSONB NOT NULL DEFAULT '{}'::jsonb,
    success_count   INTEGER NOT NULL DEFAULT 0,
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_good_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (host, form_signature)
);

-- Per-domain health + circuit breaker. After N consecutive identical errors a
-- host is quarantined (cooldown_until) so a broken/blocking ATS doesn't burn
-- job retries; a human handoff is triggered instead.
CREATE TABLE IF NOT EXISTS site_health (
    domain          TEXT PRIMARY KEY,
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_fail       TEXT NOT NULL DEFAULT '',
    last_good       TIMESTAMPTZ,
    cooldown_until  TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
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
        source: str = "",
    ) -> str:
        """Enqueue a new job application. Returns job_id.

        Deduplicates against active rows (pending / filling / awaiting_review /
        deferred) for the same apply_link, returning the existing job_id.

        ``ats_platform`` defaults to ``classify_ats(apply_link)`` so every row
        is tagged with the ATS platform the browser adapter will use — callers
        don't need to remember to pass it.
        """
        if not ats_platform:
            from autofill.ats import classify_ats

            ats_platform = classify_ats(apply_link)
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
                    job_id, apply_link, role, company, ats_platform, apply_mode, status, source
                )
                VALUES ($1, $2, $3, $4, $5, $6, 'pending', $7)
                """,
                job_id,
                apply_link,
                role,
                company,
                ats_platform,
                apply_mode,
                source,
            )
        logger.info("Job enqueued", job_id=job_id, apply_link=apply_link, mode=apply_mode)
        return job_id

    async def claim_next_job(
        self, lease_seconds: int = 600, max_retries: int | None = None
    ) -> dict[str, Any] | None:
        """Atomically claim the next pending job or expired lease using FOR UPDATE SKIP LOCKED.

        ``max_retries`` caps how many times a job may be claimed/retried before
        it is marked ``failed`` (terminal). Infra failures (exit 127) do not
        increment ``retries``, so a broken environment never burns the budget.
        """
        if max_retries is None:
            max_retries = int(os.environ.get("AUTOFILL_MAX_RETRIES", "2"))
        query = """
        WITH candidate AS (
            SELECT job_id FROM autofill_queue
            WHERE status = 'pending'
               OR (status IN ('filling', 'awaiting_review') AND lease_expires < NOW())
            ORDER BY created_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        ),
        exhausted AS (
            UPDATE autofill_queue SET status = 'failed',
                error = COALESCE(last_error, '') || ' [retries exhausted]',
                last_error = COALESCE(last_error, '') || ' [retries exhausted]',
                last_error_at = NOW(),
                updated_at = NOW()
            WHERE job_id IN (SELECT job_id FROM candidate)
              AND retries >= $2
            RETURNING job_id
        )
        UPDATE autofill_queue
        SET status = 'filling',
            lease_expires = NOW() + ($1::int * INTERVAL '1 second'),
            retries = retries + 1,
            updated_at = NOW()
        WHERE job_id = (
            SELECT job_id FROM candidate
            WHERE job_id NOT IN (SELECT job_id FROM exhausted)
        )
        RETURNING job_id, apply_link, role, company, ats_platform, apply_mode,
                  status, retries, filled_payload, screenshot_path;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, lease_seconds, max_retries)
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
        infra_failure: bool = False,
    ) -> bool:
        """Update status and payload of a job.

        Guarded: a terminal status (``deferred``, ``submitted``, ``expired``) is
        never overwritten by a later non-terminal transition, so a deferred job
        that still reaches the review step keeps its deferred status and stays
        in the morning digest, and an expired posting is never resurrected.
        The resume flow (``run_resume``) passes ``override_terminal=True``:
        after the user answers the deferred questions, clearing ``deferred``
        to ``skipped``/``failed`` is exactly what is wanted, not a downgrade.

        ``infra_failure`` marks an environment failure (e.g. runner exit 127):
        the job's ``error_count`` is NOT incremented so a broken environment
        can never burn a real job's retry budget.
        """
        if not override_terminal and status not in ("deferred", "submitted", "expired"):
            current = await self.get_job(job_id)
            if current and current.get("status") in ("deferred", "submitted", "expired"):
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

        if status == "submitted":
            updates.append("applied_at = NOW()")
        elif status == "failed":
            if not infra_failure:
                updates.append("error_count = error_count + 1")
            updates.append("last_error_at = NOW()")
            if error is not None:
                args.append(error)
                updates.append(f"last_error = ${len(args)}")
        elif status == "expired":
            # Terminal: record why so the queue shows the posting is gone.
            updates.append("last_error_at = NOW()")
            if error is not None:
                args.append(error)
                updates.append(f"last_error = ${len(args)}")

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

    async def link_known(self, apply_link: str) -> bool:
        """True when any row (any status) exists for a link.

        Used by the radar bridge so an already-applied or already-failed job
        is never re-enqueued from the stored corpus.
        """
        async with self._pool.acquire() as conn:
            return bool(
                await conn.fetchval(
                    "SELECT 1 FROM autofill_queue WHERE apply_link = $1 LIMIT 1",
                    apply_link,
                )
            )

    async def queue_summary(self) -> dict[str, Any]:
        """Applied / open / errored counts for the loop report."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    COUNT(*) FILTER (WHERE applied_at IS NOT NULL) AS applied,
                    COUNT(*) FILTER (WHERE applied_at IS NULL
                                     AND status NOT IN ('deferred', 'skipped', 'failed', 'expired'))
                        AS open,
                    COUNT(*) FILTER (WHERE status = 'deferred') AS deferred,
                    COUNT(*) FILTER (WHERE status = 'skipped') AS skipped,
                    COUNT(*) FILTER (WHERE status = 'expired') AS expired,
                    COUNT(*) FILTER (WHERE status = 'failed' OR error_count > 0) AS errored,
                    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                    COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                    COUNT(*) FILTER (WHERE status = 'filling') AS filling,
                    COUNT(*) FILTER (WHERE status = 'awaiting_review') AS awaiting_review
                FROM autofill_queue
                """
            )
            return dict(rows[0])

    async def open_mailbox_question(
        self, question_id: str, chat_id: str, message_ids: list[int], question: str
    ) -> None:
        """Register a pending question whose reply the agent must capture."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO discord_question_mailbox (question_id, chat_id, message_ids, question)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (question_id) DO NOTHING
                """,
                question_id,
                chat_id,
                message_ids,
                question,
            )

    async def append_mailbox_message_ids(self, question_id: str, message_ids: list[int]) -> None:
        """Track extra sent messages (hint / continuation chunks) for a question."""
        if not message_ids:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE discord_question_mailbox "
                "SET message_ids = message_ids || $2 WHERE question_id = $1",
                question_id,
                message_ids,
            )

    async def poll_mailbox_question(self, question_id: str) -> tuple[str, str | None] | None:
        """Return ``(state, answer)`` for a question, or None when unknown."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state, answer FROM discord_question_mailbox WHERE question_id = $1",
                question_id,
            )
            if not row:
                return None
            return row["state"], row["answer"]

    async def close_mailbox_question(self, question_id: str, state: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE discord_question_mailbox SET state = $2 WHERE question_id = $1",
                question_id,
                state,
            )

    async def mailbox_lookup_by_message(self, message_id: int) -> dict[str, Any] | None:
        """Return a mailbox row for a sent message (any state), or None."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT question_id, question, state, answer
                FROM discord_question_mailbox
                WHERE $1::bigint = ANY(message_ids)
                ORDER BY asked_at DESC
                LIMIT 1
                """,
                message_id,
            )
            return dict(row) if row else None

    async def answer_mailbox_message(self, message_id: int, answer: str) -> bool:
        """Record the user's reply/callback against a pending question.

        Called by the single Telegram poller (the ingest TelegramAgent) for
        every incoming message or callback that replies to a sent question.
        Returns True when the message matched a pending question.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT question_id FROM discord_question_mailbox
                WHERE state = 'pending' AND $1::bigint = ANY(message_ids)
                ORDER BY asked_at DESC
                LIMIT 1
                """,
                message_id,
            )
            if not row:
                return False
            await conn.execute(
                "UPDATE discord_question_mailbox "
                "SET state = 'answered', answer = $2, answered_at = NOW() "
                "WHERE question_id = $1",
                row["question_id"],
                answer,
            )
            return True

    async def heartbeat_poller(self, poller_id: str = "ingest") -> None:
        """Stamp this process as the live getUpdates consumer."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO discord_poller_state (poller_id, last_seen)
                VALUES ($1, NOW())
                ON CONFLICT (poller_id) DO UPDATE SET last_seen = NOW()
                """,
                poller_id,
            )

    async def poller_alive(self, max_age_seconds: float = 25.0, poller_id: str = "ingest") -> bool:
        """True when the ingest TelegramAgent is actively polling."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT last_seen FROM discord_poller_state WHERE poller_id = $1",
                poller_id,
            )
            if not row or row["last_seen"] is None:
                return False
            return (now_utc() - row["last_seen"]).total_seconds() <= max_age_seconds

    async def set_active_thread(self, thread_id: str, max_age_seconds: float = 3600.0) -> None:
        """Record the active Discord sweep thread id (called by the gateway)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO discord_thread_state (state_id, thread_id, updated_at)
                VALUES ('active_sweep', $1, NOW())
                ON CONFLICT (state_id) DO UPDATE SET
                    thread_id = EXCLUDED.thread_id,
                    updated_at = NOW()
                """,
                str(thread_id),
            )

    async def active_thread(self, max_age_seconds: float = 3600.0) -> str | None:
        """The current active Discord thread id, if fresh (the autofill bridge
        reads this so notifications land inside the sweep thread)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT thread_id, updated_at FROM discord_thread_state
                WHERE state_id = 'active_sweep'
                """
            )
            if not row or not row["thread_id"]:
                return None
            if (now_utc() - row["updated_at"]).total_seconds() > max_age_seconds:
                return None
            return str(row["thread_id"])

    async def unapplied_stats(self, stale_hours: int = 48) -> dict[str, int]:
        """Autofill-side job stats for the startup message.

        ``unapplied`` = queue jobs not yet applied (excludes deferred/skipped/
        failed terminal rows). ``stale`` = unapplied jobs created more than
        ``stale_hours`` ago (sitting too long). ``total`` = every queue row.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (
                        WHERE applied_at IS NULL
                          AND status NOT IN ('deferred', 'skipped', 'failed')
                    ) AS unapplied,
                    COUNT(*) FILTER (
                        WHERE applied_at IS NULL
                          AND status NOT IN ('deferred', 'skipped', 'failed')
                          AND created_at < NOW() - ($1::int * INTERVAL '1 hour')
                    ) AS stale,
                    COUNT(*) AS total
                FROM autofill_queue
                """,
                stale_hours,
            )
            return dict(row)

    async def last_applied(self) -> dict[str, Any] | None:
        """Most recently applied job (role/company/applied_at), if any."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT role, company, applied_at FROM autofill_queue
                WHERE applied_at IS NOT NULL
                ORDER BY applied_at DESC
                LIMIT 1
                """
            )
            return dict(row) if row else None

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
        self, job_id: str, questions: Sequence[str | dict[str, Any]] | None = None, reason: str = ""
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

    # ── site knowledge (procedural memory) ──────────────────────────────

    async def get_site_knowledge(self, host: str, form_signature: str) -> dict[str, Any] | None:
        """Look up learned selectors/flow for a host+form before the LLM probes."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT host, form_signature, platform, selectors, flow, strategies,
                       success_count, fail_count, last_good_at
                FROM autofill_site_knowledge
                WHERE host = $1 AND form_signature = $2
                """,
                host,
                form_signature,
            )
        if not row:
            return None
        res = dict(row)
        for key in ("selectors", "strategies"):
            if isinstance(res.get(key), str):
                res[key] = json.loads(res[key])
        return res

    async def upsert_site_knowledge(
        self,
        host: str,
        form_signature: str,
        platform: str = "generic",
        selectors: dict[str, Any] | None = None,
        flow: str = "",
        strategies: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """Record a successful/failed selector map for a host+form.

        On success: increment success_count, set last_good_at, store the
        resolved selectors/flow/strategies. On failure: increment fail_count
        (drift detection) without overwriting the last-known-good map.
        """
        async with self._pool.acquire() as conn:
            if success:
                await conn.execute(
                    """
                    INSERT INTO autofill_site_knowledge (
                        host, form_signature, platform, selectors, flow, strategies,
                        success_count, fail_count, last_good_at
                    ) VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, 1, 0, NOW())
                    ON CONFLICT (host, form_signature) DO UPDATE SET
                        platform = EXCLUDED.platform,
                        selectors = EXCLUDED.selectors,
                        flow = EXCLUDED.flow,
                        strategies = EXCLUDED.strategies,
                        success_count = autofill_site_knowledge.success_count + 1,
                        last_good_at = NOW(),
                        updated_at = NOW()
                    """,
                    host,
                    form_signature,
                    platform,
                    json.dumps(selectors or {}),
                    flow,
                    json.dumps(strategies or {}),
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO autofill_site_knowledge (
                        host, form_signature, platform, success_count, fail_count
                    ) VALUES ($1, $2, $3, 0, 1)
                    ON CONFLICT (host, form_signature) DO UPDATE SET
                        fail_count = autofill_site_knowledge.fail_count + 1,
                        updated_at = NOW()
                    """,
                    host,
                    form_signature,
                    platform,
                )

    # ── site health + circuit breaker ───────────────────────────────────

    async def site_health(self, domain: str) -> dict[str, Any] | None:
        """Current health record for a domain."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT domain, fail_count, last_fail, last_good, cooldown_until
                FROM site_health WHERE domain = $1
                """,
                domain,
            )
        return dict(row) if row else None

    async def record_site_failure(
        self, domain: str, error: str, cooldown_seconds: int = 3600
    ) -> None:
        """Increment a domain's consecutive-failure count; quarantine on threshold.

        ``error`` is a failure-taxonomy label (selector_drift / captcha / ban /
        network) or the raw message. When the consecutive failures reach the
        quarantine threshold, ``cooldown_until`` is set so the domain is skipped
        for a while instead of burning job retries.
        """
        threshold = int(os.environ.get("SITE_HEALTH_QUARANTINE", "3"))
        label = _failure_label(error)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO site_health (domain, fail_count, last_fail, cooldown_until, updated_at)
                VALUES ($1, 1, $2, NULL, NOW())
                ON CONFLICT (domain) DO UPDATE SET
                    fail_count = site_health.fail_count + 1,
                    last_fail = $2,
                    updated_at = NOW(),
                    cooldown_until = CASE
                        WHEN site_health.fail_count + 1 >= $3
                        THEN NOW() + ($4::int * INTERVAL '1 second')
                        ELSE site_health.cooldown_until
                    END
                RETURNING fail_count, cooldown_until
                """,
                domain,
                label,
                threshold,
                cooldown_seconds,
            )
            if row and row["cooldown_until"]:
                logger.warning(
                    "Domain quarantined by circuit breaker",
                    domain=domain,
                    failures=row["fail_count"],
                    until=str(row["cooldown_until"]),
                    label=label,
                )

    async def record_site_success(self, domain: str) -> None:
        """Reset a domain's failure count on a successful fill."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO site_health (domain, fail_count, last_good, updated_at)
                VALUES ($1, 0, NOW(), NOW())
                ON CONFLICT (domain) DO UPDATE SET
                    fail_count = 0,
                    last_good = NOW(),
                    cooldown_until = NULL,
                    updated_at = NOW()
                """,
                domain,
            )

    async def domain_quarantined(self, domain: str) -> bool:
        """True when a domain is in its circuit-breaker cooldown."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM site_health
                WHERE domain = $1 AND cooldown_until IS NOT NULL AND cooldown_until > NOW()
                """,
                domain,
            )
            return row is not None
