"""Standalone pgvector memory engine for agent deduplication and semantic RAG.

Connects exclusively to the ``agent-memory-db`` service.
No dependency on Firebase or any other persistence layer.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from pgvector import Vector
from pgvector.asyncpg import register_vector

from src.configuration import PostgresConfig, get_config
from src.logging import get_logger

logger = get_logger("memory_store")

VECTOR_DIM = 1024

CREATE_TABLES_SQL = f"""
CREATE TABLE IF NOT EXISTS processed_jobs (
    url            TEXT PRIMARY KEY,
    role           TEXT,
    company        TEXT,
    match_percent  INT,
    verdict        TEXT,
    raw_json       JSONB,
    created_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS resume_embeddings (
    id         SERIAL PRIMARY KEY,
    section    VARCHAR(128),
    content    TEXT NOT NULL,
    embedding  vector({VECTOR_DIM})
);

CREATE TABLE IF NOT EXISTS discovered_domains (
    domain          TEXT PRIMARY KEY,
    source_url      TEXT,
    crawled         BOOLEAN DEFAULT FALSE,
    discovered_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS telegram_notified_jobs (
    dedup_key       TEXT PRIMARY KEY,
    role            TEXT,
    company         TEXT,
    notified_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jobs_ledger (
    dedup_key     TEXT PRIMARY KEY,
    role          TEXT,
    company       TEXT,
    match_percent INT DEFAULT 0,
    shortlist_probability INT DEFAULT 0,
    salary        TEXT,
    posted_date   TEXT,
    location      TEXT DEFAULT 'Remote',
    apply_link    TEXT,
    jd_summary    TEXT DEFAULT '',
    company_description TEXT DEFAULT '',
    role_summary  TEXT DEFAULT '',
    verdict       TEXT DEFAULT 'NO_MATCH',
    is_startup    BOOLEAN DEFAULT FALSE,
    founders      JSONB DEFAULT '[]'::jsonb,
    funding_stage TEXT DEFAULT '',
    funding_info  JSONB DEFAULT '{{}}'::jsonb,
    founder_socials JSONB DEFAULT '[]'::jsonb,
    company_news  TEXT DEFAULT '',
    osint_signals JSONB DEFAULT '[]'::jsonb,
    source_url    TEXT DEFAULT '',
    raw_json      JSONB DEFAULT '{{}}'::jsonb,
    created_at    TIMESTAMP DEFAULT NOW(),
    updated_at    TIMESTAMP DEFAULT NOW()
);

ALTER TABLE IF EXISTS frontier_state
ADD COLUMN IF NOT EXISTS state TEXT DEFAULT 'pending';
ALTER TABLE IF EXISTS frontier_state
ADD COLUMN IF NOT EXISTS lease_expires DOUBLE PRECISION DEFAULT 0;
CREATE TABLE IF NOT EXISTS frontier_state (
    work_id        TEXT PRIMARY KEY, agent TEXT NOT NULL,
    node_id        TEXT NOT NULL, node_type TEXT DEFAULT 'company',
    priority       INT DEFAULT 50, depth INT DEFAULT 0,
    state          TEXT DEFAULT 'pending', retries INT DEFAULT 0,
    lease_expires  DOUBLE PRECISION DEFAULT 0,
    payload        JSONB DEFAULT '{{}}'::jsonb,
    created_at     TIMESTAMP DEFAULT NOW(),
    updated_at     DOUBLE PRECISION DEFAULT 0
);
CREATE TABLE IF NOT EXISTS frontier_completed (
    work_id        TEXT PRIMARY KEY, completed_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS source_checkpoints (
    source_id            TEXT PRIMARY KEY,
    source_type          TEXT NOT NULL DEFAULT 'unknown',
    board_url            TEXT DEFAULT '',
    last_polled          DOUBLE PRECISION DEFAULT 0,
    last_snapshot_hash   TEXT DEFAULT '',
    last_snapshot_count  INT DEFAULT 0,
    consecutive_failures INT DEFAULT 0,
    consecutive_empty    INT DEFAULT 0,
    quality_score        REAL DEFAULT 0.5,
    active               BOOLEAN DEFAULT TRUE,
    backoff_until        DOUBLE PRECISION DEFAULT 0,
    total_jobs_produced  INT DEFAULT 0,
    total_direct_url_rate REAL DEFAULT 0.0,
    company_name         TEXT DEFAULT '',
    discovery_origin     TEXT DEFAULT '',
    updated_at           TIMESTAMP DEFAULT NOW()
);

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS board_url TEXT DEFAULT '';

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS company_name TEXT DEFAULT '';

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS discovery_origin TEXT DEFAULT '';

CREATE TABLE IF NOT EXISTS job_observations (
    url_hash              TEXT PRIMARY KEY,
    url                   TEXT NOT NULL,
    source                TEXT NOT NULL DEFAULT '',
    title                 TEXT DEFAULT '',
    snippet               TEXT DEFAULT '',
    first_seen            DOUBLE PRECISION DEFAULT 0,
    last_seen             DOUBLE PRECISION DEFAULT 0,
    freshness_lane        TEXT DEFAULT 'review',
    direct_posting_verified BOOLEAN DEFAULT FALSE,
    raw_json              JSONB DEFAULT '{{}}'::jsonb,
    created_at            TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS radar_candidates (
    canonical_id          TEXT PRIMARY KEY,
    source                TEXT NOT NULL DEFAULT '',
    direct_apply_url      TEXT DEFAULT '',
    normalized_company    TEXT DEFAULT '',
    normalized_role       TEXT DEFAULT '',
    normalized_location   TEXT DEFAULT 'Remote',
    freshness_lane        TEXT DEFAULT 'review',
    source_confidence     REAL DEFAULT 0.5,
    eligibility           TEXT DEFAULT 'pending',
    rejection_reason      TEXT DEFAULT '',
    rejection_detail      TEXT DEFAULT '',
    role_family           TEXT DEFAULT 'unknown',
    salary_amount         REAL,
    salary_currency       TEXT DEFAULT '',
    salary_period         TEXT DEFAULT '',
    salary_raw            TEXT DEFAULT '',
    posted_date           TEXT DEFAULT '',
    first_seen            DOUBLE PRECISION DEFAULT 0,
    last_seen             DOUBLE PRECISION DEFAULT 0,
    matching_skills       JSONB DEFAULT '[]'::jsonb,
    missing_skills        JSONB DEFAULT '[]'::jsonb,
    match_percent         INT DEFAULT 0,
    shortlist_probability INT DEFAULT 0,
    verdict               TEXT DEFAULT 'NO_MATCH',
    jd_summary            TEXT DEFAULT '',
    company_description   TEXT DEFAULT '',
    role_summary          TEXT DEFAULT '',
    is_remote             BOOLEAN DEFAULT FALSE,
    founders              JSONB DEFAULT '[]'::jsonb,
    funding_stage         TEXT DEFAULT '',
    funding_info          JSONB DEFAULT '{{}}'::jsonb,
    founder_socials       JSONB DEFAULT '[]'::jsonb,
    company_news          TEXT DEFAULT '',
    osint_signals         JSONB DEFAULT '[]'::jsonb,
    extra                 JSONB DEFAULT '{{}}'::jsonb,
    created_at            TIMESTAMP DEFAULT NOW(),
    updated_at            TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS radar_analytics (
    id            SERIAL PRIMARY KEY,
    event_type    TEXT NOT NULL DEFAULT '',
    event_data    JSONB DEFAULT '{{}}'::jsonb,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS source_snapshots (
    source_id     TEXT PRIMARY KEY,
    snapshot_data TEXT DEFAULT '',
    updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_radar_candidates_eligibility ON radar_candidates(eligibility);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_freshness ON radar_candidates(freshness_lane);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_rejection ON radar_candidates(rejection_reason);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_role_family ON radar_candidates(role_family);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_created ON radar_candidates(created_at);
CREATE INDEX IF NOT EXISTS idx_job_observations_source ON job_observations(source);
CREATE INDEX IF NOT EXISTS idx_job_observations_first_seen ON job_observations(first_seen);
CREATE INDEX IF NOT EXISTS idx_source_checkpoints_active ON source_checkpoints(active);
"""


class MemoryStore:
    """Async connection-pool-backed pgvector store."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, config: PostgresConfig | None = None) -> MemoryStore:
        """Initialise pool, register vector type, create tables."""
        cfg = config or get_config().postgres
        pool = await asyncpg.create_pool(cfg.dsn, min_size=cfg.min_pool, max_size=cfg.max_pool)
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(CREATE_TABLES_SQL)
        logger.info("MemoryStore initialized", dsn=cfg.dsn.split("@")[-1])
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()
        logger.info("MemoryStore closed")

    # Processed_jobs

    async def is_url_processed(self, url: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT 1 FROM processed_jobs WHERE url = $1", url)
            return row is not None

    async def save_job_result(self, data: dict[str, Any]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO processed_jobs (url, role, company, match_percent,
                                            verdict, raw_json)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (url) DO UPDATE SET
                    role          = EXCLUDED.role,
                    company       = EXCLUDED.company,
                    match_percent = EXCLUDED.match_percent,
                    verdict       = EXCLUDED.verdict,
                    raw_json      = EXCLUDED.raw_json
                """,
                data.get("url", ""),
                data.get("role", ""),
                data.get("company", ""),
                data.get("match_percent", 0),
                data.get("verdict", "NO_MATCH"),
                json.dumps(data),
            )

    # Resume_embeddings

    async def index_resume_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Insert resume chunks with their pre-computed embeddings.

        Each chunk dict must have keys: ``section``, ``content``, ``embedding``
        (list[float] of length 1024).
        """
        async with self._pool.acquire() as conn, conn.transaction():
            for ch in chunks:
                emb = Vector(ch["embedding"])
                await conn.execute(
                    "INSERT INTO resume_embeddings (section, content, embedding) "
                    "VALUES ($1, $2, $3)",
                    ch["section"],
                    ch["content"],
                    emb,
                )

    async def search_similar_chunks(
        self, query_emb: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most similar resume chunks using cosine distance."""
        vec = Vector(query_emb)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT section, content, embedding <=> $1 AS distance "
                "FROM resume_embeddings "
                "ORDER BY distance ASC "
                "LIMIT $2",
                vec,
                top_k,
            )
        return [
            {
                "section": r["section"],
                "content": r["content"],
                "distance": float(r["distance"]),
            }
            for r in rows
        ]

    async def chunk_count(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM resume_embeddings")
            return row["cnt"] if row else 0

    async def clear_embeddings(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE resume_embeddings")

    # Discovered_domains

    async def add_discovered_domain(self, domain: str, source_url: str = "") -> bool:
        """Insert a new domain. Returns True if newly added, False if already exists."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                INSERT INTO discovered_domains (domain, source_url)
                VALUES ($1, $2)
                ON CONFLICT (domain) DO NOTHING
                """,
                domain,
                source_url,
            )
            return result != "INSERT 0 0"

    async def get_uncrawled_domains(self, limit: int = 50) -> list[str]:
        """Return domains that have not been crawled yet."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT domain FROM discovered_domains "
                "WHERE crawled = FALSE "
                "ORDER BY discovered_at DESC "
                "LIMIT $1",
                limit,
            )
        return [r["domain"] for r in rows]

    async def mark_domains_crawled(self, domains: list[str]) -> None:
        """Mark the given domains as crawled."""
        if not domains:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE discovered_domains SET crawled = TRUE WHERE domain = ANY($1)",
                domains,
            )

    # Telegram_notified_jobs

    async def is_telegram_notified(self, dedup_key: str) -> bool:
        """Return True if job key was already notified via Telegram."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM telegram_notified_jobs WHERE dedup_key = $1",
                dedup_key,
            )
            return row is not None

    async def mark_telegram_notified(self, dedup_key: str, role: str, company: str) -> None:
        """Mark job key as notified in PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO telegram_notified_jobs (dedup_key, role, company)
                VALUES ($1, $2, $3)
                ON CONFLICT (dedup_key) DO NOTHING
                """,
                dedup_key,
                role,
                company,
            )

    # Jobs_ledger

    _JOB_COLUMNS = (
        "dedup_key",
        "role",
        "company",
        "match_percent",
        "shortlist_probability",
        "salary",
        "posted_date",
        "location",
        "apply_link",
        "jd_summary",
        "company_description",
        "role_summary",
        "verdict",
        "is_startup",
        "founders",
        "funding_stage",
        "funding_info",
        "founder_socials",
        "company_news",
        "osint_signals",
        "source_url",
        "raw_json",
    )

    def _row_to_job(self, row: asyncpg.Record) -> dict[str, Any]:
        job: dict[str, Any] = {}
        jsonb_cols = ("founders", "funding_info", "founder_socials", "osint_signals")
        for col in self._JOB_COLUMNS:
            val = row.get(col)
            if col in jsonb_cols and isinstance(val, str):
                val = json.loads(val) if val else ({} if col == "funding_info" else [])
            job[col] = val
        if row.get("raw_json"):
            raw = row["raw_json"]
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    if k not in job or not job[k]:
                        job[k] = v
        return job

    async def upsert_job_ledger(self, dedup_key: str, data: dict[str, Any]) -> int:
        """Insert or merge a job into the ledger atomically.

        The ON CONFLICT DO UPDATE with GREATEST and COALESCE handles
        merging entirely in the database — no Python-side read-modify-write.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs_ledger (dedup_key, role, company, match_percent,
                    shortlist_probability, salary, posted_date, location,
                    apply_link, jd_summary, company_description, role_summary,
                    verdict, is_startup, founders, funding_stage,
                    funding_info, founder_socials, company_news, osint_signals,
                    source_url, raw_json)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                        $15::jsonb,$16,$17::jsonb,$18::jsonb,$19,$20::jsonb,$21,$22::jsonb)
                ON CONFLICT (dedup_key) DO UPDATE SET
                    match_percent = GREATEST(jobs_ledger.match_percent, EXCLUDED.match_percent),
                    shortlist_probability = GREATEST(
                        jobs_ledger.shortlist_probability,
                        EXCLUDED.shortlist_probability
                    ),
                    company_description = COALESCE(
                        NULLIF(jobs_ledger.company_description, ''),
                        EXCLUDED.company_description
                    ),
                    role_summary = COALESCE(
                        NULLIF(jobs_ledger.role_summary, ''),
                        EXCLUDED.role_summary
                    ),
                    salary = COALESCE(jobs_ledger.salary, EXCLUDED.salary),
                    posted_date = COALESCE(jobs_ledger.posted_date, EXCLUDED.posted_date),
                    apply_link = COALESCE(NULLIF(EXCLUDED.apply_link, ''), jobs_ledger.apply_link),
                    jd_summary = COALESCE(NULLIF(jobs_ledger.jd_summary, ''), EXCLUDED.jd_summary),
                    location = COALESCE(EXCLUDED.location, jobs_ledger.location),
                    verdict = EXCLUDED.verdict,
                    is_startup = EXCLUDED.is_startup,
                    founders = EXCLUDED.founders,
                    funding_stage = EXCLUDED.funding_stage,
                    funding_info = EXCLUDED.funding_info,
                    founder_socials = EXCLUDED.founder_socials,
                    company_news = EXCLUDED.company_news,
                    osint_signals = EXCLUDED.osint_signals,
                    source_url = EXCLUDED.source_url,
                    raw_json = EXCLUDED.raw_json,
                    updated_at = NOW()
                """,
                dedup_key,
                data.get("role", ""),
                data.get("company", ""),
                data.get("match_percent", 0),
                data.get("shortlist_probability", 0),
                data.get("salary"),
                data.get("posted_date"),
                data.get("location", "Remote"),
                data.get("apply_link", ""),
                data.get("jd_summary", ""),
                data.get("company_description", ""),
                data.get("role_summary", ""),
                str(data.get("verdict", "NO_MATCH")),
                bool(data.get("is_startup", False)),
                json.dumps(data.get("founders", [])),
                data.get("funding_stage", ""),
                json.dumps(data.get("funding_info", {})),
                json.dumps(data.get("founder_socials", [])),
                data.get("company_news", ""),
                json.dumps(data.get("osint_signals", [])),
                data.get("source_url", data.get("url", "")),
                json.dumps(data),
            )
        return 1

    async def get_job_by_key(self, dedup_key: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs_ledger WHERE dedup_key = $1", dedup_key)
            return self._row_to_job(row) if row else None

    async def get_all_jobs_ledger(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM jobs_ledger ORDER BY match_percent DESC")
        return [self._row_to_job(r) for r in rows]

    async def get_job_ledger_count(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM jobs_ledger")
            return row["cnt"] if row else 0

    async def purge_fake_job_keys(self, keys: list[str]) -> int:
        removed = 0
        async with self._pool.acquire() as conn:
            for key in keys:
                result = await conn.execute("DELETE FROM jobs_ledger WHERE dedup_key = $1", key)
                if result != "DELETE 0":
                    removed += 1
        return removed

    async def search_similar_jobs(
        self, query_emb: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Semantic search over jobs_ledger using pgvector cosine distance.
        Returns top_k similar jobs with their match data."""
        vec = Vector(query_emb)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, embedding <=> $1 AS distance
                FROM jobs_ledger
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT $2
                """,
                vec,
                top_k,
            )
        return [self._row_to_job(r) for r in rows]

    async def job_ledger_count(self) -> int:
        return await self.get_job_ledger_count()

    async def get_top_skills(self, days: int = 30, limit: int = 15) -> list[dict[str, Any]]:
        """Top matching skills across jobs seen in the last *days*."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT skill, COUNT(*) AS job_count
                FROM (
                    SELECT jsonb_array_elements_text(
                        COALESCE(raw_json->'matching_skills', '[]'::jsonb)
                    ) AS skill
                    FROM jobs_ledger
                    WHERE raw_json IS NOT NULL
                      AND created_at >= NOW() - ($1 || ' days')::interval
                ) sub
                WHERE skill IS NOT NULL AND skill != ''
                GROUP BY skill
                ORDER BY job_count DESC
                LIMIT $2
                """,
                str(days),
                limit,
            )
        return [{"skill": r["skill"], "job_count": r["job_count"]} for r in rows]

    async def get_skill_arbitrage(
        self, min_match: int = 50, max_match: int = 69
    ) -> list[dict[str, Any]]:
        """Missing skills that caused near-misses (match in [min_match, max_match])."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT skill, COUNT(*) AS miss_count,
                       AVG(NULLIF(
                           (raw_json->>'salary')::numeric, 0)
                       )::numeric(12,0) AS avg_salary
                FROM (
                    SELECT jsonb_array_elements_text(
                        COALESCE(raw_json->'missing_skills', '[]'::jsonb)
                    ) AS skill,
                        raw_json
                    FROM jobs_ledger
                    WHERE raw_json IS NOT NULL
                      AND match_percent BETWEEN $1 AND $2
                ) sub
                WHERE skill IS NOT NULL AND skill != ''
                GROUP BY skill
                ORDER BY miss_count DESC
                LIMIT 15
                """,
                min_match,
                max_match,
            )
        return [
            {
                "skill": r["skill"],
                "miss_count": r["miss_count"],
                "avg_salary": int(r["avg_salary"]) if r["avg_salary"] is not None else 0,
            }
            for r in rows
        ]

    async def get_company_aggregate_data(self, company_name: str) -> dict[str, Any]:
        """Aggregate hiring patterns for a specific company."""
        async with self._pool.acquire() as conn:
            summary = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) AS total_postings,
                    ROUND(AVG(match_percent))::int AS avg_match,
                    MAX(match_percent)::int AS best_match
                FROM jobs_ledger
                WHERE LOWER(company) = LOWER($1)
                """,
                company_name,
            )

            top_skills = await conn.fetch(
                """
                SELECT skill, COUNT(*) AS cnt
                FROM (
                    SELECT jsonb_array_elements_text(
                        COALESCE(raw_json->'matching_skills', '[]'::jsonb)
                    ) AS skill
                    FROM jobs_ledger
                    WHERE LOWER(company) = LOWER($1)
                      AND raw_json IS NOT NULL
                ) sub
                WHERE skill IS NOT NULL AND skill != ''
                GROUP BY skill
                ORDER BY cnt DESC
                LIMIT 5
                """,
                company_name,
            )

            roles = await conn.fetch(
                """
                SELECT role, match_percent, location, posted_date
                FROM jobs_ledger
                WHERE LOWER(company) = LOWER($1)
                ORDER BY match_percent DESC
                LIMIT 20
                """,
                company_name,
            )

        return {
            "company": company_name,
            "total_postings": summary["total_postings"] if summary else 0,
            "avg_match": summary["avg_match"] if summary else 0,
            "best_match": summary["best_match"] if summary else 0,
            "top_matching_skills": [{"skill": r["skill"], "count": r["cnt"]} for r in top_skills],
            "roles": [
                {
                    "role": r["role"],
                    "match_percent": r["match_percent"],
                    "location": r["location"] or "Remote",
                    "posted_date": str(r["posted_date"]) if r["posted_date"] else "",
                }
                for r in roles
            ],
        }

    async def get_tech_stack_momentum(self, limit: int = 10) -> list[dict[str, Any]]:
        """Month-over-month growth rate of skills in job descriptions.

        Returns skills with the highest 30-day velocity trending upward.
        Only includes skills with at least 5 current occurrences.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH current_month AS (
                    SELECT skill, COUNT(*) AS current_count
                    FROM jobs_ledger,
                         LATERAL jsonb_array_elements_text(
                             COALESCE(raw_json->'matching_skills', '[]'::jsonb)
                         ) AS skill
                    WHERE raw_json IS NOT NULL
                      AND created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY skill
                ),
                previous_month AS (
                    SELECT skill, COUNT(*) AS prev_count
                    FROM jobs_ledger,
                         LATERAL jsonb_array_elements_text(
                             COALESCE(raw_json->'matching_skills', '[]'::jsonb)
                         ) AS skill
                    WHERE raw_json IS NOT NULL
                      AND created_at >= NOW() - INTERVAL '60 days'
                      AND created_at < NOW() - INTERVAL '30 days'
                    GROUP BY skill
                )
                SELECT
                    c.skill,
                    c.current_count,
                    COALESCE(p.prev_count, 0) AS prev_count,
                    ROUND(
                        ((c.current_count - COALESCE(p.prev_count, 0))::numeric
                         / GREATEST(COALESCE(p.prev_count, 1), 1)) * 100, 1
                    ) AS pct_growth
                FROM current_month c
                LEFT JOIN previous_month p ON c.skill = p.skill
                WHERE c.current_count >= 5
                ORDER BY pct_growth DESC
                LIMIT $1
                """,
                limit,
            )
        return [
            {
                "skill": r["skill"],
                "current_count": r["current_count"],
                "prev_count": r["prev_count"],
                "pct_growth": float(r["pct_growth"]),
            }
            for r in rows
        ]

    async def get_ats_blackhole_index(self) -> list[dict[str, Any]]:
        """Rank ATS domains by ghost-job risk and aggregate candidate match rates.

        Extracts the domain root (e.g. 'greenhouse', 'workday') from
        apply_link and computes avg match_percent per domain so candidates
        can avoid low-response enterprise portals.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    CASE
                        WHEN apply_link ILIKE '%greenhouse%' THEN 'greenhouse'
                        WHEN apply_link ILIKE '%lever.co%' THEN 'lever'
                        WHEN apply_link ILIKE '%ashbyhq%' THEN 'ashby'
                        WHEN apply_link ILIKE '%workable%' THEN 'workable'
                        WHEN apply_link ILIKE '%myworkday%' THEN 'workday'
                        WHEN apply_link ILIKE '%smartrecruiters%' THEN 'smartrecruiters'
                        WHEN apply_link ILIKE '%rippling%' THEN 'rippling'
                        ELSE 'other'
                    END AS ats_domain,
                    COUNT(*) AS job_count,
                    ROUND(AVG(match_percent))::int AS avg_match,
                    ROUND(AVG(
                        EXTRACT(DAY FROM (NOW() - created_at))
                    ))::int AS avg_days_open
                FROM jobs_ledger
                WHERE apply_link IS NOT NULL AND apply_link != ''
                GROUP BY ats_domain
                ORDER BY avg_match DESC
                """
            )
        return [
            {
                "ats_domain": r["ats_domain"],
                "job_count": r["job_count"],
                "avg_match": r["avg_match"],
                "avg_days_open": r["avg_days_open"],
            }
            for r in rows
        ]

    async def get_marginal_skill_valuation(self) -> list[dict[str, Any]]:
        """Estimate the salary premium per skill by comparing jobs with and
        without each skill in the matching_skills array.

        Only considers skills appearing in 5+ jobs where salary data exists.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    skill,
                    COUNT(*) AS job_count,
                    ROUND(AVG(NULLIF(
                        (raw_json->>'salary')::numeric, 0)
                    ))::int AS avg_salary,
                    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
                        ORDER BY NULLIF((raw_json->>'salary')::numeric, 0)
                    ))::int AS median_salary
                FROM jobs_ledger,
                     LATERAL jsonb_array_elements_text(
                         COALESCE(raw_json->'matching_skills', '[]'::jsonb)
                     ) AS skill
                WHERE raw_json IS NOT NULL
                  AND raw_json->>'salary' IS NOT NULL
                  AND NULLIF((raw_json->>'salary')::numeric, 0) > 0
                  AND skill IS NOT NULL AND skill != ''
                GROUP BY skill
                HAVING COUNT(*) >= 3
                ORDER BY avg_salary DESC
                LIMIT 20
                """
            )
        results: list[dict[str, Any]] = []
        for r in rows:
            results.append(
                {
                    "skill": r["skill"],
                    "job_count": r["job_count"],
                    "avg_salary": r["avg_salary"],
                    "median_salary": r["median_salary"],
                }
            )
        return results

    async def discovered_domain_count(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM discovered_domains")
            return row["cnt"] if row else 0

    # ── Radar v2 methods ──────────────────────────────────────────────

    async def upsert_radar_candidate(self, data: dict[str, Any]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO radar_candidates (
                    canonical_id, source, direct_apply_url, normalized_company,
                    normalized_role, normalized_location, freshness_lane,
                    source_confidence, eligibility, rejection_reason,
                    rejection_detail, role_family, salary_amount, salary_currency,
                    salary_period, salary_raw, posted_date, first_seen, last_seen,
                    matching_skills, missing_skills, match_percent,
                    shortlist_probability, verdict, jd_summary,
                    company_description, role_summary, is_remote,
                    founders, funding_stage, funding_info, founder_socials,
                    company_news, osint_signals, extra
                ) VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,
                    $13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,
                    $28,$29,$30,$31,$32,$33,$34,$35
                )
                ON CONFLICT (canonical_id) DO UPDATE SET
                    last_seen = EXCLUDED.last_seen,
                    match_percent = GREATEST(
                        radar_candidates.match_percent, EXCLUDED.match_percent
                    ),
                    eligibility = EXCLUDED.eligibility,
                    matching_skills = EXCLUDED.matching_skills,
                    missing_skills = EXCLUDED.missing_skills,
                    verdict = EXCLUDED.verdict,
                    updated_at = NOW()
                """,
                data.get("canonical_id", ""),
                data.get("source", ""),
                data.get("direct_apply_url", ""),
                data.get("normalized_company", ""),
                data.get("normalized_role", ""),
                data.get("normalized_location", "Remote"),
                data.get("freshness_lane", "review"),
                data.get("source_confidence", 0.5),
                data.get("eligibility", "pending"),
                data.get("rejection_reason", ""),
                data.get("rejection_detail", ""),
                data.get("role_family", "unknown"),
                data.get("salary_amount"),
                data.get("salary_currency", ""),
                data.get("salary_period", ""),
                data.get("salary_raw", ""),
                data.get("posted_date", ""),
                data.get("first_seen", 0.0),
                data.get("last_seen", 0.0),
                json.dumps(data.get("matching_skills", [])),
                json.dumps(data.get("missing_skills", [])),
                int(data.get("match_percent", 0)),
                int(data.get("shortlist_probability", 0)),
                data.get("verdict", "NO_MATCH"),
                data.get("jd_summary", ""),
                data.get("company_description", ""),
                data.get("role_summary", ""),
                bool(data.get("is_remote", False)),
                json.dumps(data.get("founders", [])),
                data.get("funding_stage", ""),
                json.dumps(data.get("funding_info", {})),
                json.dumps(data.get("founder_socials", [])),
                data.get("company_news", ""),
                json.dumps(data.get("osint_signals", [])),
                json.dumps(data.get("extra", {})),
            )

    async def record_rejection(self, canonical_id: str, reason: str, detail: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE radar_candidates SET
                    eligibility = 'rejected',
                    rejection_reason = $2,
                    rejection_detail = $3,
                    updated_at = NOW()
                WHERE canonical_id = $1
                """,
                canonical_id,
                reason,
                detail,
            )

    async def get_rejection_counts_by_reason(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT rejection_reason, COUNT(*) as cnt
                FROM radar_candidates
                WHERE eligibility = 'rejected' AND rejection_reason != ''
                GROUP BY rejection_reason
                ORDER BY cnt DESC
                """
            )
        return [{"reason": r["rejection_reason"], "count": r["cnt"]} for r in rows]

    async def get_urgent_candidates(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM radar_candidates
                WHERE freshness_lane = 'urgent' AND eligibility = 'accepted'
                ORDER BY match_percent DESC LIMIT $1
                """,
                limit,
            )
        return [_row_to_radar_candidate(r) for r in rows]

    async def get_candidates_by_eligibility(
        self, eligibility: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM radar_candidates
                WHERE eligibility = $1
                ORDER BY match_percent DESC NULLS LAST
                LIMIT $2
                """,
                eligibility,
                limit,
            )
        return [_row_to_radar_candidate(r) for r in rows]

    async def insert_analytics_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO radar_analytics (event_type, event_data) VALUES ($1, $2)",
                event_type,
                json.dumps(event_data),
            )

    async def get_salary_stats(self) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) FILTER (
                            WHERE salary_amount IS NOT NULL AND salary_amount > 0
                        ) AS count_with_salary,
                        ROUND(AVG(salary_amount) FILTER (
                            WHERE salary_amount IS NOT NULL AND salary_amount > 0
                        ))::int AS avg_salary,
                        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary_amount)
                            FILTER (
                                WHERE salary_amount IS NOT NULL AND salary_amount > 0
                            ))::int AS median_salary
                    FROM radar_candidates
                    """
                )
                if row is None:
                    return {"count": 0, "avg": 0, "median": 0}
                return {
                    "count": row["count_with_salary"] or 0,
                    "avg": row["avg_salary"] or 0,
                    "median": row["median_salary"] or 0,
                }
            except Exception:
                return {"count": 0, "avg": 0, "median": 0, "error": "salary_stats_failed"}


def _row_to_radar_candidate(row: asyncpg.Record) -> dict[str, Any]:
    jsonb_cols = (
        "matching_skills",
        "missing_skills",
        "founders",
        "funding_info",
        "founder_socials",
        "osint_signals",
        "extra",
        "raw_json",
    )
    result: dict[str, Any] = {}
    for key in row:
        val = row[key]
        if key in jsonb_cols and isinstance(val, str):
            try:
                val = json.loads(val) if val else []
            except Exception:
                val = []
        result[key] = val
    return result


async def _radar_gate_stats(self: MemoryStore) -> dict[str, Any]:
    """Return counts by eligibility and rejection reason from radar_candidates."""
    async with self._pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE eligibility = 'accepted') AS accepted,
                COUNT(*) FILTER (WHERE eligibility = 'near_miss') AS near_miss,
                COUNT(*) FILTER (WHERE eligibility = 'rejected') AS rejected,
                COUNT(*) FILTER (WHERE eligibility = 'pending') AS pending,
                COUNT(*) FILTER (WHERE freshness_lane = 'urgent') AS urgent,
                COUNT(*) FILTER (WHERE freshness_lane = 'review') AS review
            FROM radar_candidates
            """
        )
        if row is None:
            return {
                "total": 0,
                "accepted": 0,
                "near_miss": 0,
                "rejected": 0,
                "pending": 0,
                "urgent": 0,
                "review": 0,
            }
        rejections = await conn.fetch(
            """
            SELECT rejection_reason, COUNT(*) AS cnt
            FROM radar_candidates
            WHERE eligibility = 'rejected' AND rejection_reason != ''
            GROUP BY rejection_reason ORDER BY cnt DESC LIMIT 8
            """
        )
        return {
            "total": row["total"] or 0,
            "accepted": row["accepted"] or 0,
            "near_miss": row["near_miss"] or 0,
            "rejected": row["rejected"] or 0,
            "pending": row["pending"] or 0,
            "urgent": row["urgent"] or 0,
            "review": row["review"] or 0,
            "top_rejection_reasons": [
                {"reason": r["rejection_reason"], "count": r["cnt"]} for r in rejections
            ],
        }


async def _radar_top_skills(self: MemoryStore, limit: int = 12) -> list[dict[str, Any]]:
    """Top matching skills from accepted/near-miss candidates."""
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT skill, COUNT(*) AS cnt
            FROM radar_candidates,
                 LATERAL jsonb_array_elements_text(matching_skills) AS skill
            WHERE eligibility IN ('accepted', 'near_miss')
              AND skill IS NOT NULL AND skill != ''
            GROUP BY skill ORDER BY cnt DESC LIMIT $1
            """,
            limit,
        )
    return [{"skill": r["skill"], "count": r["cnt"]} for r in rows]


async def _radar_skill_arbitrage(self: MemoryStore) -> list[dict[str, Any]]:
    """Missing skills that caused near-misses."""
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT skill, COUNT(*) AS miss_count
            FROM radar_candidates,
                 LATERAL jsonb_array_elements_text(missing_skills) AS skill
            WHERE eligibility = 'near_miss'
              AND skill IS NOT NULL AND skill != ''
            GROUP BY skill ORDER BY miss_count DESC LIMIT 12
            """
        )
    return [{"skill": r["skill"], "miss_count": r["miss_count"]} for r in rows]


MemoryStore.get_radar_gate_stats = _radar_gate_stats  # type: ignore[attr-defined]
MemoryStore.get_radar_top_skills = _radar_top_skills  # type: ignore[attr-defined]
MemoryStore.get_radar_skill_arbitrage = _radar_skill_arbitrage  # type: ignore[attr-defined]
