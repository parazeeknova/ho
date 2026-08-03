"""Standalone pgvector memory engine for agent deduplication and semantic RAG.

Connects exclusively to the ``agent-memory-db`` service.
No dependency on Firebase or any other persistence layer.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from typing import Any

import asyncpg
from pgvector import Vector
from pgvector.asyncpg import register_vector

from src.configuration import PostgresConfig, get_config
from src.logging import get_logger

logger = get_logger("memory_store")

VECTOR_DIM = 1024


def _url_hash(url: str) -> str:
    """Stable hash used to key obs_embeddings rows (matches md5(o.url))."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


_EMBED_NOISE = re.compile(r"[\u0000-\u001f\u007f]+")


def _company_from_url(url: str) -> str:
    """Best-effort company name from common ATS URL shapes.

    - https://jobs.ashbyhq.com/<company>[/...]
    - https://job-boards.greenhouse.io/<company>[/...]
    - https://jobs.lever.co/<company>[/...]
    - https://<company>.applytojob.com or <company>.bamboohr.com/jobs
    Falls back to the last path segment only when it looks like a name.
    """
    try:
        import urllib.parse

        host = urllib.parse.urlparse(url).netloc.lower()
        path = urllib.parse.urlparse(url).path
    except Exception:
        return ""
    for needle in ("jobs.ashbyhq.com", "boards.greenhouse.io", "jobs.lever.co"):
        if needle in host:
            idx = host.find(needle)
            return host[idx + len(needle) + 1 :].strip() or _first_path(path)
    if "greenhouse.io" in host and "job-boards" in host:
        return _first_path(path)
    if "applytojob.com" in host:
        return host.split(".")[0]
    if "bamboohr.com" in host:
        return host.split(".")[0]
    if host in ("jobicy.com", "remoteok.com", "linkedin.com", "www.linkedin.com"):
        return ""
    return ""


def _first_path(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    return (
        parts[0][:60]
        if not parts[0].startswith("jobs")
        else (parts[1][:60] if len(parts) > 1 else "")
    )


def _build_embed_text(title: str, raw_json: Any) -> str:
    """Compact, embedding-friendly text for an observation.

    Keeps the title + company + the first ~400 words of readable JD text so
    the vector captures *what* the job is, not boilerplate. Lists (posting
    bullet text) are the highest-signal part of the raw payload.
    """
    parts: list[str] = []
    if title:
        parts.append(title)
    text = ""
    if isinstance(raw_json, str) and raw_json:
        try:
            raw_json = json.loads(raw_json)
        except Exception:
            raw_json = None
    if isinstance(raw_json, dict):
        parsed = raw_json.get("parsed")
        parsed_comp = ""
        if isinstance(parsed, dict):
            parsed_comp = str(parsed.get("company") or parsed.get("name") or "")
            if parsed_comp:
                parts.append(parsed_comp)
            content = str(parsed.get("description") or parsed.get("content") or "")
            if content:
                text = content
        raw_comp = str(
            raw_json.get("company_name")
            or raw_json.get("companyName")
            or (
                (raw_json.get("company") or {}).get("name", "")
                if isinstance(raw_json.get("company"), dict)
                else (raw_json.get("company") or "")
            )
            or ""
        )
        if raw_comp and raw_comp.lower() != parsed_comp.lower():
            parts.append(raw_comp)
        lists = raw_json.get("lists")
        if isinstance(lists, list):
            blobs: list[str] = []
            for item in lists:
                if isinstance(item, dict):
                    h = str(item.get("head", "") or item.get("text", "") or "")
                    c = str(item.get("content", "") or "")
                    if h:
                        blobs.append(h)
                    if c:
                        blobs.append(c)
            if blobs:
                joined = " ".join(blobs)
                if len(joined) > len(text):
                    text = joined
        if not text:
            for key in ("content", "markdown", "description"):
                v = raw_json.get(key)
                if isinstance(v, str) and v:
                    text = v
                    break
    text = _EMBED_NOISE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts.append(text[:1200])
    return " | ".join(p for p in parts if p)


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
    id           SERIAL PRIMARY KEY,
    section      VARCHAR(128),
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    embedding    vector({VECTOR_DIM})
);

ALTER TABLE resume_embeddings
ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';

-- Backfill content hashes for rows created before the column existed, so
-- the unique index below can be built on non-empty hashes.
UPDATE resume_embeddings
SET content_hash = encode(sha256(convert_to(content, 'UTF8')), 'hex')
WHERE content_hash = '';

CREATE UNIQUE INDEX IF NOT EXISTS idx_resume_embeddings_hash
    ON resume_embeddings (content_hash);

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
    poll_lane            TEXT NOT NULL DEFAULT 'high',
    yield_per_poll       REAL NOT NULL DEFAULT 0,
    last_change_at       DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at           TIMESTAMP DEFAULT NOW()
);

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS board_url TEXT DEFAULT '';

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS company_name TEXT DEFAULT '';

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS discovery_origin TEXT DEFAULT '';

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS poll_lane TEXT NOT NULL DEFAULT 'high';

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS yield_per_poll REAL NOT NULL DEFAULT 0;

ALTER TABLE source_checkpoints
ADD COLUMN IF NOT EXISTS last_change_at DOUBLE PRECISION NOT NULL DEFAULT 0;

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

CREATE TABLE IF NOT EXISTS salary_estimates (
    lookup_key    TEXT PRIMARY KEY,
    company       TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL DEFAULT '',
    amount_usd    DOUBLE PRECISION,
    currency      TEXT NOT NULL DEFAULT 'USD',
    period        TEXT NOT NULL DEFAULT 'year',
    raw           TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'searxng',
    searched_at   DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS company_osint (
    company      TEXT PRIMARY KEY,
    data         JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    cached_at    DOUBLE PRECISION NOT NULL DEFAULT 0,
    expires_at   DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_queue (
    id            BIGSERIAL PRIMARY KEY,
    canonical_id  TEXT NOT NULL,
    version       INT NOT NULL DEFAULT 1,
    priority      INT NOT NULL DEFAULT 50,
    payload       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    status        TEXT NOT NULL DEFAULT 'pending',
    attempts      INT NOT NULL DEFAULT 0,
    enqueued_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_until   TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ,
    UNIQUE (canonical_id, version)
);

CREATE INDEX IF NOT EXISTS idx_llm_queue_claim ON llm_queue(status, priority DESC, id);

ALTER TABLE jobs_ledger
ADD COLUMN IF NOT EXISTS embedding vector({VECTOR_DIM});

CREATE INDEX IF NOT EXISTS idx_radar_candidates_eligibility ON radar_candidates(eligibility);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_freshness ON radar_candidates(freshness_lane);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_rejection ON radar_candidates(rejection_reason);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_role_family ON radar_candidates(role_family);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_created ON radar_candidates(created_at);
CREATE INDEX IF NOT EXISTS idx_job_observations_source ON job_observations(source);
CREATE INDEX IF NOT EXISTS idx_job_observations_first_seen ON job_observations(first_seen);
CREATE INDEX IF NOT EXISTS idx_source_checkpoints_active ON source_checkpoints(active);
CREATE INDEX IF NOT EXISTS idx_radar_candidates_elig_created
    ON radar_candidates(eligibility, created_at);
CREATE INDEX IF NOT EXISTS idx_job_observations_source_seen
    ON job_observations(source, last_seen);
CREATE INDEX IF NOT EXISTS idx_jobs_ledger_match
    ON jobs_ledger(match_percent DESC);

CREATE TABLE IF NOT EXISTS obs_embeddings (
    url_hash     TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    company      TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    embedding    vector({VECTOR_DIM}),
    embedded_at  TIMESTAMP DEFAULT NOW()
);

ALTER TABLE obs_embeddings
ADD COLUMN IF NOT EXISTS content_hash TEXT NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS embed_cache (
    text_hash   TEXT PRIMARY KEY,
    embedding   vector({VECTOR_DIM}),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS http_cache (
    url_hash      TEXT PRIMARY KEY,
    url           TEXT NOT NULL DEFAULT '',
    status        INT DEFAULT 200,
    etag          TEXT DEFAULT '',
    last_modified TEXT DEFAULT '',
    content_type  TEXT DEFAULT '',
    body          TEXT DEFAULT '',
    body_hash     TEXT DEFAULT '',
    fetched_at    DOUBLE PRECISION DEFAULT 0,
    ttl_seconds   INT DEFAULT 900
);
CREATE INDEX IF NOT EXISTS idx_http_cache_fetched ON http_cache (fetched_at);

CREATE TABLE IF NOT EXISTS evidence (
    company_id    TEXT NOT NULL,
    company_name  TEXT DEFAULT '',
    claim         TEXT NOT NULL,
    evidence_type TEXT NOT NULL DEFAULT 'signal',
    source        TEXT NOT NULL DEFAULT '',
    weight        REAL DEFAULT 0.3,
    contradicts   BOOLEAN DEFAULT FALSE,
    ref_url       TEXT DEFAULT '',
    observed_at   DOUBLE PRECISION DEFAULT 0,
    created_at    TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (company_id, claim, source)
);
CREATE INDEX IF NOT EXISTS idx_evidence_company_observed
    ON evidence (company_id, observed_at DESC);
"""


def _jsonb_val(val: Any, default: Any) -> Any:
    """Coerce a value to a plain container for a jsonb column.

    Guards against string-typed JSON (e.g. a model returning ``"[]"``
    instead of an empty list) that would otherwise be stored as a jsonb
    *string* and break every list-typed renderer downstream.

    Returns the raw container (never a pre-serialized string): every pool
    connection registers a jsonb codec (``json.dumps``) in ``_init``, so a
    str return here would be serialized a second time and stored as a jsonb
    string scalar, which breaks ``jsonb_array_length`` in upsert conflicts.
    """
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = None
        if parsed is not None:
            val = parsed
    if isinstance(val, (list, dict, int, float, bool)):
        return val
    return default


class MemoryStore:
    """Async connection-pool-backed pgvector store."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls, config: PostgresConfig | None = None) -> MemoryStore:
        """Initialise pool, register vector type, create tables."""
        cfg = config or get_config().postgres

        async def _init(conn) -> None:
            await register_vector(conn)
            # Every pool connection must share the same jsonb codec. Registering
            # it only on the startup connection left later-spawned connections
            # on asyncpg's default codec, so the same row could decode as a
            # JSON string on one connection and a dict on another (and INSERTs
            # would double-encode pre-serialized strings through json.dumps).
            await conn.set_type_codec(
                "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
            )

        pool = await asyncpg.create_pool(
            cfg.dsn, min_size=cfg.min_pool, max_size=cfg.max_pool, init=_init
        )
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(CREATE_TABLES_SQL)
            # Lightweight column migrations for tables that predate schema
            # additions (CREATE TABLE IF NOT EXISTS won't touch them).
            await conn.execute(
                "ALTER TABLE company_osint ADD COLUMN IF NOT EXISTS "
                "expires_at DOUBLE PRECISION NOT NULL DEFAULT 0"
            )
            await cls._create_hnsw_indexes(conn)
            await cls._prune_llm_queue(conn)
            await cls._prune_embed_cache(conn)
            await cls._prune_http_cache(conn)
        logger.info("MemoryStore initialized", dsn=cfg.dsn.split("@")[-1])
        return cls(pool)

    @staticmethod
    async def _create_hnsw_indexes(conn) -> None:
        """Build ANN indexes on the hot vector-search columns.

        Wrapped in suppress: HNSW needs pgvector >= 0.5.0, and a missing
        index must never block store startup (searches just degrade to
        exact scans).
        """
        with contextlib.suppress(Exception):
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_resume_embeddings_hnsw
                ON resume_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        with contextlib.suppress(Exception):
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_ledger_embedding_hnsw
                ON jobs_ledger USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
        with contextlib.suppress(Exception):
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_obs_embeddings_hnsw
                ON obs_embeddings USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )

    @staticmethod
    async def _prune_llm_queue(conn, older_than_days: int = 7) -> None:
        """Drop settled queue rows so llm_queue never grows unbounded."""
        with contextlib.suppress(Exception):
            await conn.execute(
                "DELETE FROM llm_queue "
                "WHERE status IN ('done', 'error') "
                "AND completed_at < NOW() - ($1::int * INTERVAL '1 day')",
                older_than_days,
            )

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
                data,
            )

    # Resume_embeddings

    async def index_resume_chunks(
        self, chunks: list[dict[str, Any]], current_hashes: set[str] | None = None
    ) -> None:
        """Upsert resume chunks by content hash.

        Each chunk dict must have keys: ``section``, ``content``,
        ``content_hash`` (sha256 of content) and ``embedding`` (list[float]
        of length 1024).

        Rows whose hash is absent from *current_hashes* (the full set of
        chunks in the resume right now) are deleted as stale, so a changed
        resume rebuilds cleanly while unchanged chunks survive without
        re-embedding. When *current_hashes* is None, chunks are inserted
        without pruning.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            if current_hashes is not None:
                rows = await conn.fetch("SELECT content_hash FROM resume_embeddings")
                stale = [
                    r["content_hash"]
                    for r in rows
                    if r["content_hash"] and r["content_hash"] not in current_hashes
                ]
                if stale:
                    await conn.execute(
                        "DELETE FROM resume_embeddings WHERE content_hash = ANY($1::text[])",
                        stale,
                    )
            for ch in chunks:
                emb = Vector(ch["embedding"])
                await conn.execute(
                    """
                    INSERT INTO resume_embeddings (section, content, content_hash, embedding)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        section = EXCLUDED.section,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                    """,
                    ch["section"],
                    ch["content"],
                    ch["content_hash"],
                    emb,
                )

    async def existing_resume_hashes(self, hashes: list[str]) -> set[str]:
        """Return which of the given content hashes already have an embedding
        row, so callers can skip re-embedding unchanged resume chunks."""
        if not hashes:
            return set()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content_hash FROM resume_embeddings WHERE content_hash = ANY($1::text[])",
                hashes,
            )
        return {r["content_hash"] for r in rows}

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

    # obs_embeddings: vector intelligence over the job corpus

    async def upsert_obs_embedding(
        self,
        url_hash: str,
        title: str,
        company: str,
        embedding: list[float],
        content_hash: str = "",
    ) -> None:
        """Insert or refresh a single observation embedding (cosine-normalized).

        *content_hash* fingerprints the embedded text (md5 of raw_json) so
        unchanged observations are never re-embedded or re-written.
        """
        vec = Vector(embedding)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO obs_embeddings
                    (url_hash, title, company, content_hash, embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (url_hash) DO UPDATE SET
                    title = EXCLUDED.title,
                    company = EXCLUDED.company,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    embedded_at = NOW()
                """,
                url_hash,
                title,
                company,
                content_hash,
                vec,
            )

    async def missing_obs_hashes(self, url_hashes: list[str], limit: int = 2000) -> list[str]:
        """Return which of the given url hashes have no embedding yet (batch-safe)."""
        if not url_hashes:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT h AS url_hash
                FROM unnest($1::text[]) AS h
                WHERE NOT EXISTS (SELECT 1 FROM obs_embeddings e WHERE e.url_hash = h)
                LIMIT $2
                """,
                url_hashes,
                limit,
            )
        return [r["url_hash"] for r in rows]

    # embed_cache: content-hash-keyed embedding cache so identical text is
    # never re-sent to the (shared) llama-server.

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        """Return the cached embedding for a text hash, or None on miss."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT embedding FROM embed_cache WHERE text_hash = $1", text_hash
            )
        if row is None or row["embedding"] is None:
            return None
        return [float(v) for v in row["embedding"].to_list()]

    async def put_cached_embedding(self, text_hash: str, embedding: list[float]) -> None:
        """Store an embedding keyed by text hash (refresh timestamp on hit)."""
        vec = Vector(embedding)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO embed_cache (text_hash, embedding, created_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (text_hash) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
                """,
                text_hash,
                vec,
            )

    @staticmethod
    async def _prune_embed_cache(conn, older_than_days: int = 30) -> None:
        """Drop embed_cache rows older than the TTL so the table stays bounded."""
        with contextlib.suppress(Exception):
            await conn.execute(
                "DELETE FROM embed_cache WHERE created_at < NOW() - ($1::int * INTERVAL '1 day')",
                older_than_days,
            )

    # http_cache: shared ETag/response cache for polled endpoints

    async def get_http_cache_row(self, url_hash: str) -> dict[str, Any] | None:
        """Return the cached response row for a URL hash, or None on miss."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT url_hash, url, status, etag, last_modified, content_type, "
                "body, body_hash, fetched_at, ttl_seconds "
                "FROM http_cache WHERE url_hash = $1",
                url_hash,
            )
        return dict(row) if row else None

    async def upsert_http_cache(
        self,
        url_hash: str,
        url: str,
        status: int,
        etag: str,
        last_modified: str,
        content_type: str,
        body: str,
        body_hash: str,
        ttl_seconds: int,
    ) -> None:
        """Insert or refresh a cached response row."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO http_cache
                    (url_hash, url, status, etag, last_modified, content_type,
                     body, body_hash, fetched_at, ttl_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (url_hash) DO UPDATE SET
                    url = EXCLUDED.url,
                    status = EXCLUDED.status,
                    etag = EXCLUDED.etag,
                    last_modified = EXCLUDED.last_modified,
                    content_type = EXCLUDED.content_type,
                    body = EXCLUDED.body,
                    body_hash = EXCLUDED.body_hash,
                    fetched_at = EXCLUDED.fetched_at,
                    ttl_seconds = EXCLUDED.ttl_seconds
                """,
                url_hash,
                url,
                status,
                etag,
                last_modified,
                content_type,
                body,
                body_hash,
                time.time(),
                ttl_seconds,
            )

    async def update_http_cache_fetched(self, url_hash: str, etag: str, last_modified: str) -> None:
        """Refresh fetched_at/etag without replacing the body (200, unchanged)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE http_cache SET fetched_at = $2, "
                "etag = $3, last_modified = $4 WHERE url_hash = $1",
                url_hash,
                time.time(),
                etag,
                last_modified,
            )

    @staticmethod
    async def _prune_http_cache(conn, older_than_days: int = 30) -> None:
        """Drop cache rows untouched for the TTL so the table stays bounded."""
        with contextlib.suppress(Exception):
            await conn.execute(
                "DELETE FROM http_cache WHERE fetched_at < $1::double precision",
                time.time() - older_than_days * 86400,
            )

    # evidence: weighted belief ledger per company (hiring signals)

    async def record_evidence(
        self,
        company_id: str,
        claim: str,
        source: str,
        *,
        company_name: str = "",
        evidence_type: str = "signal",
        weight: float = 0.3,
        contradicts: bool = False,
        ref_url: str = "",
    ) -> None:
        """Upsert one evidence row: (company_id, claim, source) is unique, so
        repeated observations refresh the timestamp instead of accumulating
        duplicate rows. Age-based freshness decay handles staleness.
        """
        with contextlib.suppress(Exception):
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO evidence
                        (company_id, company_name, claim, evidence_type, source,
                         weight, contradicts, ref_url, observed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (company_id, claim, source) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        weight = EXCLUDED.weight,
                        contradicts = EXCLUDED.contradicts,
                        ref_url = EXCLUDED.ref_url,
                        observed_at = EXCLUDED.observed_at
                    """,
                    company_id,
                    company_name,
                    claim,
                    evidence_type,
                    source,
                    weight,
                    contradicts,
                    ref_url,
                    time.time(),
                )

    async def get_evidence(
        self, company_id: str, since: float | None = None
    ) -> list[dict[str, Any]]:
        """Return evidence rows for a company, most recent first."""
        async with self._pool.acquire() as conn:
            if since is not None:
                rows = await conn.fetch(
                    "SELECT company_id, company_name, claim, evidence_type, source, "
                    "weight, contradicts, ref_url, observed_at "
                    "FROM evidence WHERE company_id = $1 AND observed_at >= $2 "
                    "ORDER BY observed_at DESC",
                    company_id,
                    since,
                )
            else:
                rows = await conn.fetch(
                    "SELECT company_id, company_name, claim, evidence_type, source, "
                    "weight, contradicts, ref_url, observed_at "
                    "FROM evidence WHERE company_id = $1 ORDER BY observed_at DESC",
                    company_id,
                )
        return [dict(r) for r in rows]

    async def evidence_summary(self, company_id: str) -> dict[str, Any]:
        """Compact belief summary for a company: supporting vs contradicting
        rows plus an effective confidence via combine_evidence.
        """
        rows = await self.get_evidence(company_id)
        if not rows:
            return {"rows": [], "support": 0, "contradict": 0, "confidence": 0.5}
        from src.graph.entity import combine_evidence

        support = sum(1 for r in rows if not r.get("contradicts"))
        contradict = sum(1 for r in rows if r.get("contradicts"))
        confidence = combine_evidence(rows)
        return {
            "rows": rows,
            "support": support,
            "contradict": contradict,
            "confidence": confidence.score,
        }

    async def unembedded_obs(
        self,
        limit: int = 2000,
        software_first: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch observations that lack an embedding row, software-first.

        Returns dicts with ``url_hash``, ``title``, ``company`` and the text to
        embed. ``company`` is best-effort: the crawler stores it inside
        ``raw_json`` (often ``parsed.company`` or ``company.name``); when absent
        we fall back to the first URL path segment as a weak signal.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT o.url, o.title, o.raw_json, o.first_seen,
                       md5(o.raw_json::text) AS content_hash
                FROM job_observations o
                LEFT JOIN obs_embeddings e ON e.url_hash = md5(o.url)
                WHERE (e.url_hash IS NULL
                       OR e.content_hash IS DISTINCT FROM md5(o.raw_json::text))
                  AND o.url IS NOT NULL
                ORDER BY (
                    CASE WHEN $2::bool AND lower(o.title) ~
                        'software|engineer|developer|full.?stack|backend|frontend|'
                        'devops|sre|data|machine|ml|ai|python|java|golang|rust|'
                        'intern|new grad|junior|entry|graduate'
                         THEN 0 ELSE 1 END
                ), o.first_seen DESC
                LIMIT $1
                """,
                limit,
                software_first,
            )
        out: list[dict[str, Any]] = []
        for r in rows:
            raw = r["raw_json"]
            company = ""
            if isinstance(raw, str) and raw:
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = None
            if isinstance(raw, dict):
                parsed = raw.get("parsed") or {}
                if isinstance(parsed, dict):
                    company = str(parsed.get("company") or parsed.get("name") or "")
                if not company:
                    comp = raw.get("company")
                    if isinstance(comp, dict):
                        company = str(comp.get("name") or "")
                    elif isinstance(comp, str):
                        company = comp
                if not company:
                    company = str(raw.get("company_name") or raw.get("companyName") or "")
                if not company:
                    # greenhouse/jobicy etc. often carry company in nested fields
                    for key in ("organization", "employer", "hiring_company", "host_company"):
                        v = raw.get(key)
                        if isinstance(v, dict):
                            company = str(v.get("name") or "")
                            if company:
                                break
            if not company:
                company = _company_from_url(r["url"])
            out.append(
                {
                    "url_hash": _url_hash(r["url"]),
                    "title": r["title"] or "",
                    "company": company,
                    "content_hash": r["content_hash"] or "",
                    "text": _build_embed_text(r["title"], r["raw_json"]),
                }
            )
        return out

    async def resume_centroid(self) -> list[float] | None:
        """Mean of all resume chunk embeddings (L2-normalized). None if empty."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT embedding FROM resume_embeddings WHERE embedding IS NOT NULL"
            )
        if not rows:
            return None
        dim = VECTOR_DIM
        import numpy as np

        acc = np.zeros(dim, dtype=np.float32)
        for r in rows:
            acc += np.asarray(r["embedding"].to_list(), dtype=np.float32)
        acc /= len(rows)
        n = np.linalg.norm(acc)
        if n > 0:
            acc /= n
        return acc.tolist()

    async def obs_nearest(
        self, query_emb: list[float], top_k: int = 10, exclude: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Top-k nearest embedded observations by cosine distance."""
        vec = Vector(query_emb)
        async with self._pool.acquire() as conn:
            if exclude:
                rows = await conn.fetch(
                    """
                    SELECT e.title, e.company, e.embedding <=> $1 AS distance
                    FROM obs_embeddings e
                    WHERE e.url_hash <> ALL($2::text[])
                    ORDER BY distance ASC LIMIT $3
                    """,
                    vec,
                    list(exclude),
                    top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT e.title, e.company, e.embedding <=> $1 AS distance
                    FROM obs_embeddings e
                    ORDER BY distance ASC LIMIT $2
                    """,
                    vec,
                    top_k,
                )
        return [
            {"title": r["title"], "company": r["company"], "distance": float(r["distance"])}
            for r in rows
        ]

    async def company_centroids(self, min_obs: int = 2) -> list[dict[str, Any]]:
        """Mean embedding per company (L2-normalized) plus obs count."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT company, COUNT(*) AS n
                FROM obs_embeddings
                WHERE company <> '' AND embedding IS NOT NULL
                GROUP BY company HAVING COUNT(*) >= $1
                """,
                min_obs,
            )
        if not rows:
            return []
        import numpy as np

        dim = VECTOR_DIM
        out: list[dict[str, Any]] = []
        async with self._pool.acquire() as conn:
            for r in rows:
                embs = await conn.fetch(
                    "SELECT embedding FROM obs_embeddings "
                    "WHERE company = $1 AND embedding IS NOT NULL",
                    r["company"],
                )
                acc = np.zeros(dim, dtype=np.float32)
                for e in embs:
                    acc += np.asarray(e["embedding"].to_list(), dtype=np.float32)
                acc /= max(len(embs), 1)
                n = np.linalg.norm(acc)
                if n > 0:
                    acc /= n
                out.append({"company": r["company"], "centroid": acc.tolist(), "n": r["n"]})
        return out

    # Companies / intel

    async def accepted_companies(self, limit: int = 200) -> list[str]:
        """Companies with at least one accepted candidate, newest first."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT normalized_company
                FROM radar_candidates
                WHERE eligibility = 'accepted' AND normalized_company <> ''
                GROUP BY normalized_company
                ORDER BY MAX(created_at) DESC
                LIMIT $1
                """,
                limit,
            )
        return [r["normalized_company"] for r in rows]

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
                _jsonb_val(data.get("founders", []), []),
                data.get("funding_stage", ""),
                _jsonb_val(data.get("funding_info", {}), {}),
                _jsonb_val(data.get("founder_socials", []), []),
                data.get("company_news", ""),
                _jsonb_val(data.get("osint_signals", []), []),
                data.get("source_url", data.get("url", "")),
                data,
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

    # Radar v2 methods

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
                    first_seen = LEAST(
                        radar_candidates.first_seen, EXCLUDED.first_seen
                    ),
                    match_percent = GREATEST(
                        radar_candidates.match_percent, EXCLUDED.match_percent
                    ),
                    shortlist_probability = GREATEST(
                        radar_candidates.shortlist_probability,
                        EXCLUDED.shortlist_probability
                    ),
                    eligibility = EXCLUDED.eligibility,
                    rejection_reason = COALESCE(
                        NULLIF(EXCLUDED.rejection_reason, ''),
                        radar_candidates.rejection_reason
                    ),
                    matching_skills = EXCLUDED.matching_skills,
                    missing_skills = EXCLUDED.missing_skills,
                    verdict = EXCLUDED.verdict,
                    jd_summary = COALESCE(
                        NULLIF(EXCLUDED.jd_summary, ''),
                        radar_candidates.jd_summary
                    ),
                    company_description = COALESCE(
                        NULLIF(EXCLUDED.company_description, ''),
                        radar_candidates.company_description
                    ),
                    role_summary = COALESCE(
                        NULLIF(EXCLUDED.role_summary, ''),
                        radar_candidates.role_summary
                    ),
                    founders = CASE
                        WHEN jsonb_array_length(EXCLUDED.founders) > 0
                        THEN EXCLUDED.founders
                        ELSE radar_candidates.founders
                    END,
                    funding_stage = COALESCE(
                        NULLIF(EXCLUDED.funding_stage, ''),
                        radar_candidates.funding_stage
                    ),
                    funding_info = CASE
                        WHEN jsonb_typeof(EXCLUDED.funding_info) = 'object'
                             AND (EXCLUDED.funding_info <> '{}'::jsonb)
                        THEN EXCLUDED.funding_info
                        ELSE radar_candidates.funding_info
                    END,
                    founder_socials = CASE
                        WHEN jsonb_array_length(EXCLUDED.founder_socials) > 0
                        THEN EXCLUDED.founder_socials
                        ELSE radar_candidates.founder_socials
                    END,
                    company_news = COALESCE(
                        NULLIF(EXCLUDED.company_news, ''),
                        radar_candidates.company_news
                    ),
                    osint_signals = CASE
                        WHEN jsonb_array_length(EXCLUDED.osint_signals) > 0
                        THEN EXCLUDED.osint_signals
                        ELSE radar_candidates.osint_signals
                    END,
                    extra = radar_candidates.extra || EXCLUDED.extra,
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
                _jsonb_val(data.get("matching_skills", []), []),
                _jsonb_val(data.get("missing_skills", []), []),
                int(data.get("match_percent", 0)),
                int(data.get("shortlist_probability", 0)),
                data.get("verdict", "NO_MATCH"),
                data.get("jd_summary", ""),
                data.get("company_description", ""),
                data.get("role_summary", ""),
                bool(data.get("is_remote", False)),
                _jsonb_val(data.get("founders", []), []),
                data.get("funding_stage", ""),
                _jsonb_val(data.get("funding_info", {}), {}),
                _jsonb_val(data.get("founder_socials", []), []),
                data.get("company_news", ""),
                _jsonb_val(data.get("osint_signals", []), []),
                _jsonb_val(data.get("extra", {}), {}),
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
                event_data,
            )

    _OSINT_CACHE_TTL_SECONDS = 7 * 86400
    _OSINT_DEGRADED_TTL_SECONDS = 6 * 3600

    async def get_company_osint(self, company: str) -> dict[str, Any] | None:
        """Return cached company OSINT enrichment if fresh, else None.

        Cache keys are normalized to lowercase on write and read, so a
        ``Cloudflare`` candidate matches a ``cloudflare`` cache row.
        """
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    "SELECT data, expires_at FROM company_osint WHERE company = $1",
                    (company or "").strip().lower(),
                )
            except Exception:
                return None
        if row is None:
            return None
        if time.time() >= (row.get("expires_at") or 0):
            return None
        data = row.get("data")
        try:
            if isinstance(data, str):
                return json.loads(data or "{}")
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    async def put_company_osint(
        self, company: str, data: dict[str, Any], degraded: bool = False
    ) -> None:
        """Insert or refresh the cached OSINT payload for a company.

        A "degraded" payload (rate-limited sources, no enrichment produced)
        gets a short TTL so the next sweep retries instead of serving the
        empty result for the full week.
        """
        ttl = self._OSINT_DEGRADED_TTL_SECONDS if degraded else self._OSINT_CACHE_TTL_SECONDS
        now = time.time()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO company_osint (company, data, cached_at, expires_at) "
                "VALUES ($1, $2::jsonb, $3, $4) "
                "ON CONFLICT (company) DO UPDATE SET "
                "data = EXCLUDED.data, cached_at = EXCLUDED.cached_at, "
                "expires_at = EXCLUDED.expires_at",
                (company or "").strip().lower(),
                data,
                now,
                now + ttl,
            )

    async def get_salary_stats(self) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) AS count_with_salary,
                        ROUND(AVG(salary_amount))::int AS avg_salary,
                        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary_amount))::int
                            AS median_salary
                    FROM radar_candidates
                    WHERE salary_amount IS NOT NULL AND salary_amount > 0
                      AND salary_currency = 'USD'
                      AND salary_period = 'year'
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

    async def learned_title_scores(self, min_obs: int = 8, top_k: int = 200) -> dict[str, float]:
        """Learn a per-keyword gate-pass score from historical candidates.

        For each space-separated token in ``normalized_role``, count how often a
        candidate with that token passed the gate (accepted/near_miss) vs was
        rejected. Returns a map of keyword -> pass-rate in [0,1]. Tokens with
        fewer than ``min_obs`` observations are omitted (unreliable signal).

        ``min_obs`` defaults to 8 so that one-off keywords (e.g. a single
        "restaurant" or "music" role that happened to pass) don't pollute the
        signal. The drain uses this to order never-gated observations by
        *learned* pass probability instead of hand-maintained regex tiers, so it
        self-adapts as the corpus and the gate evolve.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT lower(normalized_role) AS role, eligibility
                FROM radar_candidates
                WHERE normalized_role IS NOT NULL AND normalized_role != ''
                """
            )
        counts: dict[str, int] = {}
        passes: dict[str, int] = {}
        for r in rows:
            tokens = set((r["role"] or "").split())
            ok = r["eligibility"] in ("accepted", "near_miss")
            for tok in tokens:
                tok = tok.strip("(),/&+-")
                if len(tok) < 2:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
                if ok:
                    passes[tok] = passes.get(tok, 0) + 1
        scores: dict[str, float] = {}
        for tok, cnt in counts.items():
            if cnt >= min_obs:
                scores[tok] = passes.get(tok, 0) / cnt
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], -counts[kv[0]]))
        return dict(ranked[:top_k])


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
    """Top matching skills from accepted/near-miss candidates.

    ``matching_skills`` can be a proper jsonb array OR a string that itself
    holds a JSON array (a double-encoded artifact of the LLM path). We first
    normalise any scalar/string to a jsonb array so ``jsonb_array_elements``
    never crashes on a scalar.
    """
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH norm AS (
                SELECT CASE
                    WHEN jsonb_typeof(matching_skills) = 'array' THEN matching_skills
                    WHEN jsonb_typeof(matching_skills) = 'string'
                         THEN (matching_skills #>> '{}')::jsonb
                    ELSE '[]'::jsonb
                END AS skills
                FROM radar_candidates
                WHERE eligibility IN ('accepted', 'near_miss')
            )
            SELECT skill, COUNT(*) AS cnt
            FROM norm, LATERAL jsonb_array_elements_text(skills) AS skill
            WHERE skill IS NOT NULL AND skill != ''
            GROUP BY skill ORDER BY cnt DESC LIMIT $1
            """,
            limit,
        )
    return [{"skill": r["skill"], "count": r["cnt"]} for r in rows]


async def _radar_skill_arbitrage(self: MemoryStore) -> list[dict[str, Any]]:
    """Missing skills that caused near-misses (double-encoding safe)."""
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            WITH norm AS (
                SELECT CASE
                    WHEN jsonb_typeof(missing_skills) = 'array' THEN missing_skills
                    WHEN jsonb_typeof(missing_skills) = 'string'
                         THEN (missing_skills #>> '{}')::jsonb
                    ELSE '[]'::jsonb
                END AS skills
                FROM radar_candidates
                WHERE eligibility = 'near_miss'
            )
            SELECT skill, COUNT(*) AS miss_count
            FROM norm, LATERAL jsonb_array_elements_text(skills) AS skill
            WHERE skill IS NOT NULL AND skill != ''
            GROUP BY skill ORDER BY miss_count DESC LIMIT 12
            """
        )
    return [{"skill": r["skill"], "miss_count": r["miss_count"]} for r in rows]


async def _get_recent_accepts(self: MemoryStore, hours: int = 24) -> list[dict[str, Any]]:
    """Accepted candidates in the last N hours with their timestamp."""
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT normalized_company, normalized_role, created_at
            FROM radar_candidates
            WHERE eligibility = 'accepted' AND created_at > NOW() - ($1::int * INTERVAL '1 hour')
            ORDER BY created_at DESC
            """,
            hours,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        ts = r["created_at"].timestamp() if r["created_at"] else 0.0
        out.append(
            {
                "company": r["normalized_company"],
                "role": r["normalized_role"],
                "ts": ts,
            }
        )
    return out


async def _get_near_miss_count(self: MemoryStore) -> int:
    async with self._pool.acquire() as conn:
        return (
            await conn.fetchval(
                "SELECT COUNT(*) FROM radar_candidates WHERE eligibility = 'near_miss'"
            )
            or 0
        )


async def _get_top_companies(self: MemoryStore, limit: int = 8) -> list[dict[str, Any]]:
    """Accepted companies ranked by accepted-count then avg match percent."""
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT normalized_company AS company,
                   COUNT(*) AS accepted,
                   ROUND(AVG(match_percent))::int AS avg_match,
                   MAX(funding_stage) AS funding_stage
            FROM radar_candidates
            WHERE eligibility = 'accepted' AND normalized_company <> ''
            GROUP BY normalized_company
            ORDER BY accepted DESC, avg_match DESC
            LIMIT $1
            """,
            limit,
        )
    return [
        {
            "company": r["company"],
            "accepted": r["accepted"],
            "avg_match": r["avg_match"],
            "funding_stage": r["funding_stage"] or "seed",
        }
        for r in rows
    ]


async def _get_sector_signal(self: MemoryStore, limit: int = 6) -> list[dict[str, Any]]:
    """Sector labels from accepted candidates.

    Uses role_family when it's meaningful; otherwise infers a sector from the
    role title so "unknown" never dominates the report.
    """
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT CASE
                WHEN role_family IS NOT NULL AND role_family NOT IN ('', 'unknown')
                     THEN role_family
                WHEN lower(normalized_role) ~ 'machine learning|ml|ai/|ai |research|llm|genai'
                     THEN 'ai_ml'
                WHEN lower(normalized_role) ~ 'data (engineer|scientist|analyst)|analytics'
                     THEN 'data'
                WHEN lower(normalized_role) ~ 'full.?stack|frontend|front end|react|ui'
                     THEN 'frontend'
                WHEN lower(normalized_role) ~ 'backend|api|server'
                     THEN 'backend'
                WHEN lower(normalized_role) ~ 'devops|sre|platform|infra|cloud|site reliability'
                     THEN 'infra_platform'
                WHEN lower(normalized_role) ~ 'security|cyber'
                     THEN 'security'
                WHEN lower(normalized_role) ~ 'mobile|ios|android'
                     THEN 'mobile'
                WHEN lower(normalized_role) ~ 'founder|founding|co.founder|head of engineering'
                     THEN 'startup_founding'
                ELSE 'general_swe'
            END AS label,
            COUNT(*) AS cnt
            FROM radar_candidates
            WHERE eligibility = 'accepted'
            GROUP BY label
            ORDER BY cnt DESC
            LIMIT $1
            """,
            limit,
        )
        total = (
            await conn.fetchval(
                "SELECT COUNT(*) FROM radar_candidates WHERE eligibility = 'accepted'"
            )
            or 0
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        cnt = r["cnt"]
        out.append(
            {
                "label": r["label"],
                "count": cnt,
                "pct": round(cnt / total * 100) if total else 0,
            }
        )
    return out


MemoryStore.get_radar_gate_stats = _radar_gate_stats  # type: ignore[attr-defined]
MemoryStore.get_radar_top_skills = _radar_top_skills  # type: ignore[attr-defined]
MemoryStore.get_radar_skill_arbitrage = _radar_skill_arbitrage  # type: ignore[attr-defined]
MemoryStore.get_recent_accepts = _get_recent_accepts  # type: ignore[attr-defined]
MemoryStore.get_near_miss_count = _get_near_miss_count  # type: ignore[attr-defined]
MemoryStore.get_top_companies = _get_top_companies  # type: ignore[attr-defined]
MemoryStore.get_sector_signal = _get_sector_signal  # type: ignore[attr-defined]
