"""Standalone pgvector memory engine for agent deduplication and semantic RAG.

Connects exclusively to the ``agent-memory-db`` service (port 5433).
No dependency on Firebase or any other persistence layer.
"""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from pgvector import Vector
from pgvector.asyncpg import register_vector

DSN = "postgresql://postgres:postgres@localhost:5433/agent_memory"
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
"""


class MemoryStore:
    """Async connection-pool-backed pgvector store."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def create(cls) -> MemoryStore:
        """Initialise pool, register vector type, create tables."""
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=4)
        async with pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(CREATE_TABLES_SQL)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    # ── processed_jobs (deduplication ledger) ────────────────────────────

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

    # ── resume_embeddings (semantic RAG memory) ──────────────────────────

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
