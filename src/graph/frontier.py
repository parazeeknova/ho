"""Crawl Frontier — deduplicable, dependency-aware work queue with dynamic
priority recalculation and persistent state.

The frontier decides *what* executes next. Every entry has a deterministic
ID so duplicate work merges instead of executing twice.
"""

from __future__ import annotations

import json
import time
from heapq import heappop, heappush

import asyncpg

from src.graph.entity import FrontierEntry, NodeType

MAX_QUEUE_SIZE = 500


class CrawlFrontier:
    def __init__(self, pool: asyncpg.Pool | None = None, max_size: int = MAX_QUEUE_SIZE) -> None:
        self._pool = pool
        self._max_size = max_size
        self._heap: list[tuple[int, int, FrontierEntry]] = []  # (-priority, seq, entry)
        self._index: dict[str, FrontierEntry] = {}
        self._seq = 0
        self._completed: set[str] = set()
        self._total_enqueued = 0
        self._total_completed = 0

    # ── Enqueue ──────────────────────────────────────────────────────────────

    def push(self, entry: FrontierEntry) -> bool:
        """Push a new entry. If the same work ID already exists, merge or skip."""
        if entry.id in self._completed:
            return False

        existing = self._index.get(entry.id)
        if existing:
            existing.priority = max(existing.priority, entry.priority)
            existing.retries = min(existing.retries, entry.retries)
            existing.payload = {**existing.payload, **entry.payload}
            existing.updated_at = time.monotonic()
            return True

        self._seq += 1
        self._total_enqueued += 1
        self._index[entry.id] = entry
        heappush(self._heap, (-entry.priority, self._seq, entry))
        return True

    def push_many(self, entries: list[FrontierEntry]) -> int:
        added = 0
        for e in entries:
            if self.push(e):
                added += 1
        self._trim()
        return added

    # ── Dequeue ──────────────────────────────────────────────────────────────

    def pop(self) -> FrontierEntry | None:
        """Pop the highest-priority entry whose dependencies are satisfied."""
        retry_list: list[FrontierEntry] = []
        result: FrontierEntry | None = None

        while self._heap:
            _, _, entry = heappop(self._heap)
            if entry.id in self._completed:
                continue
            if not entry.can_execute:
                continue
            if not self._dependencies_satisfied(entry):
                retry_list.append(entry)
                continue
            result = entry
            break

        for e in retry_list:
            self._seq += 1
            heappush(self._heap, (-e.priority, self._seq, e))

        return result

    def mark_done(self, entry_id: str) -> None:
        self._completed.add(entry_id)
        self._total_completed += 1
        self._index.pop(entry_id, None)

    def requeue(self, entry: FrontierEntry) -> None:
        entry.retries += 1
        entry.updated_at = time.monotonic()
        self._seq += 1
        heappush(self._heap, (-entry.priority, self._seq, entry))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _dependencies_satisfied(self, entry: FrontierEntry) -> bool:
        if not entry.dependencies:
            return True
        return all(dep in self._completed for dep in entry.dependencies)

    def _trim(self) -> None:
        while len(self._heap) > self._max_size:
            _, _, entry = heappop(self._heap)
            self._index.pop(entry.id, None)

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def pending(self) -> int:
        return len(self._heap)

    @property
    def completed(self) -> int:
        return self._total_completed

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued

    @property
    def empty(self) -> bool:
        return len(self._heap) == 0

    # ── Persistence (optional, via pgvector) ─────────────────────────────────

    async def persist(self) -> int:
        if self._pool is None:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            for entry in self._index.values():
                await conn.execute(
                    """
                    INSERT INTO frontier_state (work_id, agent, node_id, priority,
                        depth, retries, payload, updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8)
                    ON CONFLICT (work_id) DO UPDATE SET
                        priority = GREATEST(frontier_state.priority, EXCLUDED.priority),
                        retries = LEAST(frontier_state.retries, EXCLUDED.retries),
                        payload = frontier_state.payload || EXCLUDED.payload,
                        updated_at = EXCLUDED.updated_at
                    """,
                    entry.id,
                    entry.agent,
                    entry.node_id,
                    entry.priority,
                    entry.depth,
                    entry.retries,
                    json.dumps(entry.payload),
                    entry.updated_at,
                )
                count += 1
        return count

    async def restore(self) -> int:
        if self._pool is None:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM frontier_state ORDER BY priority DESC LIMIT $1",
                self._max_size,
            )
            for row in rows:
                work_id = row["work_id"]
                if work_id in self._completed:
                    continue
                entry = FrontierEntry(
                    id=work_id,
                    agent=row["agent"],
                    node_id=row["node_id"],
                    node_type=NodeType(row.get("node_type", "company")),
                    priority=row["priority"],
                    depth=row.get("depth", 0),
                    retries=row.get("retries", 0),
                    payload=row.get("payload") or {},
                    updated_at=row.get("updated_at", time.monotonic()),
                )
                self._index[work_id] = entry
                self._seq += 1
                heappush(self._heap, (-entry.priority, self._seq, entry))
                count += 1
        self._total_enqueued += count
        return count


def create_frontier_sql() -> str:
    return """
    CREATE TABLE IF NOT EXISTS frontier_state (
        work_id     TEXT PRIMARY KEY,
        agent       TEXT NOT NULL,
        node_id     TEXT NOT NULL,
        node_type   TEXT DEFAULT 'company',
        priority    INT DEFAULT 50,
        depth       INT DEFAULT 0,
        retries     INT DEFAULT 0,
        payload     JSONB DEFAULT '{}'::jsonb,
        created_at  TIMESTAMP DEFAULT NOW(),
        updated_at  DOUBLE PRECISION DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS frontier_completed (
        work_id     TEXT PRIMARY KEY,
        completed_at TIMESTAMP DEFAULT NOW()
    );
    """
