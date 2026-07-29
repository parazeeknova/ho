"""Concurrency-safe Crawl Frontier with leasing heartbeats, dependency
auto-unlock, work batching, and atomic distributed acquisition.

Lock-protected (asyncio.Lock). Workers block on not_empty Event.
Every state change auto-persists to DB when pool is available.
Restore uses FOR UPDATE SKIP LOCKED for safe distributed acquisition.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from heapq import heappop, heappush

import asyncpg

from src.configuration import SchedulerConfig, get_config
from src.graph.entity import (
    FrontierEntry,
    NodeType,
    WorkBatch,
    WorkState,
)
from src.logging import get_logger

logger = get_logger("frontier")


class CrawlFrontier:
    def __init__(
        self,
        pool: asyncpg.Pool | None = None,
        max_size: int | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        cfg = config or get_config().scheduler
        self._pool = pool
        self._max_size = max_size if max_size is not None else cfg.max_queue_size
        self._heap: list[tuple[int, int, FrontierEntry]] = []
        self._index: dict[str, FrontierEntry] = {}
        self._seq = 0
        self._total_enqueued = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_expired = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._dependency_index: dict[str, set[str]] = {}  # dep_id -> set of blocked entry_ids

    # Enqueue

    async def push(self, entry: FrontierEntry) -> bool:
        async with self._lock:
            if entry.state == WorkState.COMPLETED:
                return False
            existing = self._index.get(entry.id)
            if existing:
                existing.priority = max(existing.priority, entry.priority)
                existing.payload = {**existing.payload, **entry.payload}
                existing.updated_at = time.monotonic()
                await self._persist_one(existing)
                return True
            entry.state = WorkState.PENDING
            self._seq += 1
            self._total_enqueued += 1
            self._index[entry.id] = entry
            for dep in entry.dependencies:
                self._dependency_index.setdefault(dep, set()).add(entry.id)
            heappush(self._heap, (-entry.priority, self._seq, entry))
            await self._persist_one(entry)
            self._not_empty.set()
            return True

    async def push_many(self, entries: list[FrontierEntry]) -> int:
        added = 0
        for e in entries:
            if await self.push(e):
                added += 1
        async with self._lock:
            await self._trim()
        return added

    # Lease

    async def lease(self, worker_id: int) -> FrontierEntry | None:
        while True:
            entry = await self._try_lease(worker_id)
            if entry is not None:
                return entry
            self._not_empty.clear()
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(self._not_empty.wait(), timeout=2.0)

    async def lease_batch(self, agent: str, worker_id: int, max_batch: int = 5) -> WorkBatch | None:
        entries: list[FrontierEntry] = []
        for _ in range(max_batch):
            e = await self._try_lease(worker_id, agent=agent)
            if e is None:
                break
            entries.append(e)
            self._not_empty.clear()
        return WorkBatch(entries=entries, agent=agent) if entries else None

    async def _try_lease(self, worker_id: int, agent: str | None = None) -> FrontierEntry | None:
        async with self._lock:
            self._expire_stale_leases()
            retry: list[FrontierEntry] = []
            while self._heap:
                _, _, entry = heappop(self._heap)
                if entry.state == WorkState.COMPLETED:
                    continue
                if entry.state == WorkState.LEASED and not entry.lease_expired:
                    retry.append(entry)
                    continue
                if not entry.can_execute:
                    continue
                if agent is not None and entry.agent != agent:
                    retry.append(entry)
                    continue
                if entry.dependencies and not self._deps_satisfied(entry):
                    retry.append(entry)
                    continue

                entry.state = WorkState.LEASED
                entry.lease_holder = worker_id
                entry.lease_expires = time.monotonic() + entry._lease_ttl
                result = entry

                for e in retry:
                    self._seq += 1
                    heappush(self._heap, (-e.priority, self._seq, e))
                await self._persist_one(entry)
                return result

            for e in retry:
                self._seq += 1
                heappush(self._heap, (-e.priority, self._seq, e))
        return None

    async def renew_lease(self, entry_id: str) -> bool:
        async with self._lock:
            entry = self._index.get(entry_id)
            if entry and entry.state == WorkState.LEASED:
                entry.renew_lease()
                await self._persist_one(entry)
                return True
        return False

    # Complete / Fail

    async def complete(self, entry_id: str) -> list[FrontierEntry]:
        """Mark done. Returns newly-unblocked dependent entries."""
        async with self._lock:
            unblocked = self._unblock_dependents(entry_id)
            entry = self._index.pop(entry_id, None)
            if entry:
                entry.state = WorkState.COMPLETED
                self._total_completed += 1
            await self._persist_completed(entry_id)
            if unblocked:
                self._not_empty.set()
            return unblocked

    async def fail(self, entry_id: str, retry: bool = True) -> None:
        async with self._lock:
            entry = self._index.get(entry_id)
            if entry is None:
                return
            if retry and entry.retries < entry.max_retries:
                entry.retries += 1
                entry.state = WorkState.PENDING
                entry.lease_holder = -1
                entry.lease_expires = 0.0
                entry.updated_at = time.monotonic()
                self._seq += 1
                heappush(self._heap, (-entry.priority, self._seq, entry))
                self._not_empty.set()
                logger.info(
                    "Frontier entry retried", entity=entry.node_id, retry_count=entry.retries
                )
            else:
                self._index.pop(entry_id, None)
                self._total_failed += 1
                logger.warning(
                    "Frontier entry permanently failed",
                    entity=entry.node_id,
                    retry_count=entry.retries,
                )

    # Dependencies

    def _deps_satisfied(self, entry: FrontierEntry) -> bool:
        return all(
            self._index.get(d, FrontierEntry("x", "x", "x")).state == WorkState.COMPLETED
            for d in entry.dependencies
        )

    def _unblock_dependents(self, completed_id: str) -> list[FrontierEntry]:
        blocked_ids = self._dependency_index.pop(completed_id, set())
        unblocked: list[FrontierEntry] = []
        for bid in blocked_ids:
            entry = self._index.get(bid)
            if entry and entry.state == WorkState.PENDING and self._deps_satisfied(entry):
                self._seq += 1
                heappush(self._heap, (-entry.priority, self._seq, entry))
                unblocked.append(entry)
        return unblocked

    # Internal

    def _expire_stale_leases(self) -> int:
        now = time.monotonic()
        count = 0
        for _, entry in list(self._index.items()):
            if entry.state == WorkState.LEASED and now > entry.lease_expires:
                entry.state = WorkState.PENDING
                entry.lease_holder = -1
                entry.lease_expires = 0.0
                self._seq += 1
                heappush(self._heap, (-entry.priority, self._seq, entry))
                self._not_empty.set()
                count += 1
                logger.debug("Lease expired", entity=entry.node_id)
        self._total_expired += count
        return count

    async def _trim(self) -> None:
        while len(self._heap) > self._max_size:
            _, _, entry = heappop(self._heap)
            self._index.pop(entry.id, None)
            logger.debug("Frontier trimmed entry", entity=entry.node_id)

    # Queries

    @property
    def pending(self) -> int:
        return sum(1 for e in self._index.values() if e.state == WorkState.PENDING)

    @property
    def completed(self) -> int:
        return self._total_completed

    @property
    def failed(self) -> int:
        return self._total_failed

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued

    @property
    def empty(self) -> bool:
        return self.pending == 0

    @property
    def not_empty_event(self) -> asyncio.Event:
        return self._not_empty

    # Persistence

    async def _persist_one(self, entry: FrontierEntry) -> None:
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO frontier_state (work_id, agent, node_id, priority,
                        depth, state, retries, lease_expires, payload, updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10)
                    ON CONFLICT (work_id) DO UPDATE SET
                        priority = EXCLUDED.priority, state = EXCLUDED.state,
                        retries = EXCLUDED.retries, lease_expires = EXCLUDED.lease_expires,
                        payload = frontier_state.payload || EXCLUDED.payload,
                        updated_at = EXCLUDED.updated_at""",
                    entry.id,
                    entry.agent,
                    entry.node_id,
                    entry.priority,
                    entry.depth,
                    entry.state.value,
                    entry.retries,
                    entry.lease_expires,
                    json.dumps(entry.payload),
                    entry.updated_at,
                )
        except Exception as e:
            logger.warning("Frontier persist failed", entity=entry.node_id, exception=str(e))

    async def _persist_completed(self, entry_id: str) -> None:
        if not self._pool:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("DELETE FROM frontier_state WHERE work_id = $1", entry_id)
                await conn.execute(
                    "INSERT INTO frontier_completed (work_id) VALUES ($1) ON CONFLICT DO NOTHING",
                    entry_id,
                )
        except Exception as e:
            logger.warning("Frontier persist completed failed", entity=entry_id, exception=str(e))

    async def persist_all(self) -> int:
        if not self._pool:
            return 0
        count = 0
        async with self._lock:
            for entry in self._index.values():
                await self._persist_one(entry)
                count += 1
        return count

    async def restore(self) -> int:
        if not self._pool:
            return 0
        count = 0
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM frontier_state WHERE state != 'completed'
                   ORDER BY priority DESC LIMIT $1 FOR UPDATE SKIP LOCKED""",
                self._max_size,
            )
            async with self._lock:
                for row in rows:
                    wid = row["work_id"]
                    if wid in self._index:
                        continue
                    entry = FrontierEntry(
                        id=wid,
                        agent=row["agent"],
                        node_id=row["node_id"],
                        node_type=NodeType(row.get("node_type", "company")),
                        priority=row["priority"],
                        depth=row.get("depth", 0),
                        state=WorkState(row.get("state", "pending")),
                        retries=row.get("retries", 0),
                        lease_expires=row.get("lease_expires", 0.0),
                        payload=row.get("payload") or {},
                        updated_at=row.get("updated_at", time.monotonic()),
                    )
                    if entry.state in (WorkState.LEASED, WorkState.RUNNING):
                        entry.lease_expires = 0.0
                        entry.lease_holder = -1
                        entry.state = WorkState.PENDING
                    self._index[wid] = entry
                    self._seq += 1
                    for dep in entry.dependencies:
                        self._dependency_index.setdefault(dep, set()).add(wid)
                    heappush(self._heap, (-entry.priority, self._seq, entry))
                    count += 1
        self._total_enqueued += count
        if count:
            self._not_empty.set()
        return count
