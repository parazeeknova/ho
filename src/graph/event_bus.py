"""Event Bus — publish-only. Fire-and-forget: publishing spawns handler
tasks and returns immediately. Handlers produce FrontierEntries that
are enqueued asynchronously into the scheduler.

Key properties:
  • fire() returns immediately (no awaiting subscribers).
  • Handlers are pure translators: event → list[FrontierEntry].
  • A callback (set_enqueue_cb) receives results for async enqueuing.
  • Deterministic event IDs prevent duplicate processing.
  • Background tasks are supervised with automatic cleanup and diagnostics.
  • Event deduplication uses a size-bounded TTL cache to prevent unbounded
    memory growth.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field

from src.configuration import EventBusConfig, get_config
from src.graph.entity import FrontierEntry, GraphEvent, NodeType, make_event_id
from src.logging import get_logger

Handler = Callable[[GraphEvent], Awaitable[list[FrontierEntry]]]
EnqueueCallback = Callable[[list[FrontierEntry]], Awaitable[None]]

logger = get_logger("event_bus")


# Size-bounded TTL cache for event deduplication


@dataclass
class _CacheEntry:
    inserted_at: float = field(default_factory=time.monotonic)


class TTLSet:
    """A size-bounded set with TTL expiry. Entries older than *ttl* seconds
    are considered absent (expired). When at capacity, the oldest entries
    are evicted first.
    """

    def __init__(self, maxsize: int = 10000, ttl: float = 3600.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._dict: dict[str, _CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    def _evict_stale(self) -> None:
        now = time.monotonic()
        stale = [k for k, v in self._dict.items() if now - v.inserted_at >= self._ttl]
        for k in stale:
            del self._dict[k]

    def add(self, key: str) -> None:
        self._evict_stale()
        if key in self._dict:
            # Already present; refresh TTL by reinserting
            self._dict[key] = _CacheEntry()
            return
        if len(self._dict) >= self._maxsize:
            # Evict oldest
            oldest = min(self._dict, key=lambda k: self._dict[k].inserted_at)
            del self._dict[oldest]
        self._dict[key] = _CacheEntry()

    def __contains__(self, key: str) -> bool:
        entry = self._dict.get(key)
        if entry is None:
            self._misses += 1
            return False
        now = time.monotonic()
        if now - entry.inserted_at >= self._ttl:
            del self._dict[key]
            self._misses += 1
            return False
        self._hits += 1
        return True

    def __len__(self) -> int:
        self._evict_stale()
        return len(self._dict)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def stats(self) -> dict:
        return {
            "size": len(self),
            "capacity": self._maxsize,
            "ttl_s": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
        }


# Background task supervisor


class TaskSupervisor:
    """Registry of background tasks with diagnostics and graceful shutdown."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._completed_total: int = 0
        self._failed_total: int = 0

    def _done_callback(self, task: asyncio.Task) -> None:
        """Called when a background task finishes (success or failure)."""
        # Schedule the removal to avoid modifying the set during callback
        asyncio.ensure_future(self._cleanup_task(task))

    async def _cleanup_task(self, task: asyncio.Task) -> None:
        async with self._lock:
            self._tasks.discard(task)
        try:
            exc = task.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                self._failed_total += 1
                logger.exception("EventBus background task failed", exc=exc)
            else:
                self._completed_total += 1
        except asyncio.CancelledError:
            self._failed_total += 1
            logger.error("EventBus task cleanup was cancelled")
        except Exception as e:
            self._failed_total += 1
            logger.exception("EventBus task result check failed", exc=e)

    async def spawn(self, coro: Coroutine[None, None, None]) -> None:
        """Create a supervised background task."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        async with self._lock:
            self._tasks.add(task)
        task.add_done_callback(self._done_callback)

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel remaining tasks and wait for completion."""
        async with self._lock:
            tasks = list(self._tasks)
            self._tasks.clear()

        for t in tasks:
            t.cancel()

        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            for t in pending:
                logger.warning(
                    f"Background task did not terminate within {timeout}s",
                    task_name=getattr(t, "get_name", lambda: "?")(),
                )

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    @property
    def stats(self) -> dict:
        return {
            "active": len(self._tasks),
            "completed": self._completed_total,
            "failed": self._failed_total,
        }


# EventBus


class EventBus:
    def __init__(self, config: EventBusConfig | None = None) -> None:
        cfg = config or get_config().event_bus
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._seen_ids = TTLSet(maxsize=cfg.cache_maxsize, ttl=cfg.cache_ttl)
        self._lock = asyncio.Lock()
        self._fired_count = 0
        self._duplicate_count = 0
        self._enqueue_cb: EnqueueCallback | None = None
        self._supervisor = TaskSupervisor()
        self._shutdown_event = asyncio.Event()

        async def _evict_loop() -> None:
            while not self._shutdown_event.is_set():
                self._seen_ids._evict_stale()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=60.0)

        _task = asyncio.create_task(_evict_loop())
        self._supervisor._tasks.add(_task)
        _task.add_done_callback(self._supervisor._done_callback)

    # callback registration

    def set_enqueue_callback(self, cb: EnqueueCallback) -> None:
        self._enqueue_cb = cb

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers[event_type].remove(handler)

    # event creation

    def new_event(
        self,
        event_type: str,
        node_id: str,
        node_type: NodeType,
        payload: dict | None = None,
    ) -> GraphEvent:
        return GraphEvent(
            event_type=event_type,
            node_id=node_id,
            node_type=node_type,
            payload=payload or {},
        )

    # fire

    async def fire(self, event: GraphEvent) -> None:
        """Fire-and-forget: spawn handler tasks, return immediately.

        Results are enqueued asynchronously via the callback set by
        set_enqueue_callback. Deduplication uses deterministic event IDs
        backed by a size-bounded TTL cache.
        """
        event_id = make_event_id(event.event_type, event.node_id)
        async with self._lock:
            if event_id in self._seen_ids:
                self._duplicate_count += 1
                return
            self._seen_ids.add(event_id)

        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return

        self._fired_count += 1

        async def _run_and_enqueue() -> None:
            tasks = [h(event) for h in handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            entries: list[FrontierEntry] = []
            for r in results:
                if isinstance(r, list):
                    entries.extend(r)
                elif isinstance(r, Exception):
                    logger.exception(
                        "EventBus handler failed",
                        exc=r,
                        event_type=event.event_type,
                        node_id=event.node_id,
                    )
            if entries and self._enqueue_cb:
                await self._enqueue_cb(entries)

        await self._supervisor.spawn(_run_and_enqueue())

    # lifecycle

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Gracefully shut down background tasks including TTL eviction loop."""
        self._shutdown_event.set()
        await self._supervisor.shutdown(timeout=timeout)

    # diagnostics

    @property
    def fired_count(self) -> int:
        return self._fired_count

    @property
    def duplicate_count(self) -> int:
        return self._duplicate_count

    @property
    def active_tasks(self) -> int:
        return self._supervisor.active_count

    @property
    def cache_stats(self) -> dict:
        return self._seen_ids.stats

    @property
    def task_stats(self) -> dict:
        return self._supervisor.stats
