"""Event Bus — publish-only. Fire-and-forget: publishing spawns handler
tasks and returns immediately. Handlers produce FrontierEntries that
are enqueued asynchronously into the scheduler.

Key properties:
  • fire() returns immediately (no awaiting subscribers).
  • Handlers are pure translators: event → list[FrontierEntry].
  • A callback (set_enqueue_cb) receives results for async enqueuing.
  • Deterministic event IDs prevent duplicate processing.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from collections.abc import Awaitable, Callable

from src.graph.entity import FrontierEntry, GraphEvent, NodeType, make_event_id

Handler = Callable[[GraphEvent], Awaitable[list[FrontierEntry]]]
EnqueueCallback = Callable[[list[FrontierEntry]], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._seen_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._fired_count = 0
        self._enqueue_cb: EnqueueCallback | None = None

    def set_enqueue_callback(self, cb: EnqueueCallback) -> None:
        self._enqueue_cb = cb

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers[event_type].remove(handler)

    def new_event(
        self, event_type: str, node_id: str, node_type: NodeType, payload: dict | None = None
    ) -> GraphEvent:
        return GraphEvent(
            event_type=event_type, node_id=node_id, node_type=node_type, payload=payload or {}
        )

    async def fire(self, event: GraphEvent) -> None:
        """Fire-and-forget: spawn handler tasks, return immediately.

        Results are enqueued asynchronously via the callback set by
        set_enqueue_callback. Deduplication uses deterministic event IDs.
        """
        event_id = make_event_id(event.event_type, event.node_id)
        async with self._lock:
            if event_id in self._seen_ids:
                return
            self._seen_ids.add(event_id)

        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return

        async def _run_and_enqueue():
            tasks = [h(event) for h in handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            entries: list[FrontierEntry] = []
            for r in results:
                if isinstance(r, list):
                    entries.extend(r)
            if entries and self._enqueue_cb:
                await self._enqueue_cb(entries)

        asyncio.create_task(_run_and_enqueue())
        self._fired_count += 1

    @property
    def fired_count(self) -> int:
        return self._fired_count
