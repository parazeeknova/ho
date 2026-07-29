"""Async Event Bus — publish-only event system.

Handlers subscribe by event type. When an event fires, handlers run
concurrently and return FrontierEntries for the scheduler. The
scheduler is the sole execution authority; the bus does not
coordinate or execute work.

Deduplication uses deterministic event IDs.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from src.graph.entity import FrontierEntry, GraphEvent, NodeType, make_event_id

Handler = Callable[[GraphEvent], Awaitable[list[FrontierEntry]]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._seen_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._fired_count = 0
        self._start_time = time.monotonic()

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers[event_type].remove(handler)

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

    async def fire(self, event: GraphEvent) -> list[FrontierEntry]:
        """Fire event: run subscribers, return FrontierEntries for scheduler.

        Dedup: deterministic event IDs prevent duplicate processing.
        """
        event_id = make_event_id(event.event_type, event.node_id)
        async with self._lock:
            if event_id in self._seen_ids:
                return []
            self._seen_ids.add(event_id)

        handlers = self._subscribers.get(event.event_type, [])
        if not handlers:
            return []

        tasks = [h(event) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        entries: list[FrontierEntry] = []
        for r in results:
            if isinstance(r, list):
                entries.extend(r)

        self._fired_count += 1
        return entries

    @property
    def fired_count(self) -> int:
        return self._fired_count

    @property
    def uptime(self) -> float:
        return time.monotonic() - self._start_time
