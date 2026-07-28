"""Native async job queue for producer-consumer decoupling within a single process."""

import asyncio
from dataclasses import dataclass, field


@dataclass
class QueuedJob:
    markdown: str
    url: str = ""
    title: str = ""
    snippet: str = ""


@dataclass
class JobPipeline:
    maxsize: int = 0
    _queue: asyncio.Queue[QueuedJob | None] = field(default_factory=lambda: asyncio.Queue())
    _done: asyncio.Event = field(default_factory=asyncio.Event)
    _scraped: int = 0
    _matched: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def push(self, job: QueuedJob) -> None:
        await self._queue.put(job)
        async with self._lock:
            self._scraped += 1

    def signal_done(self) -> None:
        self._done.set()

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    async def pop(self, timeout: float = 2.0) -> QueuedJob | None:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return item
        except TimeoutError:
            return None

    async def task_done(self) -> None:
        async with self._lock:
            self._matched += 1

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def scraped_count(self) -> int:
        return self._scraped

    @property
    def matched_count(self) -> int:
        return self._matched

    def log_status(self) -> str:
        return f"Queue: {self.pending} pending | {self._scraped} scraped | {self._matched} matched"
