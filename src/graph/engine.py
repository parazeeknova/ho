"""Persistent WorkScheduler — long-lived worker pool as the central execution
engine. Replaces batch-oriented PriorityScheduler.

Workers consume from the CrawlFrontier continuously. New work can be
inserted at any time; no call to run() is needed. Graceful shutdown
drains outstanding work. Metrics are exposed for observability.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from src.graph.entity import FrontierEntry, SchedulerMetrics
from src.graph.frontier import CrawlFrontier

console = Console()

AgentHandler = Any  # Callable[[FrontierEntry], Coroutine[Any, Any, list[FrontierEntry]]]


class WorkScheduler:
    def __init__(
        self,
        frontier: CrawlFrontier,
        worker_count: int = 4,
        idle_sleep: float = 1.0,
    ) -> None:
        self.frontier = frontier
        self._worker_count = worker_count
        self._idle_sleep = idle_sleep
        self._agents: dict[str, AgentHandler] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._drain_event = asyncio.Event()
        self._active_sem: asyncio.Semaphore | None = None

        # Metrics
        self._start_time = 0.0
        self._metrics = SchedulerMetrics()
        self._metrics_lock = asyncio.Lock()

        # Adaptive concurrency
        self._consecutive_failures = 0
        self._consecutive_idles = 0
        self._base_worker_count = worker_count

    # ── Agent registration ────────────────────────────────────────────────────

    def register_agent(self, name: str, handler: AgentHandler) -> None:
        self._agents[name] = handler

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, worker_count: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._shutdown_event.clear()
        self._drain_event.clear()
        self._start_time = time.monotonic()
        self._active_sem = asyncio.Semaphore(worker_count or self._worker_count)
        for i in range(worker_count or self._worker_count):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))
        console.print(f"  ⚙️  [WorkScheduler] Started {worker_count or self._worker_count} workers")

    async def shutdown(self, drain: bool = True) -> None:
        if not self._running:
            return
        console.print("  ⚙️  [WorkScheduler] Shutting down...")
        self._running = False
        if drain:
            self._drain_event.set()
        self._shutdown_event.set()

        for w in self._workers:
            w.cancel()
        results = await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        console.print(f"  ⚙️  [WorkScheduler] Shutdown complete ({len(results)} workers)")

    async def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            entry = self.frontier.pop()
            if entry is None:
                self._consecutive_idles += 1
                if self._drain_event.is_set() and self.frontier.empty:
                    return
                await asyncio.sleep(self._idle_sleep)
                continue

            handler = self._agents.get(entry.agent)
            if handler is None:
                self.frontier.mark_done(entry.id)
                continue

            async with self._active_sem:  # type: ignore[union-attr]
                async with self._metrics_lock:
                    self._metrics.active_workers += 1

                try:
                    result = await handler(entry)
                    self._consecutive_idles = 0
                    self._consecutive_failures = 0
                    async with self._metrics_lock:
                        self._metrics.completed_work += 1

                    # Feed results back into frontier
                    if isinstance(result, list):
                        added = self.frontier.push_many(result)
                        if added:
                            async with self._metrics_lock:
                                self._metrics.total_enqueued += added

                    self.frontier.mark_done(entry.id)

                except Exception as e:
                    self._consecutive_failures += 1
                    async with self._metrics_lock:
                        self._metrics.retried_work += 1
                    console.print(
                        f"  [yellow]Worker {worker_id} failed {entry.agent}: {e}[/yellow]"
                    )
                    if entry.can_execute:
                        self.frontier.requeue(entry)
                    else:
                        async with self._metrics_lock:
                            self._metrics.failed_work += 1
                        self.frontier.mark_done(entry.id)
                finally:
                    async with self._metrics_lock:
                        self._metrics.active_workers -= 1

            # Adaptive concurrency: scale down on consecutive failures
            if self._consecutive_failures >= 5:
                self._active_sem = asyncio.Semaphore(max(1, self._base_worker_count // 2))  # type: ignore[assignment]

    # ── Enqueue from outside ──────────────────────────────────────────────────

    def enqueue(self, entry: FrontierEntry) -> bool:
        return self.frontier.push(entry)

    def enqueue_many(self, entries: list[FrontierEntry]) -> int:
        return self.frontier.push_many(entries)

    # ── Metrics ───────────────────────────────────────────────────────────────

    async def get_metrics(self) -> SchedulerMetrics:
        async with self._metrics_lock:
            m = SchedulerMetrics(
                active_workers=self._metrics.active_workers,
                pending_work=self.frontier.pending,
                completed_work=self._metrics.completed_work,
                retried_work=self._metrics.retried_work,
                failed_work=self._metrics.failed_work,
                total_enqueued=self._metrics.total_enqueued,
                uptime_s=time.monotonic() - self._start_time,
            )
        return m

    async def metrics_table(self) -> Table:
        m = await self.get_metrics()
        t = Table(title="WorkScheduler")
        t.add_column("Metric", style="bold")
        t.add_column("Value")
        t.add_row("Active workers", str(m.active_workers))
        t.add_row("Pending work", str(m.pending_work))
        t.add_row("Completed", str(m.completed_work))
        t.add_row("Retried", str(m.retried_work))
        t.add_row("Failed", str(m.failed_work))
        t.add_row("Total enqueued", str(m.total_enqueued))
        t.add_row("Uptime", f"{m.uptime_s:.0f}s")
        return t

    # ── Query ─────────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def worker_count(self) -> int:
        return len(self._workers)
