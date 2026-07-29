"""Event-driven WorkScheduler — proper global concurrency, lease heartbeats,
work batching, and utility-based scheduling.

Key invariants:
  • Global per-agent asyncio.Semaphores (created once) enforce true
    concurrency limits — not worker-local copies.
  • Long-running tasks heartbeat their lease every HEARTBEAT_INTERVAL.
  • Compatible work items (same agent) are batched before execution.
  • Workers block on frontier.lease() — no polling.
  • Adaptive scaling based on queue depth and failure rate.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rich.console import Console
from rich.table import Table

from src.graph.entity import (
    AGENT_BATCHABLE,
    AGENT_CONCURRENCY,
    HEARTBEAT_INTERVAL,
    FrontierEntry,
    SchedulerMetrics,
    WorkBatch,
    WorkState,
)
from src.graph.frontier import CrawlFrontier

console = Console()
AgentHandler = Any


class WorkScheduler:
    def __init__(self, frontier: CrawlFrontier, worker_count: int = 4) -> None:
        self.frontier = frontier
        self._base_workers = worker_count
        self._max_workers = worker_count * 3
        self._agents: dict[str, AgentHandler] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._shutdown = asyncio.Event()
        self._drain_mode = asyncio.Event()

        # Global per-agent semaphores — created ONCE, shared across all workers
        self._agent_sems: dict[str, asyncio.Semaphore] = {
            name: asyncio.Semaphore(limit) for name, limit in AGENT_CONCURRENCY.items()
        }

        # Track active leases for heartbeat cancellation on shutdown
        self._active_leases: dict[str, asyncio.Task[None]] = {}

        # Adaptive
        self._consecutive_failures = 0
        self._consecutive_empty = 0

        # Metrics
        self._start_time = 0.0
        self._metrics = SchedulerMetrics()
        self._metrics_lock = asyncio.Lock()

    def register_agent(self, name: str, handler: AgentHandler) -> None:
        self._agents[name] = handler
        if name not in self._agent_sems:
            self._agent_sems[name] = asyncio.Semaphore(3)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, worker_count: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._shutdown.clear()
        self._drain_mode.clear()
        self._start_time = time.monotonic()
        for i in range(worker_count or self._base_workers):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))
        console.print(f"  ⚙️  [Scheduler] {worker_count or self._base_workers} workers")

    async def shutdown(self, drain: bool = True) -> None:
        if not self._running:
            return
        console.print("  ⚙️  [Scheduler] Shutting down...")
        self._running = False
        if drain:
            self._drain_mode.set()
        self._shutdown.set()

        for task in self._active_leases.values():
            task.cancel()
        self._active_leases.clear()

        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await self.frontier.persist_all()
        console.print("  ⚙️  [Scheduler] Shutdown complete")

    # ── Worker loop ────────────────────────────────────────────────────────

    async def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                entry = await asyncio.wait_for(self.frontier.lease(worker_id), timeout=3.0)
            except TimeoutError:
                if self._drain_mode.is_set() and self.frontier.pending == 0:
                    return
                self._adapt_idle()
                continue

            if entry is None:
                if self._drain_mode.is_set():
                    return
                self._adapt_idle()
                continue

            if entry.agent not in self._agents:
                await self.frontier.complete(entry.id)
                continue

            self._consecutive_empty = 0
            handler = self._agents[entry.agent]
            sem = self._agent_sems.get(entry.agent, asyncio.Semaphore(3))

            async with sem:
                async with self._metrics_lock:
                    self._metrics.active_workers += 1

                entry.state = WorkState.RUNNING
                heartbeat_task = asyncio.create_task(self._heartbeat_loop(entry.id))
                self._active_leases[entry.id] = heartbeat_task

                try:
                    # Check if agent supports batching
                    if entry.agent in AGENT_BATCHABLE:
                        batch = await self.frontier.lease_batch(entry.agent, worker_id, max_batch=3)
                        if batch and len(batch.entries) > 1:
                            async with self._metrics_lock:
                                self._metrics.batches_executed += 1
                            new_entries = await self._execute_batch(batch, handler)
                        else:
                            new_entries = await self._execute_one(entry, handler)
                    else:
                        new_entries = await self._execute_one(entry, handler)

                    self._consecutive_failures = 0
                    async with self._metrics_lock:
                        self._metrics.completed_work += 1
                        if hasattr(entry, "cost"):
                            self._metrics.cost_consumed += entry.cost.total_apx

                    if isinstance(new_entries, list) and new_entries:
                        added = await self.frontier.push_many(new_entries)
                        if added:
                            async with self._metrics_lock:
                                self._metrics.total_enqueued += added

                    unblocked = await self.frontier.complete(entry.id)
                    if unblocked:
                        await self.frontier.push_many(unblocked)

                except Exception as e:
                    self._consecutive_failures += 1
                    async with self._metrics_lock:
                        self._metrics.retried_work += 1
                    console.print(f"  [yellow]W{worker_id} {entry.agent}: {e}[/yellow]")
                    await self.frontier.fail(entry.id, retry=entry.retries < entry.max_retries)
                finally:
                    heartbeat_task.cancel()
                    self._active_leases.pop(entry.id, None)
                    async with self._metrics_lock:
                        self._metrics.active_workers -= 1

            self._adapt_concurrency()

    async def _execute_one(self, entry: FrontierEntry, handler) -> list[FrontierEntry]:
        try:
            return await handler(entry)
        except Exception:
            raise

    async def _execute_batch(self, batch: WorkBatch, handler) -> list[FrontierEntry]:
        results: list[FrontierEntry] = []
        for entry in batch.entries:
            try:
                r = await handler(entry)
                if r:
                    results.extend(r)
                unblocked = await self.frontier.complete(entry.id)
                if unblocked:
                    results.extend(unblocked)
            except Exception:
                await self.frontier.fail(entry.id, retry=entry.retries < entry.max_retries)
        return results

    async def _heartbeat_loop(self, entry_id: str) -> None:
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            ok = await self.frontier.renew_lease(entry_id)
            if not ok:
                return

    # ── Adaptive ───────────────────────────────────────────────────────────

    async def _adapt_concurrency(self) -> None:
        pending = self.frontier.pending
        if self._consecutive_failures >= 5:
            for name in AGENT_CONCURRENCY:
                self._agent_sems[name] = asyncio.Semaphore(
                    max(1, AGENT_CONCURRENCY.get(name, 3) // 2)
                )
            self._consecutive_failures = 0
            console.print("  [dim]Backpressure: halved all agent concurrency[/dim]")
        elif pending > 50:
            for name, limit in AGENT_CONCURRENCY.items():
                sem = self._agent_sems.get(name)
                if sem is None:
                    self._agent_sems[name] = asyncio.Semaphore(limit)
                else:
                    self._agent_sems[name] = asyncio.Semaphore(min(limit * 2, limit + 4))

    def _adapt_idle(self) -> None:
        self._consecutive_empty += 1
        if self._consecutive_empty > 20:
            for name in AGENT_CONCURRENCY:
                self._agent_sems[name] = asyncio.Semaphore(
                    max(1, AGENT_CONCURRENCY.get(name, 3) // 2)
                )
            self._consecutive_empty = 0

    # ── Public enqueue ─────────────────────────────────────────────────────

    async def enqueue(self, entry: FrontierEntry) -> bool:
        return await self.frontier.push(entry)

    async def enqueue_many(self, entries: list[FrontierEntry]) -> int:
        return await self.frontier.push_many(entries)

    # ── Metrics ────────────────────────────────────────────────────────────

    async def get_metrics(self) -> SchedulerMetrics:
        async with self._metrics_lock:
            m = SchedulerMetrics(
                active_workers=self._metrics.active_workers,
                pending_work=self.frontier.pending,
                completed_work=self._metrics.completed_work,
                retried_work=self._metrics.retried_work,
                failed_work=self._metrics.failed_work,
                expired_work=self.frontier.failed,
                total_enqueued=self._metrics.total_enqueued,
                batches_executed=self._metrics.batches_executed,
                uptime_s=time.monotonic() - self._start_time,
                cost_consumed=self._metrics.cost_consumed,
            )
        return m

    async def metrics_table(self) -> Table:
        m = await self.get_metrics()
        t = Table(title="WorkScheduler")
        t.add_column("Metric", style="bold")
        t.add_column("Value")
        for label, val in [
            ("Active workers", str(m.active_workers)),
            ("Pending", str(m.pending_work)),
            ("Completed", str(m.completed_work)),
            ("Retried", str(m.retried_work)),
            ("Failed", str(m.failed_work)),
            ("Batches", str(m.batches_executed)),
            ("Total enqueued", str(m.total_enqueued)),
            ("Cost consumed", f"{m.cost_consumed:.1f}"),
            ("Uptime", f"{m.uptime_s:.0f}s"),
        ]:
            t.add_row(label, val)
        return t

    @property
    def is_running(self) -> bool:
        return self._running
