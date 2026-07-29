"""WorkScheduler — persistent event-driven execution engine with AdaptiveSemaphore,
true batch dispatch, graph expansion loop, and comprehensive observability.

Key properties:
  • AdaptiveSemaphore for mutable concurrency limits (no object recreation).
  • Batch-capable agents receive WorkBatch directly (shared HTTP, LLM, embeds).
  • GraphExpansionEngine auto-generates FrontierEntries from graph state.
  • Lease heartbeats for long-running tasks.
  • Latency, throughput, and retry-reason tracking in metrics.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from src.graph.entity import (
    AGENT_BATCHABLE,
    AGENT_CONCURRENCY,
    HEARTBEAT_INTERVAL,
    MUTATION_EXPANSION_RULES,
    AdaptiveSemaphore,
    BatchHandlerType,
    FrontierEntry,
    HandlerType,
    MutationEvent,
    NodeType,
    SchedulerMetrics,
    WorkState,
    make_work_id,
)
from src.graph.frontier import CrawlFrontier

console = Console()


class GraphExpansionEngine:
    """On every mutation, check expansion rules and enqueue work.
    Only the affected neighborhood is evaluated."""

    def __init__(self) -> None:
        self._seen_triggers: set[str] = set()

    async def on_mutation(
        self, event: MutationEvent, enqueue: Any, get_node: Any, _get_adjacency: Any = None
    ) -> int:
        if event.event_id in self._seen_triggers:
            return 0
        self._seen_triggers.add(event.event_id)
        generated = 0

        for rule in MUTATION_EXPANSION_RULES:
            if rule["change"] != event.change:
                continue
            if rule.get("node_type") and event.node_type != rule["node_type"]:
                continue
            if rule.get("edge_type") and event.edge_type != rule["edge_type"]:
                continue

            nid = event.mutated_id
            node = await get_node(nid)
            if node is None:
                continue

            if "check_field" in rule:
                field = rule["check_field"]
                if node.data.get(field):
                    continue

            wid = make_work_id(rule["agent"], nid, rule["depth"])
            if wid in self._seen_triggers:
                continue
            self._seen_triggers.add(wid)

            payload: dict[str, Any] = {"company": node.data.get("name", ""), "node_id": nid}
            if event.related_id and rule.get("edge_type"):
                payload["related_id"] = event.related_id

            await enqueue(
                FrontierEntry(
                    id=wid,
                    agent=rule["agent"],
                    node_id=nid,
                    node_type=event.node_type,
                    priority=rule["priority"],
                    depth=rule["depth"],
                    payload=payload,
                )
            )
            generated += 1

        return generated


class WorkScheduler:
    def __init__(self, frontier: CrawlFrontier, worker_count: int = 4) -> None:
        self.frontier = frontier
        self._base_workers = worker_count
        self._agents: dict[str, HandlerType] = {}
        self._batch_agents: dict[str, BatchHandlerType] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._shutdown = asyncio.Event()
        self._drain_mode = asyncio.Event()
        self._expansion = GraphExpansionEngine()

        # AdaptiveSemaphores — mutable limits, no object recreation
        self._agent_sems: dict[str, AdaptiveSemaphore] = {
            name: AdaptiveSemaphore(limit) for name, limit in AGENT_CONCURRENCY.items()
        }

        self._active_leases: dict[str, asyncio.Task[None]] = {}
        self._consecutive_failures = 0
        self._start_time = 0.0
        self._metrics = SchedulerMetrics()
        self._metrics_lock = asyncio.Lock()

        # Latency tracking
        self._latency_samples: dict[str, list[float]] = defaultdict(list)
        self._agent_fail_reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def register_agent(self, name: str, handler: HandlerType) -> None:
        self._agents[name] = handler
        if name not in self._agent_sems:
            self._agent_sems[name] = AdaptiveSemaphore(3)

    def register_batch_agent(self, name: str, handler: BatchHandlerType) -> None:
        self._batch_agents[name] = handler
        if name not in self._agent_sems:
            self._agent_sems[name] = AdaptiveSemaphore(3)

    async def expansion_cycle(self, graph_store) -> int:
        nodes = await graph_store.get_nodes_by_type(NodeType.COMPANY, limit=200)
        edges = await graph_store.get_all_edges(limit=500)
        node_map = {n.id: n for n in nodes}
        edge_dicts = [
            {"source": e.source_id, "target": e.target_id, "type": e.edge_type.value} for e in edges
        ]
        node_dicts = [{"id": n.id, "node_type": n.node_type.value, "data": n.data} for n in nodes]
        return await self._expansion.expand(node_dicts, edge_dicts, node_map, self.enqueue)

    # Lifecycle

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

    # Worker

    async def _worker_loop(self, worker_id: int) -> None:
        while self._running:
            try:
                entry = await asyncio.wait_for(self.frontier.lease(worker_id), timeout=3.0)
            except TimeoutError:
                if self._drain_mode.is_set() and self.frontier.pending == 0:
                    return
                self._consecutive_empty += 1
                await self._maybe_scale_down()
                continue

            if entry is None:
                if self._drain_mode.is_set():
                    return
                self._consecutive_empty += 1
                await self._maybe_scale_down()
                continue

            if entry.agent not in self._agents and entry.agent not in self._batch_agents:
                await self.frontier.complete(entry.id)
                continue

            self._consecutive_empty = 0
            is_batch = entry.agent in self._batch_agents
            handler = (
                self._batch_agents.get(entry.agent) if is_batch else self._agents.get(entry.agent)
            )
            sem = self._agent_sems.get(entry.agent, AdaptiveSemaphore(3))

            async with sem:
                t_start = time.monotonic()
                async with self._metrics_lock:
                    self._metrics.active_workers += 1

                entry.state = WorkState.RUNNING
                hb_task = asyncio.create_task(self._heartbeat_loop(entry.id))
                self._active_leases[entry.id] = hb_task

                try:
                    if is_batch and entry.agent in AGENT_BATCHABLE:
                        batch = await self.frontier.lease_batch(entry.agent, worker_id, max_batch=3)
                        if batch and len(batch.entries) > 1:
                            async with self._metrics_lock:
                                self._metrics.batches_executed += 1
                            new_entries = await handler(batch)
                        else:
                            new_entries = await self._agents.get(entry.agent, lambda x: [])(entry)
                    else:
                        new_entries = await handler(entry) if handler else []

                    elapsed = time.monotonic() - t_start
                    self._latency_samples[entry.agent].append(elapsed)
                    self._consecutive_failures = 0

                    async with self._metrics_lock:
                        self._metrics.completed_work += 1
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
                    reason = type(e).__name__
                    self._agent_fail_reasons[entry.agent][reason] += 1
                    async with self._metrics_lock:
                        self._metrics.retried_work += 1
                    await self.frontier.fail(entry.id, retry=entry.retries < entry.max_retries)
                finally:
                    hb_task.cancel()
                    self._active_leases.pop(entry.id, None)
                    async with self._metrics_lock:
                        self._metrics.active_workers -= 1

            await self._adapt_concurrency()

    async def _heartbeat_loop(self, entry_id: str) -> None:
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            if not await self.frontier.renew_lease(entry_id):
                return

    # Adaptive concurrency

    async def _adapt_concurrency(self) -> None:
        pending = self.frontier.pending
        if self._consecutive_failures >= 5:
            for sem in self._agent_sems.values():
                await sem.set_limit(max(1, sem.limit // 2))
            self._consecutive_failures = 0
        elif pending > 50:
            for sem in self._agent_sems.values():
                new_limit = min(sem.limit + 2, sem.limit * 2)
                await sem.set_limit(new_limit)

    async def _maybe_scale_down(self) -> None:
        if self._consecutive_empty <= 20:
            return
        for sem in self._agent_sems.values():
            await sem.set_limit(max(1, sem.limit // 2))
        self._consecutive_empty = 0

    # Enqueue

    async def enqueue(self, entry: FrontierEntry) -> bool:
        return await self.frontier.push(entry)

    async def enqueue_many(self, entries: list[FrontierEntry]) -> int:
        return await self.frontier.push_many(entries)

    # Metrics

    async def get_metrics(self) -> SchedulerMetrics:
        async with self._metrics_lock:
            latencies: dict[str, float] = {}
            for agent, samples in self._latency_samples.items():
                if samples:
                    latencies[agent] = sum(samples[-20:]) / len(samples[-20:])
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
            ("Active", str(m.active_workers)),
            ("Pending", str(m.pending_work)),
            ("Done", str(m.completed_work)),
            ("Retried", str(m.retried_work)),
            ("Failed", str(m.failed_work)),
            ("Batches", str(m.batches_executed)),
            ("Total", str(m.total_enqueued)),
            ("Cost", f"{m.cost_consumed:.1f}"),
            ("Uptime", f"{m.uptime_s:.0f}s"),
        ]:
            t.add_row(label, val)
        return t

    @property
    def is_running(self) -> bool:
        return self._running
