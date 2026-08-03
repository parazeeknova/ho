"""WorkScheduler - persistent event-driven execution engine with AdaptiveSemaphore,
true batch dispatch, graph expansion loop, and comprehensive observability.

Key properties:
  - AdaptiveSemaphore for mutable concurrency limits (no object recreation).
  - Batch-capable agents receive WorkBatch directly (shared HTTP, LLM, embeds).
  - InferenceDrivenExpander auto-generates FrontierEntries from graph state
    using capability graphs and information gain scoring.
  - Lease heartbeats for long-running tasks.
  - Latency, throughput, and retry-reason tracking in metrics.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table
from src.configuration import SchedulerConfig, get_config
from src.graph.entity import (
    AGENT_BATCHABLE,
    AGENT_CONCURRENCY,
    CAPABILITY_GRAPH,
    AdaptiveSemaphore,
    BatchHandlerType,
    EdgeType,
    FrontierEntry,
    HandlerType,
    MutationEvent,
    NodeType,
    SchedulerMetrics,
    WorkState,
    compute_uncertainty_score,
    make_work_id,
)
from src.graph.frontier import CrawlFrontier
from src.logging import get_logger

console = Console()
logger = get_logger("scheduler")


class InferenceDrivenExpander:
    """Declarative, inference-driven expansion engine.

    Replaces static mutation rules with:
    1. Capability graph: expected relationship patterns per node type.
    2. Information gain: compute uncertainty/completeness per node.
    3. Dynamic frontier: generate tasks based on expected information gain
       rather than hardcoded mutation triggers.

    On any mutation event, evaluates the affected node's capability graph
    to identify missing relationships and enqueues information-gathering
    tasks ordered by expected information gain.
    """

    _CAPABILITY_TO_AGENT: dict[EdgeType, str] = {
        EdgeType.FOUNDED_BY: "founder_miner",
        EdgeType.USES_ATS: "ats_crawler",
        EdgeType.HAS_FUNDING: "funding_agent",
        EdgeType.HAS_CAREER_SITE: "career_site_detector",
        EdgeType.POSTED_JOB: "ats_crawler",
        EdgeType.USES_TECH: "tech_stack_agent",
    }

    _CAPABILITY_PRIORITY: dict[EdgeType, int] = {
        EdgeType.FOUNDED_BY: 70,
        EdgeType.USES_ATS: 55,
        EdgeType.HAS_FUNDING: 50,
        EdgeType.HAS_CAREER_SITE: 60,
        EdgeType.POSTED_JOB: 45,
        EdgeType.USES_TECH: 35,
    }

    def __init__(self) -> None:
        self._seen_triggers: set[str] = set()
        self._expansion_count: int = 0

    async def on_mutation(
        self,
        event: MutationEvent,
        enqueue,
        get_node,
        get_adjacency=None,
    ) -> int:
        if event.event_id in self._seen_triggers:
            return 0
        self._seen_triggers.add(event.event_id)

        nid = event.mutated_id
        node = await get_node(nid)
        if node is None:
            return 0

        patterns = CAPABILITY_GRAPH.get(node.node_type, [])
        if not patterns:
            return 0

        adjacency: dict[str, Any] = {"edges_out": set(), "edges_in": set()}
        if get_adjacency is not None:
            adjacency = await get_adjacency(nid)

        uncertainty = compute_uncertainty_score(node, adjacency, CAPABILITY_GRAPH)

        generated = 0
        for pattern in sorted(
            patterns,
            key=lambda p: (1.0 if p.required else 0.0, p.min_count),
            reverse=True,
        ):
            edge_key = f"{pattern.edge_type.value}:"
            existing_matches = sum(
                1 for e in adjacency.get("edges_out", set()) if edge_key in e
            ) + sum(1 for e in adjacency.get("edges_in", set()) if edge_key in e)

            if existing_matches >= pattern.min_count:
                continue

            agent = self._CAPABILITY_TO_AGENT.get(pattern.edge_type)
            if agent is None:
                continue

            info_gain = uncertainty.total
            if info_gain < 0.4:
                continue

            priority = self._CAPABILITY_PRIORITY.get(
                pattern.edge_type,
                int(info_gain * 100),
            )
            priority = min(99, priority + int(info_gain * 40))

            wid = make_work_id(agent, nid, 1)
            if wid in self._seen_triggers:
                continue
            self._seen_triggers.add(wid)

            payload = {
                "company": node.data.get("name", ""),
                "node_id": nid,
                "missing_edge": pattern.edge_type.value,
                "info_gain": round(info_gain, 3),
                "required": pattern.required,
            }

            new_depth = 1
            if new_depth >= 2:
                continue

            await enqueue(
                FrontierEntry(
                    id=wid,
                    agent=agent,
                    node_id=nid,
                    node_type=node.node_type,
                    priority=priority,
                    depth=new_depth,
                    payload=payload,
                )
            )
            generated += 1
            self._expansion_count += 1

        return generated

    async def evaluate_node(
        self,
        node_id: str,
        get_node,
        get_adjacency,
        enqueue,
    ) -> int:
        """Proactively evaluate a node and generate expansion tasks."""
        node = await get_node(node_id)
        if node is None:
            return 0

        patterns = CAPABILITY_GRAPH.get(node.node_type, [])
        if not patterns:
            return 0

        adjacency: dict[str, Any] = {"edges_out": set(), "edges_in": set()}
        if get_adjacency:
            adjacency = await get_adjacency(node_id)

        uncertainty = compute_uncertainty_score(node, adjacency)
        generated = 0

        for pattern in patterns:
            edge_key = f"{pattern.edge_type.value}:"
            existing = sum(1 for e in adjacency.get("edges_out", set()) if edge_key in e) + sum(
                1 for e in adjacency.get("edges_in", set()) if edge_key in e
            )
            if existing >= pattern.min_count:
                continue

            agent = self._CAPABILITY_TO_AGENT.get(pattern.edge_type)
            if agent is None:
                continue

            if uncertainty.total < 0.4:
                continue

            priority = self._CAPABILITY_PRIORITY.get(
                pattern.edge_type,
                int(uncertainty.total * 100),
            )
            wid = make_work_id(f"proactive:{agent}", node_id, 1)
            if wid in self._seen_triggers:
                continue
            self._seen_triggers.add(wid)

            new_depth = 1
            if new_depth >= 2:
                continue

            await enqueue(
                FrontierEntry(
                    id=wid,
                    agent=agent,
                    node_id=node_id,
                    node_type=node.node_type,
                    priority=min(99, priority + int(uncertainty.total * 40)),
                    depth=new_depth,
                    payload={
                        "company": node.data.get("name", ""),
                        "node_id": node_id,
                        "missing_edge": pattern.edge_type.value,
                        "info_gain": round(uncertainty.total, 3),
                        "required": pattern.required,
                    },
                )
            )
            generated += 1

        return generated

    @property
    def expansion_count(self) -> int:
        return self._expansion_count


class WorkScheduler:
    def __init__(
        self,
        frontier: CrawlFrontier,
        worker_count: int | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        cfg = config or get_config().scheduler
        self.frontier = frontier
        self._base_workers = worker_count if worker_count is not None else cfg.worker_count
        self._agents: dict[str, HandlerType] = {}
        self._batch_agents: dict[str, BatchHandlerType] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._shutdown = asyncio.Event()
        self._drain_mode = asyncio.Event()
        self._expansion = InferenceDrivenExpander()

        self._agent_sems: dict[str, AdaptiveSemaphore] = {
            name: AdaptiveSemaphore(limit) for name, limit in AGENT_CONCURRENCY.items()
        }

        self._active_leases: dict[str, asyncio.Task[None]] = {}
        self._consecutive_failures = 0
        self._consecutive_empty = 0
        self._start_time = 0.0
        self._metrics = SchedulerMetrics()
        self._metrics_lock = asyncio.Lock()

        self._latency_samples: dict[str, list[float]] = defaultdict(list)
        self._agent_fail_reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._consecutive_failure_threshold = cfg.consecutive_failure_threshold
        self._consecutive_empty_threshold = cfg.consecutive_empty_threshold
        self._batch_max = cfg.batch_max
        self.config = cfg
        self.agent_caps = dict(AGENT_CONCURRENCY)

    def register_agent(self, name: str, handler: HandlerType) -> None:
        self._agents[name] = handler
        if name not in self._agent_sems:
            self._agent_sems[name] = AdaptiveSemaphore(3)

    def register_batch_agent(self, name: str, handler: BatchHandlerType) -> None:
        self._batch_agents[name] = handler
        if name not in self._agent_sems:
            self._agent_sems[name] = AdaptiveSemaphore(3)

    async def expansion_cycle(self, graph_store) -> int:
        """Proactively evaluate company nodes for missing capability edges.

        Scans all company nodes, computes uncertainty scores, and
        enqueues information-gathering tasks for the highest-priority gaps.
        Returns the total number of expansion tasks generated.
        """
        nodes = await graph_store.get_nodes_by_type(NodeType.COMPANY, limit=200)
        total_generated = 0
        for node in nodes:
            generated = await self._expansion.evaluate_node(
                node.id,
                graph_store.get_node,
                lambda nid: graph_store.get_node_adjacency(nid),
                self.enqueue,
            )
            total_generated += generated
        return total_generated

    # Lifecycle

    def start(self, worker_count: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._shutdown.clear()
        self._drain_mode.clear()
        self._start_time = time.monotonic()
        wc = worker_count or self._base_workers
        for i in range(wc):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))
        logger.info(f"Scheduler started {wc} workers")
        console.print(f"  ⚙️  [Scheduler] {wc} workers")

    async def shutdown(self, drain: bool = True) -> None:
        if not self._running:
            return
        logger.info("Scheduler shutting down")
        console.print("  ⚙️  [Scheduler] Shutting down...")
        self._running = False
        if drain:
            self._drain_mode.set()
            while self.frontier.pending > 0 or len(self._active_leases) > 0:
                await asyncio.sleep(1.0)
        self._shutdown.set()
        for task in self._active_leases.values():
            task.cancel()
        self._active_leases.clear()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await self.frontier.persist_all()
        logger.info("Scheduler shutdown complete")
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
            except asyncio.CancelledError:
                return

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
            batch_handler = self._batch_agents.get(entry.agent)
            handler = self._agents.get(entry.agent)
            is_batch = batch_handler is not None
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
                        batch = await self.frontier.lease_batch(
                            entry.agent, worker_id, max_batch=self._batch_max
                        )
                        if batch and len(batch.entries) > 1:
                            async with self._metrics_lock:
                                self._metrics.batches_executed += 1
                            assert batch_handler is not None
                            new_entries = await batch_handler(batch)
                        else:
                            single_handler = self._agents.get(entry.agent)
                            new_entries = await single_handler(entry) if single_handler else []
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

                    logger.info(
                        f"Worker {worker_id} completed {entry.agent}",
                        worker=str(worker_id),
                        entity=entry.node_id,
                        latency=elapsed,
                    )

                except asyncio.CancelledError:
                    logger.warning(f"Worker {worker_id} cancelled mid-task", worker=str(worker_id))
                    await self.frontier.fail(entry.id, retry=entry.retries < entry.max_retries)
                    raise
                except Exception as e:
                    self._consecutive_failures += 1
                    reason = type(e).__name__
                    self._agent_fail_reasons[entry.agent][reason] += 1
                    async with self._metrics_lock:
                        self._metrics.retried_work += 1
                    logger.exception(
                        f"Worker {worker_id} failed {entry.agent}",
                        exc=e,
                        worker=str(worker_id),
                        entity=entry.node_id,
                    )
                    await self.frontier.fail(entry.id, retry=entry.retries < entry.max_retries)
                finally:
                    try:
                        hb_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await hb_task
                    except Exception:
                        pass
                    self._active_leases.pop(entry.id, None)
                    async with self._metrics_lock:
                        self._metrics.active_workers -= 1

            await self._adapt_concurrency()

    async def _heartbeat_loop(self, entry_id: str) -> None:
        cfg = get_config().scheduler
        while self._running:
            await asyncio.sleep(cfg.heartbeat_interval)
            if not await self.frontier.renew_lease(entry_id):
                return

    # Adaptive concurrency

    async def _adapt_concurrency(self) -> None:
        if self._consecutive_failures >= self._consecutive_failure_threshold:
            for sem in self._agent_sems.values():
                await sem.set_limit(max(1, sem.limit // 2))
            self._consecutive_failures = 0
            logger.warning("Adaptive concurrency: halved limits due to consecutive failures")
        else:
            for agent_id, sem in self._agent_sems.items():
                target = self.agent_caps.get(agent_id, max(self.agent_caps.values()) or 1)
                if sem.limit < target:
                    await sem.set_limit(min(target, sem.limit + 1))

    async def _maybe_scale_down(self) -> None:
        if self._consecutive_empty <= self._consecutive_empty_threshold:
            return
        for sem in self._agent_sems.values():
            await sem.set_limit(max(1, sem.limit // 2))
        self._consecutive_empty = 0
        logger.debug("Adaptive concurrency: scaled down")

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

    async def process_mutation(self, event: MutationEvent, graph_store) -> int:
        """Process a mutation event through the expansion engine."""
        return await self._expansion.on_mutation(
            event,
            self.enqueue,
            graph_store.get_node,
        )
