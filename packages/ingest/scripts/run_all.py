#!/usr/bin/env python3
"""One-command full-stack runner: infra + embed + radar + bridge + autofill.

Everything ho offers, from a single invocation:

  1. docker compose up the ingest stack (searxng, neo4j, agent-memory-db) and
     wait for health;
  2. start the local embedding server (:8900);
  3. if persona.json is missing, seed memory (resume + persona) non-interactively;
  4. run the end-to-end loop: radar pipeline (master + workers) discovers and
     LLM-matches jobs, the bridge drains accepted roles into the autofill
     queue, and the autofill worker (1 concurrent browser) auto-applies.
     Crashed children are restarted; the run continues overnight until stopped.

Local-only by default: company discovery uses the local adapters (yc, dealroom,
hn, remoteok, ...), not the Azure relic. Set AZURE=1 to re-enable relic discovery.

Page rendering is in-process: static pages are fetched with httpx + markitdown,
and JS-only pages are rendered on demand with a lazily-spawned Playwright
browser (torn down after each use) — no Firecrawl api/playwright-service/queue
services, no resident browser.

Usage:
    uv run python scripts/run_all.py            # full stack, foreground
    uv run python scripts/run_all.py --dry-run  # check infra only, no pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent  # packages/ingest
REPO = PROJECT.parent.parent  # repo root
COMPOSE = PROJECT / "docker-compose.yaml"

DOCKER_SERVICES = [
    "searxng",
    "neo4j",
    "agent-memory-db",
    "redis",
    # Steel browser backend: only needed when STEEL_BASE_URL is set, but the
    # compose `up -d` below is what pulls the image on first run, so it is
    # always part of the infra bring-up (harmless when unused).
    "steel",
]

# Infra readiness probes (host, port).
HOST_PROBES = {
    "searxng": (8080, 15),
    "neo4j": (7687, 15),
    "agent-memory-db": (5433, 20),
    "redis": (6379, 15),
    "steel": (3000, 60),
    "embed": (8900, 60),
}


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "ho/run_all"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


async def _wait_for(name: str, timeout: float) -> bool:
    print(f"[ho] waiting for {name}...", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if name == "searxng" and _http_ok("http://localhost:8080/"):
            return True
        if name == "embed" and _http_ok("http://localhost:8900/health"):
            return True
        probe = HOST_PROBES.get(name)
        if probe and _port_open("localhost", probe[0]):
            return True
        await asyncio.sleep(1.0)
    return False


def _docker(args: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE), *args],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT),
        )
        return r.returncode, (r.stdout or "")[-2000:] + (r.stderr or "")[-2000:]
    except Exception as e:
        return -1, str(e)


def _compose_up() -> bool:
    print("[ho] starting docker services...", flush=True)
    code, out = _docker(["up", "-d", *DOCKER_SERVICES])
    if code != 0:
        print(f"[ho] docker compose up failed: {out}", flush=True)
        return False
    return True


def _preflight_backup() -> None:
    """Checkpoint volumes before the sweep starts (fire-and-forget, 90s cap)."""
    import subprocess as _sp

    try:
        print("[ho] pre-run backup (checkpoint)...", flush=True)
        r = _sp.run(
            ["uv", "run", "python", "scripts/backup/auto_backup.py"],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=90,
        )
        print(f"[ho] backup: {r.stdout.strip()[-200:] if r.stdout else 'ok'}", flush=True)
    except Exception as e:
        print(f"[ho] backup skipped: {e}", flush=True)


async def _ensure_infra() -> bool:
    if not _compose_up():
        return False
    ok = True
    for name in ("agent-memory-db", "neo4j", "searxng", "redis", "steel", "embed"):
        # embed is started below; if already up, great.
        if name == "embed" and not _http_ok("http://localhost:8900/health"):
            continue
        if not await _wait_for(name, HOST_PROBES.get(name, (0, 15))[1]):
            print(f"[ho] WARNING: {name} not ready", flush=True)
            # Steel is optional infra: the runner falls back to a direct
            # browser launch when STEEL_BASE_URL is unreachable, so a Steel
            # that never comes up must warn, not abort the pipeline. All the
            # other services (pg/neo4j/searxng/redis) are required.
            if name != "steel":
                ok = False
        else:
            print(f"[ho] ✓ {name} ready", flush=True)
    # Pre-run backup once infra is confirmed
    _preflight_backup()
    return ok


def _ensure_embed_server() -> None:
    if _http_ok("http://localhost:8900/health"):
        print("[ho] embedding server already up", flush=True)
        return
    print("[ho] starting embedding server...", flush=True)
    log = REPO / "logs"
    log.mkdir(exist_ok=True)
    with (log / "embed_server.log").open("ab") as out:
        subprocess.Popen(
            [sys.executable, str(PROJECT / "scripts" / "serve.py")],
            cwd=str(PROJECT),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def _memory_status() -> str:
    """Human-readable memory status: persona present, resume indexed, summary
    freshness vs the current resume source. Returns a status string."""
    import json as _json

    persona = REPO / "data" / "persona.json"
    if not persona.exists():
        return "NO persona.json — run `bun run init-memory` first"
    try:
        data = _json.loads(persona.read_text())
    except Exception:
        return "persona.json unreadable"
    answers = len(data.get("answers", []))
    summary = str(data.get("resume_summary") or "")
    # Freshness probe: if the summary lacks the projects this repo's resume.tex
    # carries, it was built before the latest resume and the matcher grounding
    # is stale. (Only advisory — the resume may legitimately differ.)
    fresh = True
    for marker in ("Asocialmedia", "Verso", "Chorus", "Lumen", "asocialmedia"):
        if marker in summary:
            break
    else:
        if summary and "Singularity" not in summary:
            fresh = False
    parts = [f"persona: {answers} answers"]
    if summary:
        parts.append(f"resume_summary: {len(summary)} chars")
    else:
        parts.append("resume_summary: MISSING")
    if not fresh:
        parts.append("⚠ resume_summary may be STALE (missing latest projects)")
    return "; ".join(parts)


def _persona_missing(persona_path: Path) -> list[str]:
    """Critical persona items that are missing/blank: core answer categories
    and identity contact fields. Empty list = complete."""
    import json as _json

    try:
        data = _json.loads(persona_path.read_text())
    except Exception:
        return ["persona.json unreadable"]
    missing: list[str] = []
    answered = {(a.get("category") or "").strip() for a in data.get("answers", [])}
    core = {
        "current_location",
        "work_model",
        "relocation",
        "nationality",
        "work_authorization",
        "visa_sponsorship",
        "expected_compensation",
        "education",
    }
    for cat in core:
        if cat not in answered:
            missing.append(f"answer: {cat}")
    identity = data.get("identity") or {}
    for field in ("firstName", "lastName", "email", "phone", "linkedin", "github"):
        if not (identity.get(field) or "").strip():
            missing.append(f"identity: {field}")
    return missing


def _ensure_memory() -> None:
    persona = REPO / "data" / "persona.json"
    if not persona.exists():
        print("[ho] persona.json missing — seeding memory non-interactively...", flush=True)
        try:
            env = dict(os.environ)
            env["NON_INTERACTIVE"] = "1"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO / "packages" / "autofill" / "scripts" / "init_memory.py"),
                ],
                cwd=str(REPO),
                env=env,
                timeout=600,
            )
        except Exception as e:
            print(f"[ho] memory seed skipped ({e}); pipeline will use defaults", flush=True)
        return
    # Present but possibly stale: report it so the user knows to re-run
    # init-memory after a resume update instead of silently matching on old
    # grounding.
    print(f"[ho] memory: {_memory_status()}", flush=True)
    # Completeness: if critical persona items are missing, ask the user to run
    # the interactive grill so the matcher never scores against a half-built
    # persona.
    missing = _persona_missing(persona)
    if missing:
        print(
            f"[ho] persona is INCOMPLETE — missing: {', '.join(missing[:6])}",
            flush=True,
        )
        try:
            import sys as _sys

            if _sys.stdin and _sys.stdin.isatty():
                answer = (
                    input("[ho] Run the interactive persona wizard now? [Y/n] ").strip().lower()
                )
                if answer in ("", "y", "yes"):
                    print("[ho] Launching persona wizard...", flush=True)
                    subprocess.run(
                        [
                            sys.executable,
                            str(REPO / "packages" / "autofill" / "scripts" / "grill_persona.py"),
                        ],
                        cwd=str(REPO),
                        timeout=1800,
                    )
        except Exception as e:
            print(f"[ho] persona wizard prompt skipped ({e})", flush=True)


async def _run_loop(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    paths = [str(PROJECT), str(REPO / "packages")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("OVERNIGHT_LOOP", "true")
    # Force local discovery: the user runs everything locally, no Azure relic.
    # Explicitly override any inherited DISCOVERY_SOURCE (e.g. none from an
    # earlier shell export) so the local adapters always run.
    env["DISCOVERY_SOURCE"] = "all"
    env.setdefault("AUTOFILL_MAX_CONCURRENT", "1")

    cmd = [
        sys.executable,
        str(PROJECT / "scripts" / "loop.py"),
        "--radar-workers",
        str(args.radar_workers),
        "--bridge-interval",
        str(args.bridge_interval),
        "--bridge-batch",
        str(args.bridge_batch),
    ]
    if args.no_fill:
        cmd.append("--no-fill")
    if args.max_minutes:
        cmd += ["--max-minutes", str(args.max_minutes)]
    print(f"[ho] launching loop: {' '.join(cmd)}", flush=True)
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    # Stream the loop's output to BOTH the console (so the user sees live
    # sweep/status progress) AND run_all.log (for tailing). A bare pipe to
    # stdout would deadlock if un-drained, so we drain it here.
    log_path = log_dir / "run_all.log"
    run_log_path = log_dir / "run.log"
    loop_log_path = log_dir / "loop.log"
    with (
        log_path.open("ab") as out1,
        run_log_path.open("ab") as out2,
        loop_log_path.open("ab") as out3,
    ):
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def _tee() -> None:
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.readline()
                if not chunk:
                    break
                line = chunk.decode(errors="replace")
                out1.write(chunk)
                out1.flush()
                out2.write(chunk)
                out2.flush()
                out3.write(chunk)
                out3.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        tee_task = asyncio.create_task(_tee())
        rc = await proc.wait()
        await tee_task
        return rc


def _handle_sig(signum: int, frame) -> None:  # noqa: ANN001
    import contextlib

    with contextlib.suppress(Exception):
        print(f"\n[ho] signal {signum}; shutting down...", flush=True)
    raise KeyboardInterrupt


class _LogTee:
    """Tee output to both console stream and file."""

    def __init__(self, original_stream: object, log_file: object) -> None:
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data: str) -> None:
        getattr(self.original_stream, "write", lambda s: None)(data)
        getattr(self.original_stream, "flush", lambda: None)()
        with contextlib.suppress(Exception):
            getattr(self.log_file, "write", lambda s: None)(data)
            getattr(self.log_file, "flush", lambda: None)()

    def flush(self) -> None:
        getattr(self.original_stream, "flush", lambda: None)()
        with contextlib.suppress(Exception):
            getattr(self.log_file, "flush", lambda: None)()


def _status_report(watch: bool = False) -> int:
    """`bun run run --status`: a full read-only snapshot of every counter the
    pipeline tracks — DB, graph, RAG, ML, autofill, epochs, sources, and
    whether anything is currently running. No side effects: does not touch
    containers, the lock, or the embed server."""
    import asyncio
    import os as _os
    from typing import Any

    # Quiet the pipeline loggers (MemoryStore/AutofillDB emit INFO on open/close).
    _os.environ["LOG_LEVEL"] = "WARNING"

    async def _pg_section() -> dict[str, Any]:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            async with store._pool.acquire() as c:
                obs = await c.fetchval("SELECT COUNT(*) FROM job_observations")
                cand = await c.fetchval("SELECT COUNT(*) FROM radar_candidates")
                sources = await c.fetchval("SELECT COUNT(*) FROM source_checkpoints")
                sources_active = await c.fetchval(
                    "SELECT COUNT(*) FROM source_checkpoints WHERE active"
                )
                sources_polled = await c.fetchval(
                    "SELECT COUNT(*) FROM source_checkpoints WHERE last_polled IS NOT NULL"
                )
                events = await c.fetchval("SELECT COUNT(*) FROM decision_events")
                rewards = await c.fetchval(
                    "SELECT COUNT(*) FROM decision_events WHERE reward IS NOT NULL"
                )
                imp = await c.fetchval("SELECT COUNT(*) FROM impressions")
                outcomes = await c.fetchval("SELECT COUNT(*) FROM unattributed_outcomes")
                frontier = await c.fetchval(
                    "SELECT COUNT(*) FROM job_observations WHERE last_seen > "
                    "(extract(epoch from now()) - 86400)"
                )
                discovered = await c.fetchval(
                    "SELECT COUNT(*) FROM job_observations WHERE source LIKE '%searxng%' "
                    "OR source LIKE '%github%' OR source LIKE '%discovered%'"
                )
                model = await c.fetchrow(
                    "SELECT version, status FROM model_registry "
                    "WHERE status='active' ORDER BY created_at DESC LIMIT 1"
                )
                gmail = await c.fetchval(
                    "SELECT 1 FROM gmail_push_state WHERE history_id != '' LIMIT 1"
                )
                return {
                    "obs": obs,
                    "cand": cand,
                    "sources": sources,
                    "sources_active": sources_active,
                    "sources_polled": sources_polled,
                    "events": events,
                    "rewards": rewards,
                    "impressions": imp,
                    "outcomes": outcomes,
                    "frontier": frontier,
                    "discovered": discovered,
                    "model": f"{model['version']}({model['status']})"
                    if model
                    else "rule_v1 (heuristic baseline)",
                    "gmail": "ON" if gmail else "off",
                }
        finally:
            await store.close()

    async def _eligibility_section() -> dict[str, int]:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            async with store._pool.acquire() as c:
                rows = await c.fetch(
                    "SELECT eligibility, COUNT(*) AS n FROM radar_candidates GROUP BY eligibility"
                )
                return {str(r["eligibility"]): r["n"] for r in rows}
        finally:
            await store.close()

    async def _queue_section() -> dict[str, int]:
        from autofill.src.core.db import AutofillDB

        db = await AutofillDB.create()
        try:
            async with db._pool.acquire() as c:
                rows = await c.fetch(
                    "SELECT status, COUNT(*) AS n FROM autofill_queue GROUP BY status"
                )
                fills = await c.fetchval("SELECT COUNT(*) FROM autofill_fills")
            d = {str(r["status"]): r["n"] for r in rows}
            d["fills"] = fills
            return d
        finally:
            await db.close()

    async def _epoch_section() -> dict[str, Any]:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            async with store._pool.acquire() as c:
                active = await c.fetchrow(
                    "SELECT epoch_id, started_at, target_submissions "
                    "FROM learning_epochs "
                    "WHERE status='active' ORDER BY started_at DESC LIMIT 1"
                )
                total = await c.fetchval("SELECT COUNT(*) FROM learning_epochs")
                done = await c.fetchval(
                    "SELECT COUNT(*) FROM learning_epochs WHERE status IN "
                    "('completed','target_reached')"
                )
                a = dict(active) if active else None
                if a:
                    a["completed_submissions"] = await c.fetchval(
                        "SELECT COUNT(*) FROM autofill_queue "
                        "WHERE epoch_id = $1 AND applied_at IS NOT NULL",
                        a["epoch_id"],
                    )
            return {"active": a, "total": total, "done": done}
        finally:
            await store.close()

    async def _rag_section() -> dict[str, int]:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            resume = await store.chunk_count()
            persona = await store.persona_chunk_count()
            async with store._pool.acquire() as c:
                obs_emb = await c.fetchval("SELECT COUNT(*) FROM obs_embeddings")
                embed_cache = await c.fetchval("SELECT COUNT(*) FROM embed_cache")
                render_cache = await c.fetchval("SELECT COUNT(*) FROM render_cache")
            return {
                "resume_chunks": resume,
                "persona_chunks": persona,
                "obs_embeddings": obs_emb,
                "embed_cache": embed_cache,
                "render_cache": render_cache,
            }
        finally:
            await store.close()

    async def _graph_section() -> str:
        try:
            from src.graph.graph_store import GraphStore

            g = await GraphStore.create()
            try:
                nodes = await g.node_count()
                rels = await g.relationship_count()
            finally:
                await g.close()
            return f"nodes={nodes} rels={rels}"
        except Exception:
            return "graph unavailable"

    async def _top_sources() -> list[tuple[str, int]]:
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        try:
            async with store._pool.acquire() as c:
                rows = await c.fetch(
                    "SELECT source, COUNT(*) AS n FROM radar_candidates "
                    "GROUP BY source ORDER BY n DESC LIMIT 12"
                )
                return [(str(r["source"]), r["n"]) for r in rows]
        finally:
            await store.close()

    async def _rates_section() -> dict[str, Any]:
        from autofill.src.core.db import AutofillDB
        from src.memory.pgvector_store import MemoryStore

        store = await MemoryStore.create()
        db = await AutofillDB.create()
        try:
            async with store._pool.acquire() as c:
                obs_15m = await c.fetchval(
                    "SELECT COUNT(*) FROM job_observations WHERE last_seen > "
                    "(extract(epoch from now()) - 900)"
                )
                cand_15m = await c.fetchval(
                    "SELECT COUNT(*) FROM radar_candidates WHERE updated_at > "
                    "(now() - interval '15 minutes')"
                )
            async with db._pool.acquire() as c:
                fills_15m = await c.fetchval(
                    "SELECT COUNT(*) FROM autofill_fills WHERE created_at > "
                    "(now() - interval '15 minutes')"
                )
                sub_60m = await c.fetchval(
                    "SELECT COUNT(*) FROM autofill_queue WHERE applied_at > "
                    "(now() - interval '60 minutes')"
                )
            return {
                "obs_rate": round((obs_15m or 0) / 15.0, 1),
                "obs_total_15m": obs_15m or 0,
                "cand_rate": round((cand_15m or 0) / 15.0, 1),
                "cand_total_15m": cand_15m or 0,
                "fill_rate": round((fills_15m or 0) / 15.0, 1),
                "fill_total_15m": fills_15m or 0,
                "sub_rate": round((sub_60m or 0) / 60.0, 2),
                "sub_total_60m": sub_60m or 0,
            }
        finally:
            await store.close()
            await db.close()

    async def _collect_all_stats() -> dict[str, Any]:
        rates, pg, elig, queue, epoch, rag, graph, top = await asyncio.gather(
            _rates_section(),
            _pg_section(),
            _eligibility_section(),
            _queue_section(),
            _epoch_section(),
            _rag_section(),
            _graph_section(),
            _top_sources(),
        )
        running = _pipeline_running()
        return {
            "rates": rates,
            "pg": pg,
            "elig": elig,
            "queue": queue,
            "epoch": epoch,
            "rag": rag,
            "graph": graph,
            "top": top,
            "running": running,
        }

    def _render(stats: dict[str, Any]) -> Any:
        from rich.console import Group
        from rich.table import Table

        running = stats["running"]
        pid_str = f"RUNNING (PID {running})" if running else "STOPPED"
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # Header
        t_hdr = Table(box=None, show_header=False, expand=True, padding=(0, 0))
        t_hdr.add_column("title", style="bold italic cyan", justify="left")
        t_hdr.add_column("time", style="dim", justify="right")
        t_hdr.add_row(f"HO AGENT PIPELINE STATUS  ·  {pid_str}", now_str)

        # 1. Rates Table
        rates = stats["rates"]
        t_rates = Table(
            title="LIVE PROCESS RATES (/min)",
            title_style="bold italic cyan",
            title_justify="left",
            box=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 2),
        )
        t_rates.add_column("Process Pipeline Stage", style="cyan", justify="left")
        t_rates.add_column("Current Rate", style="bold white", justify="right")
        t_rates.add_column("Window Total", style="dim", justify="right")
        t_rates.add_row(
            "Job Discovery (obs/min)",
            f"{rates['obs_rate']} / min",
            f"{rates['obs_total_15m']:,} obs (15m)",
        )
        t_rates.add_row(
            "Candidate Gating (cand/min)",
            f"{rates['cand_rate']} / min",
            f"{rates['cand_total_15m']:,} cand (15m)",
        )
        t_rates.add_row(
            "Autofill Form Filling (fills/min)",
            f"{rates['fill_rate']} / min",
            f"{rates['fill_total_15m']:,} fills (15m)",
        )
        t_rates.add_row(
            "Application Submissions (sub/min)",
            f"{rates['sub_rate']} / min",
            f"{rates['sub_total_60m']:,} sub (60m)",
        )

        # 2. Postgres DB & Ingestion
        pg = stats["pg"]
        t_ingest = Table(
            title="POSTGRES DB AND DISCOVERY",
            title_style="bold italic cyan",
            title_justify="left",
            box=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 2),
        )
        t_ingest.add_column("Metric", style="cyan", justify="left", no_wrap=True)
        t_ingest.add_column("Count", style="bold white", justify="right")
        t_ingest.add_column("Details / Status", style="dim", justify="right")
        t_ingest.add_row("Raw Observations (obs)", f"{pg.get('obs', 0):,}", "Ingested raw job URLs")
        t_ingest.add_row(
            "Canonical Candidates (cand)", f"{pg.get('cand', 0):,}", "Normalized postings"
        )
        t_ingest.add_row(
            "Active Company Sources",
            f"{pg.get('sources_active', 0):,}",
            f"Polled: {pg.get('sources_polled', 0):,}",
        )
        t_ingest.add_row("24h Queue Frontier", f"{pg.get('frontier', 0):,}", "Observed in last 24h")
        t_ingest.add_row(
            "Dynamic Discovered", f"{pg.get('discovered', 0):,}", "SearXNG / GitHub / YC feeds"
        )
        t_ingest.add_row(
            "Decision Events / Impressions",
            f"{pg.get('events', 0):,} / {pg.get('impressions', 0):,}",
            f"Rewards: {pg.get('rewards', 0):,}",
        )
        t_ingest.add_row(
            "Active Model Ranker", str(pg.get("model", "none")), "Candidate scoring model"
        )
        t_ingest.add_row("Gmail Outcome Push", str(pg.get("gmail", "off")), "Outcome push listener")

        # 3. Eligibility Breakdown
        elig = stats["elig"]
        total_elig = sum(elig.values()) or 1
        acc = elig.get("accepted", 0)
        nm = elig.get("near_miss", 0)
        rej = elig.get("rejected", 0)
        err = elig.get("error", 0)

        t_elig = Table(
            title="CANDIDATE ELIGIBILITY BREAKDOWN",
            title_style="bold italic cyan",
            title_justify="left",
            box=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 2),
        )
        t_elig.add_column("Eligibility Filter State", style="cyan", justify="left", no_wrap=True)
        t_elig.add_column("Candidate Count", style="bold white", justify="right")
        t_elig.add_column("Share Percent", style="bold", justify="right")
        t_elig.add_row("Accepted (High Fit)", f"{acc:,}", f"{acc / total_elig * 100:.1f}%")
        t_elig.add_row("Near Miss", f"{nm:,}", f"{nm / total_elig * 100:.1f}%")
        t_elig.add_row("Rejected", f"{rej:,}", f"{rej / total_elig * 100:.1f}%")
        if err:
            t_elig.add_row("Error", f"{err:,}", f"{err / total_elig * 100:.1f}%")

        # 4. Autofill Queue & Learning Epoch
        q = stats["queue"]
        ep = stats["epoch"]
        t_queue = Table(
            title="AUTOFILL WORKER QUEUE AND LEARNING EPOCHS",
            title_style="bold italic cyan",
            title_justify="left",
            box=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 2),
        )
        t_queue.add_column("Queue / Epoch Metric", style="cyan", justify="left", no_wrap=True)
        t_queue.add_column("Value / Status", style="bold white", justify="right")
        t_queue.add_row("Total Form Fills Executed", f"{q.get('fills', 0):,}")
        t_queue.add_row("Pending Queue Jobs", f"{q.get('pending', 0):,}")
        t_queue.add_row("Currently Filling Jobs", f"{q.get('filling', 0):,}")
        t_queue.add_row("Confirmed Applications Submitted", f"{q.get('submitted', 0):,}")
        t_queue.add_row("Awaiting Review", f"{q.get('awaiting_review', 0):,}")
        t_queue.add_row(
            "Failed / Deferred Jobs", f"{q.get('failed', 0):,} / {q.get('deferred', 0):,}"
        )
        t_queue.add_row("Skipped Jobs", f"{q.get('skipped', 0):,}")

        if ep["active"]:
            a = ep["active"]
            tgt = a.get("target_submissions") or 0
            tgt_str = (
                f"{a['completed_submissions']}/{tgt}"
                if tgt > 0
                else f"{a['completed_submissions']} (no target cap)"
            )
            t_queue.add_row("Active Learning Epoch ID", f"{a['epoch_id']} ({tgt_str})")
        else:
            t_queue.add_row("Active Learning Epoch ID", "None")
        t_queue.add_row("Learning Epochs (Total / Completed)", f"{ep['total']} / {ep['done']}")

        # 5. RAG Memory & Graph
        rag = stats["rag"]
        graph = stats["graph"]
        t_rag = Table(
            title="RAG VECTOR MEMORY AND GRAPH DATABASE",
            title_style="bold italic cyan",
            title_justify="left",
            box=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 2),
        )
        t_rag.add_column("Subsystem Metric", style="cyan", justify="left", no_wrap=True)
        t_rag.add_column("Count / Details", style="bold white", justify="right")
        t_rag.add_row("Resume Vector Chunks", f"{rag.get('resume_chunks', 0):,}")
        t_rag.add_row("Persona Vector Chunks", f"{rag.get('persona_chunks', 0):,}")
        t_rag.add_row("Embed Cache Entries", f"{rag.get('embed_cache', 0):,}")
        t_rag.add_row("Page Render Cache Entries", f"{rag.get('render_cache', 0):,}")
        t_rag.add_row("Neo4j Graph Database Counts", graph)

        # 6. Top Sources Table
        t_top = Table(
            title="TOP ATS DISCOVERY SOURCES",
            title_style="bold italic cyan",
            title_justify="left",
            box=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            padding=(0, 2),
        )
        t_top.add_column(
            "Source Platform and Board Slug", style="cyan", justify="left", no_wrap=True
        )
        t_top.add_column("Gated Candidates", style="bold white", justify="right")
        for src, n in stats["top"][:10]:
            t_top.add_row(src, f"{n:,}")

        return Group(t_hdr, t_rates, t_ingest, t_elig, t_queue, t_rag, t_top)

    async def _live_loop() -> int:
        from rich.console import Console
        from rich.live import Live

        console = Console()
        if watch:
            console.print(
                "[bold dim]Entering live status dashboard (Press Ctrl+C to exit)...[/bold dim]\n"
            )
            try:
                with Live(console=console, refresh_per_second=1, screen=False) as live:
                    while True:
                        stats = await _collect_all_stats()
                        live.update(_render(stats))
                        await asyncio.sleep(1.5)
            except KeyboardInterrupt, asyncio.CancelledError:
                console.print("\n[dim]Status monitor exited.[/dim]")
                return 0
        else:
            stats = await _collect_all_stats()
            console.print(_render(stats))
            return 0

    return asyncio.run(_live_loop())


def _pipeline_running() -> int | None:
    """Return the loop.py pid if the pipeline is running, else None."""
    import subprocess as _sp

    try:
        r = _sp.run(
            ["pgrep", "-f", "scripts/loop.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p) for p in (r.stdout or "").split() if p.isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


async def _system_stats() -> tuple[str, str, str, str, str]:
    """Startup snapshot of DB, graph, RAG, ML, and Discord status.

    Returns (pg, graph, rag, ml, discord) human-readable status strings.
    """
    import asyncio as _asyncio

    async def _pg() -> str:
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                async with store._pool.acquire() as c:
                    obs = await c.fetchval("SELECT COUNT(*) FROM job_observations")
                    cand = await c.fetchval("SELECT COUNT(*) FROM radar_candidates")
                    acc = await c.fetchval(
                        "SELECT COUNT(*) FROM radar_candidates WHERE eligibility='accepted'"
                    )
                    events = await c.fetchval("SELECT COUNT(*) FROM decision_events")
                    rewards = await c.fetchval(
                        "SELECT COUNT(*) FROM decision_events WHERE reward IS NOT NULL"
                    )
                    model = await c.fetchrow(
                        "SELECT version, status FROM model_registry "
                        "WHERE status='active' ORDER BY created_at DESC LIMIT 1"
                    )
                bits = [f"obs={obs}", f"cand={cand}", f"accepted={acc}"]
                bits.append(f"events={events}")
                bits.append(f"rewards={rewards}")
                if model:
                    bits.append(f"model={model['version']}({model['status']})")
                return " | ".join(bits)
            finally:
                await store.close()
        except Exception:
            return "pg unavailable"

    async def _graph() -> str:
        try:
            from src.graph.graph_store import GraphStore

            g = await GraphStore.create()
            try:
                nodes = await g.node_count()
                rels = await g.relationship_count()
            finally:
                await g.close()
            return f"nodes={nodes} rels={rels}"
        except Exception:
            return "graph unavailable"

    async def _rag() -> str:
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                resume = await store.chunk_count()
                persona = await store.persona_chunk_count()
            finally:
                await store.close()
            return f"resume_chunks={resume} persona_chunks={persona}"
        except Exception:
            return "rag unavailable"

    async def _ml() -> str:
        try:
            from src.memory.pgvector_store import MemoryStore

            store = await MemoryStore.create()
            try:
                async with store._pool.acquire() as c:
                    imp = await c.fetchval("SELECT COUNT(*) FROM impressions")
                    outcomes = await c.fetchval("SELECT COUNT(*) FROM unattributed_outcomes")
                    push = await c.fetchval(
                        "SELECT 1 FROM gmail_push_state WHERE history_id != '' LIMIT 1"
                    )
            finally:
                await store.close()
            bits = [f"impressions={imp}"]
            if outcomes:
                bits.append(f"outcomes={outcomes}")
            bits.append("gmail_push ON" if push else "gmail_push off")
            return " | ".join(bits)
        except Exception:
            return "ml unavailable"

    async def _discord() -> str:
        import os as _os

        token = (_os.getenv("DISCORD_BOT_TOKEN") or "").strip()
        channel = (_os.getenv("DISCORD_CHANNEL_ID") or "").strip()
        if not token or not channel:
            return "NOT CONFIGURED"
        # The Discord gateway runs inside the radar orchestrator process
        # (loop.py spawns it). Check for that process, not "discord_agent".
        try:
            import subprocess as _sp

            r = _sp.run(
                ["pgrep", "-f", "radar.engine.orchestrator"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            alive = bool((r.stdout or "").strip())
        except Exception:
            alive = False
        return (  # noqa: E501
            "configured · " + ("gateway in orchestrator" if alive else "no orchestrator running")
        )

    async def _all() -> tuple[str, str, str, str, str]:
        return await _asyncio.gather(_pg(), _graph(), _rag(), _ml(), _discord())

    return await _all()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true", help="Print all pipeline counters and exit")
    ap.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Continuously watch status updates in real-time dashboard",
    )
    ap.add_argument(
        "--once", action="store_true", help="Print status snapshot once and exit (no live watch)"
    )
    ap.add_argument("--dry-run", action="store_true", help="Start infra only, then stop")
    ap.add_argument("--no-fill", action="store_true", help="Skip the autofill worker")
    ap.add_argument("--radar-workers", type=int, default=2, help="Extra radar worker procs")
    ap.add_argument("--bridge-interval", type=int, default=120, help="Bridge drain seconds")
    ap.add_argument("--bridge-batch", type=int, default=50, help="Max candidates per drain")
    ap.add_argument("--max-minutes", type=int, default=0, help="Hard stop after N minutes")
    args = ap.parse_args()

    # Read-only status: no lock, no takeover, no infra — display dashboard.
    if args.status or args.watch:
        import sys as _sys

        is_watch = args.watch or (_sys.stdout.isatty() and not args.once)
        return _status_report(watch=is_watch)

    # Single-instance lock: only one `bun run run` may own the pipeline at a time.
    import os as _os
    import signal as _signal
    import sys as _sys

    lock_path = REPO / "logs" / "ho_run.lock"
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)

    # Automatically stream all run logs to logs/run.log & logs/loop.log
    try:
        run_log_path = log_dir / "run.log"
        log_fp = run_log_path.open("a", encoding="utf-8")
        _sys.stdout = _LogTee(_sys.stdout, log_fp)
        _sys.stderr = _LogTee(_sys.stderr, log_fp)
        latest_log_path = log_dir / "latest.log"
        latest_log_path.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            latest_log_path.symlink_to(run_log_path.name)
    except Exception:
        pass

    def _stop_pipeline_tree(old_pid: int) -> None:
        """Gracefully stop a running pipeline: the run_all + its loop, radar
        master/workers, and autofill worker. SIGINT first so in-flight jobs
        wrap up cleanly; SIGKILL only stragglers. Kills by pattern too so
        orphans (children of a died run_all) are reaped, while never
        signalling our own pid."""
        import subprocess as _sp

        def _pk(sig: str) -> None:
            patterns = (
                "scripts/loop.py",
                "scripts/run_all.py",
                "radar.engine.orchestrator",
                "autofill.src.core.worker",
            )
            own_pgid = _os.getpgid(_os.getpid())
            for pat in patterns:
                with contextlib.suppress(Exception):
                    r = _sp.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=5)
                    for pid in r.stdout.split():
                        try:
                            pid_i = int(pid)
                            # Skip our own tree (incl. the `uv run` wrapper,
                            # which has a different PID but shares our PGID and
                            # would forward SIGINT back to us).
                            if pid_i == _os.getpid() or _os.getpgid(pid_i) == own_pgid:
                                continue
                            _os.kill(pid_i, getattr(_signal, "SIG" + sig))
                        except ValueError, ProcessLookupError:
                            pass

        def _pk_alive() -> bool:
            """True if any pipeline pattern process (outside our group) lives."""
            patterns = (
                "scripts/loop.py",
                "radar.engine.orchestrator",
                "autofill.src.core.worker",
            )
            own_pgid = _os.getpgid(_os.getpid())
            for pat in patterns:
                try:
                    r = _sp.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=5)
                    for pid in r.stdout.split():
                        try:
                            pid_i = int(pid)
                            if pid_i == _os.getpid() or _os.getpgid(pid_i) == own_pgid:
                                continue
                            return True
                        except ValueError, ProcessLookupError:
                            pass
                except Exception:
                    pass
            return False

        with contextlib.suppress(Exception):
            _os.kill(old_pid, _signal.SIGINT)
        _pk("INT")
        # Grace period for graceful shutdown. Watch old_pid AND the pattern
        # matches: when run_all exits on SIGINT the loop would otherwise break
        # immediately, stranding stuck children (a hung orchestrator ignores
        # SIGINT) as orphans.
        for _ in range(20):
            with contextlib.suppress(ProcessLookupError):
                _os.kill(old_pid, 0)
            if not _pk_alive():
                break
            time.sleep(0.5)
        _pk("KILL")
        with contextlib.suppress(Exception):
            _os.kill(old_pid, _signal.SIGKILL)
        time.sleep(1)

    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            _os.kill(old_pid, 0)  # raises ProcessLookupError if dead
            print(f"[ho] stopping previous run (pid {old_pid})...", flush=True)
            _stop_pipeline_tree(old_pid)
            time.sleep(2)
            lock_path.unlink(missing_ok=True)
        except ProcessLookupError, ValueError:
            lock_path.unlink(missing_ok=True)
    else:
        # No lock file, but a pipeline may still be running (e.g. a run started
        # before the lock existed). Reap any lingering loop/radar/worker procs
        # so `bun run run` always starts one clean pipeline.
        with contextlib.suppress(Exception):
            import subprocess as _sp

            own_pgid = _os.getpgid(_os.getpid())
            for pat in (
                "scripts/loop.py",
                "radar.engine.orchestrator",
                "autofill.src.core.worker",
            ):
                r = _sp.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=5)
                for pid in r.stdout.split():
                    try:
                        pid_i = int(pid)
                        if pid_i == _os.getpid() or _os.getpgid(pid_i) == own_pgid:
                            continue
                        _os.kill(pid_i, _signal.SIGINT)
                    except ValueError, ProcessLookupError:
                        pass
            time.sleep(1)
    lock_path.write_text(str(_os.getpid()))

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    async def _run() -> int:
        if not await _ensure_infra():
            print("[ho] infra failed; aborting", flush=True)
            return 1
        # Autoheal: if containers exist but are Exited, restart before embed
        _autoheal_containers()
        _ensure_embed_server()
        await asyncio.sleep(8)
        if not await _wait_for("embed", HOST_PROBES["embed"][1]):
            print("[ho] embedding server not ready; continuing anyway", flush=True)
        else:
            print("[ho] embed server ✓ (:8900)", flush=True)
        _ensure_memory()
        _gmail_check()
        # Startup snapshot: DB / graph / RAG / ML / Discord.
        pg, graph, rag, ml, disc = await _system_stats()
        print(f"[ho] db: {pg}", flush=True)
        print(f"[ho] graph: {graph}", flush=True)
        print(f"[ho] rag: {rag}", flush=True)
        print(f"[ho] ml: {ml}", flush=True)
        print(f"[ho] discord: {disc}", flush=True)
        # One-line sweep intent so the user knows what this run will do
        import os as _os2

        epoch_target = int(_os2.environ.get("RADAR_SESSION_APPLICATION_TARGET", "0") or "0")
        target_desc = f"target={epoch_target} confirmed" if epoch_target > 0 else "no target cap"
        print(  # noqa: E501
            f"[ho] sweep config: workers={args.radar_workers}"
            f" bridge={args.bridge_interval}s {target_desc} (per-epoch); "
            "stops on queue idle / --max-minutes / Ctrl+C",
            flush=True,
        )
        if args.dry_run:
            print("[ho] dry-run complete; infra is up", flush=True)
            return 0
        return await _run_loop(args)

    try:
        rc = asyncio.run(_run())
        return rc
    except KeyboardInterrupt:
        print("[ho] stopped.", flush=True)
        return 0
    finally:
        # Release the single-instance lock so the next `bun run run` can start.
        with contextlib.suppress(Exception):
            if lock_path.exists() and lock_path.read_text().strip() == str(_os.getpid()):
                lock_path.unlink()


def _gmail_check() -> None:
    import os as _os

    if _os.getenv("GMAIL_PUSH", "").strip() != "1":
        print(  # noqa: E501
            "[ho] Gmail listener: disabled (GMAIL_PUSH != 1) — "
            "set GMAIL_PUSH=1 to capture outcome emails",
            flush=True,
        )
        return
    missing = [
        k
        for k in (
            "GMAIL_REFRESH_TOKEN",
            "GOOGLE_CLIENT_ID",
            "GOOGLE_CLIENT_SECRET",
            "GCP_PUBSUB_PROJECT",
        )
        if not _os.getenv(k)
    ]
    if missing:
        print(  # noqa: E501
            f"[ho] Gmail listener: GMAIL_PUSH=1 but missing {missing!r}"
            " — outcome emails will not be captured",
            flush=True,
        )
        return
    # The push daemon is a loop.py child (gmail_push). Verify it is actually
    # running, not just configured.
    push_alive = False
    try:
        import subprocess as _sp

        out = _sp.run(
            ["pgrep", "-f", "ml.src.outcomes.gmail_push"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        push_alive = bool((out.stdout or "").strip())
    except Exception:
        push_alive = False
    state = (  # noqa: E501
        "daemon running" if push_alive else "daemon NOT running (loop will start it)"
    )
    print(  # noqa: E501
        f"[ho] Gmail listener: configured (project={_os.getenv('GCP_PUBSUB_PROJECT')}) · {state}",
        flush=True,
    )


def _autoheal_containers() -> None:
    import subprocess as _sp

    for name in (
        "ho_searxng_1",
        "ho_agent-memory-db_1",
        "ho_neo4j_1",
        "ho_redis_1",
        "ho_steel_1",
    ):
        try:
            r = _sp.run(
                ["podman", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            status = (r.stdout or "").strip().split()[0] if r.stdout else "missing"
            if status == "Exited":
                print(f"[ho] autoheal: restarting {name} (was Exited)", flush=True)
                _sp.run(["podman", "start", name], capture_output=True, timeout=15)
            elif status == "missing":
                print(f"[ho] autoheal: {name} missing — compose up", flush=True)
                svc = name.replace("ho_", "").replace("_1", "")
                _sp.run(
                    ["docker", "compose", "-f", str(COMPOSE), "up", "-d", svc],
                    capture_output=True,
                    timeout=60,
                )
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
