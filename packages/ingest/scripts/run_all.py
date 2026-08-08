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
]

# Infra readiness probes (host, port). searxng/neo4j/agent-memory-db are the
# only remaining compose services — Firecrawl (api/playwright/rabbitmq/redis/
# nuq-postgres) is gone, and the embed server is started below.
HOST_PROBES = {
    "searxng": (8080, 15),
    "neo4j": (7687, 15),
    "agent-memory-db": (5433, 20),
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
    print(f"[run_all] waiting for {name}...", flush=True)
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
    print("[run_all] starting docker services...", flush=True)
    code, out = _docker(["up", "-d", *DOCKER_SERVICES])
    if code != 0:
        print(f"[run_all] docker compose up failed: {out}", flush=True)
        return False
    return True


def _preflight_backup() -> None:
    """Checkpoint volumes before the sweep starts (fire-and-forget, 90s cap)."""
    import subprocess as _sp

    try:
        print("[run_all] pre-run backup (checkpoint)...", flush=True)
        r = _sp.run(
            ["uv", "run", "python", "scripts/backup/auto_backup.py"],
            cwd=str(PROJECT),
            capture_output=True,
            text=True,
            timeout=90,
        )
        print(f"[run_all] backup: {r.stdout.strip()[-200:] if r.stdout else 'ok'}", flush=True)
    except Exception as e:
        print(f"[run_all] backup skipped: {e}", flush=True)


async def _ensure_infra() -> bool:
    if not _compose_up():
        return False
    ok = True
    for name in ("agent-memory-db", "neo4j", "searxng", "embed"):
        # embed is started below; if already up, great.
        if name == "embed" and not _http_ok("http://localhost:8900/health"):
            continue
        if not await _wait_for(name, HOST_PROBES.get(name, (0, 15))[1]):
            print(f"[run_all] WARNING: {name} not ready", flush=True)
            ok = False
        else:
            print(f"[run_all] ✓ {name} ready", flush=True)
    # Pre-run backup once infra is confirmed
    _preflight_backup()
    return ok


def _ensure_embed_server() -> None:
    if _http_ok("http://localhost:8900/health"):
        print("[run_all] embedding server already up", flush=True)
        return
    print("[run_all] starting embedding server...", flush=True)
    log = PROJECT / "logs"
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


def _ensure_memory() -> None:
    persona = REPO / "data" / "persona.json"
    if not persona.exists():
        print("[run_all] persona.json missing — seeding memory non-interactively...", flush=True)
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
            print(f"[run_all] memory seed skipped ({e}); pipeline will use defaults", flush=True)
        return
    # Present but possibly stale: report it so the user knows to re-run
    # init-memory after a resume update instead of silently matching on old
    # grounding.
    print(f"[run_all] memory: {_memory_status()}", flush=True)


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
    print(f"[run_all] launching loop: {' '.join(cmd)}", flush=True)
    log_dir = PROJECT / "logs"
    log_dir.mkdir(exist_ok=True)
    # Stream the loop's output to a file, never a pipe we don't drain — a pipe
    # fills up (the radar children flood stdout) and the loop deadlocks.
    with (log_dir / "run_all.log").open("ab") as out:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
        return await proc.wait()


def _handle_sig(signum: int, frame) -> None:  # noqa: ANN001
    import contextlib

    with contextlib.suppress(Exception):
        print(f"\n[run_all] signal {signum}; shutting down...", flush=True)
    raise KeyboardInterrupt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="Start infra only, then stop")
    ap.add_argument("--no-fill", action="store_true", help="Skip the autofill worker")
    ap.add_argument("--radar-workers", type=int, default=2, help="Extra radar worker procs")
    ap.add_argument("--bridge-interval", type=int, default=120, help="Bridge drain seconds")
    ap.add_argument("--bridge-batch", type=int, default=50, help="Max candidates per drain")
    ap.add_argument("--max-minutes", type=int, default=0, help="Hard stop after N minutes")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    async def _run() -> int:
        if not await _ensure_infra():
            print("[run_all] infra failed; aborting", flush=True)
            return 1
        # Autoheal: if containers exist but are Exited, restart before embed
        _autoheal_containers()
        _ensure_embed_server()
        await asyncio.sleep(8)
        if not await _wait_for("embed", HOST_PROBES["embed"][1]):
            print("[run_all] embedding server not ready; continuing anyway", flush=True)
        else:
            print("[run_all] embed server ✓ (:8900)", flush=True)
        _ensure_memory()
        _gmail_check()
        # One-line sweep intent so the user knows what this run will do
        print(  # noqa: E501
            f"[run_all] sweep config: workers={args.radar_workers}"
            f" bridge={args.bridge_interval}s target=20 confirmed (per-epoch)",
            flush=True,
        )
        if args.dry_run:
            print("[run_all] dry-run complete; infra is up", flush=True)
            return 0
        return await _run_loop(args)

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("[run_all] stopped.", flush=True)
        return 0


def _gmail_check() -> None:
    import os as _os

    if _os.getenv("GMAIL_PUSH", "").strip() != "1":
        print(  # noqa: E501
            "[run_all] Gmail listener: disabled (GMAIL_PUSH != 1) — "
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
            f"[run_all] Gmail listener: GMAIL_PUSH=1 but missing {missing!r}"
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
        f"[run_all] Gmail listener: configured (project="
        f"{_os.getenv('GCP_PUBSUB_PROJECT')}) · {state}",
        flush=True,
    )


def _autoheal_containers() -> None:
    import subprocess as _sp

    for name in ("ho_searxng_1", "ho_agent-memory-db_1", "ho_neo4j_1"):
        try:
            r = _sp.run(
                ["podman", "ps", "-a", "--filter", f"name={name}", "--format", "{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            status = (r.stdout or "").strip().split()[0] if r.stdout else "missing"
            if status == "Exited":
                print(f"[run_all] autoheal: restarting {name} (was Exited)", flush=True)
                _sp.run(["podman", "start", name], capture_output=True, timeout=15)
            elif status == "missing":
                print(f"[run_all] autoheal: {name} missing — compose up", flush=True)
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
