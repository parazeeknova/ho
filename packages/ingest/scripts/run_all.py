#!/usr/bin/env python3
"""One-command full-stack runner: infra + embed + radar + bridge + autofill.

Everything ho offers, from a single invocation:

  1. docker compose up the ingest stack (redis, nuq-postgres, searxng, neo4j,
     rabbitmq, playwright, firecrawl api, agent-memory-db) and wait for health;
  2. start the local embedding server (:8900);
  3. if persona.json is missing, seed memory (resume + persona) non-interactively;
  4. run the end-to-end loop: radar pipeline (master + workers) discovers and
     LLM-matches jobs, the bridge drains accepted roles into the autofill
     queue, and the autofill worker (1 concurrent browser) auto-applies.
     Crashed children are restarted; the run continues overnight until stopped.

Local-only by default: company discovery uses the local adapters (yc, dealroom,
hn, remoteok, ...), not the Azure relic. Set AZURE=1 to re-enable relic discovery.

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
    "redis",
    "nuq-postgres",
    "searxng",
    "neo4j",
    "rabbitmq",
    "playwright-service",
    "api",
    "agent-memory-db",
]

# Infra readiness probes (host, port). Internal-only services (rabbitmq,
# playwright, nuq-postgres) are reachable inside the compose network but not
# necessarily on the host, so those are skipped here — the api container's
# healthcheck gate covers them.
HOST_PROBES = {
    "redis": (6379, 10),
    "searxng": (8080, 15),
    "neo4j": (7687, 15),
    "api": (3002, 60),
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
        if name == "api" and _http_ok("http://localhost:3002/"):
            return True
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


async def _ensure_infra() -> bool:
    if not _compose_up():
        return False
    # Give the api container a moment (it depends on rabbitmq health; the
    # compose healthcheck gate handles the wait, but podman sometimes reports
    # healthy a touch early).
    await asyncio.sleep(5)
    ok = True
    for name in ("agent-memory-db", "neo4j", "redis", "searxng", "api", "embed"):
        # embed is started below; if already up, great.
        if name == "embed" and not _http_ok("http://localhost:8900/health"):
            continue
        if not await _wait_for(name, HOST_PROBES.get(name, (0, 15))[1]):
            print(f"[run_all] WARNING: {name} not ready", flush=True)
            ok = False
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


def _ensure_memory() -> None:
    persona = REPO / "data" / "persona.json"
    if persona.exists():
        return
    print("[run_all] persona.json missing — seeding memory non-interactively...", flush=True)
    try:
        env = dict(os.environ)
        env["NON_INTERACTIVE"] = "1"
        subprocess.run(
            [sys.executable, str(REPO / "packages" / "autofill" / "scripts" / "init_memory.py")],
            cwd=str(REPO),
            env=env,
            timeout=600,
        )
    except Exception as e:
        print(f"[run_all] memory seed skipped ({e}); pipeline will use defaults", flush=True)


async def _run_loop(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    paths = [str(PROJECT), str(REPO / "packages" / "autofill")]
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
        _ensure_embed_server()
        await asyncio.sleep(8)
        if not await _wait_for("embed", HOST_PROBES["embed"][1]):
            print("[run_all] embedding server not ready; continuing anyway", flush=True)
        _ensure_memory()
        if args.dry_run:
            print("[run_all] dry-run complete; infra is up", flush=True)
            return 0
        return await _run_loop(args)

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("[run_all] stopped.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
