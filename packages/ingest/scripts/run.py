#!/usr/bin/env python3
"""Full pipeline launcher: start all services, wait for health, run orchestrator.

Usage:
    make run          → full pipeline
    make dev          → infrastructure only (same as --no-pipeline)
"""

import argparse
import contextlib
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT = Path(__file__).resolve().parent.parent
DOCKER_COMPOSE = f"docker compose -f {PROJECT}/docker-compose.yaml"

console = Console()

STATUS_UP = "[green]READY[/green]"
STATUS_WAIT = "[yellow]·[/yellow]"
STATUS_DOWN = "[red]DOWN[/red]"


def _env_value_or_default(key: str, default: str) -> str:
    """Read ``key`` from the process env, then the project's .env file (so a
    value the user set in .env is honored without requiring it in the shell).
    Returns ``default`` only when neither source specifies the key."""
    value = os.environ.get(key)
    if value is not None and value.strip() != "":
        return value.strip()
    try:
        dotenv_path = PROJECT / ".env"
        if dotenv_path.exists():
            for raw_line in dotenv_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return default


_proc: subprocess.Popen | None = None


def run(cmd: str, silent: bool = True, timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=silent, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[:500]
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def check_neo4j_ready() -> bool:
    """Verify Neo4j is actually serving queries, not just TCP accepting."""
    raw = _docker_exec(
        "firecrawl-neo4j-1",
        "cypher-shell -u neo4j -p password 'RETURN 1 AS ready' 2>/dev/null",
    )
    return "ready" in raw.lower() and "1" in raw


def check_http(url: str) -> bool:
    with contextlib.suppress(Exception):
        import urllib.request

        urllib.request.urlopen(url, timeout=3)
        return True
    return False


def check_port(host: str, port: int) -> bool:
    with contextlib.suppress(Exception), socket.create_connection((host, port), timeout=2):
        return True
    return False


def container_running(name: str) -> bool:
    code, _ = run(
        f"docker ps --filter name='{name}' --filter status=running --format '{{{{.Names}}}}'"
    )
    return code == 0


def row(t: Table, name: str, status: str, port: str = "") -> None:
    t.add_row(name, status, port)


def stop_all() -> None:
    console.print("\n[yellow]Stopping all services...[/yellow]")
    run("killall llama-server 2>/dev/null", silent=True)
    # Stop pipeline containers but leave the sync stack (agent-memory-db,
    # the ingest's Postgres) untouched so azure sync keeps running.
    run(
        f"{DOCKER_COMPOSE} stop redis playwright-service nuq-postgres "
        "searxng neo4j api 2>/dev/null",
        silent=True,
    )
    run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
    console.print("[green]All services stopped.[/green]")


def _docker_exec(container: str, cmd: str) -> str:
    code, out = run(
        f"docker exec {container} {cmd}",
        silent=True,
        timeout=10,
    )
    return out.strip() if code == 0 else ""


def deep_stats() -> dict[str, Any]:
    """Collect rich infra stats: DB counts, queue depth, embed info."""
    info: dict[str, Any] = {}

    # Neo4j node count
    if check_port("localhost", 7687):
        raw = _docker_exec(
            "firecrawl-neo4j-1",
            "cypher-shell -u neo4j -p password 'MATCH (n) RETURN count(n) AS nodes' 2>/dev/null",
        )
        for line in raw.split("\n"):
            if line.strip().isdigit():
                info["neo4j_nodes"] = int(line.strip())
                break

    # pgvector row counts
    if check_port("localhost", 5433):
        for table in ("job_observations", "job_candidates", "discovered_sources"):
            raw = _docker_exec(
                "firecrawl-agent-memory-db-1",
                f"psql -U postgres -d agent_memory -t "
                f"-c 'SELECT COUNT(*) FROM {table}' 2>/dev/null",
            )
            val = raw.strip()
            if val.isdigit():
                info[f"pg_{table}"] = int(val)

    # RabbitMQ queue depths
    if container_running("firecrawl_rabbitmq"):
        raw = _docker_exec(
            "firecrawl_rabbitmq_1",
            "rabbitmqctl list_queues name messages 2>/dev/null",
        )
        for line in raw.split("\n"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                info[f"mq_{parts[0]}"] = int(parts[1])

    # Embedding model info
    if check_http("http://localhost:8900/health"):
        try:
            import json as _json
            import urllib.request

            r = urllib.request.urlopen("http://localhost:8900/v1/models", timeout=5)
            data = _json.loads(r.read())
            # llama.cpp returns {"data": [...]} or {"models": [...]} depending on version
            models_list = data.get("data", data.get("models", []))
            if models_list and isinstance(models_list, list):
                first = models_list[0]
                name = first.get("name", first.get("id", "?"))
                if name:
                    info["embed_model"] = name
                info["embed_slots"] = len(models_list)
        except Exception:
            pass

    # Container CPU/mem
    try:
        r = subprocess.run(
            "docker stats --no-stream "
            "--format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}' "
            "firecrawl-api-1 firecrawl-playwright-service-1 "
            "firecrawl_rabbitmq_1 firecrawl-redis-1 "
            "firecrawl-neo4j-1 2>/dev/null",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) >= 4:
                    name = parts[0].replace("firecrawl_", "")
                    info[f"cpu_{name}"] = parts[1]
                    info[f"mem_{name}"] = parts[2]
    except Exception:
        pass

    return info


def format_stats(info: dict[str, Any]) -> str:
    """Format deep stats into a human-readable line."""
    parts: list[str] = []

    # CPU/Mem section
    for key, label in [
        ("cpu_api_1", "api"),
        ("cpu_playwright-service_1", "pw"),
        ("cpu_rabbitmq_1", "mq"),
        ("cpu_redis_1", "redis"),
        ("cpu_neo4j_1", "neo4j"),
    ]:
        if key in info:
            cpu = info.get(key, "?")
            mem_key = key.replace("cpu_", "mem_")
            mem = info.get(mem_key, "?")
            parts.append(f"{label} cpu={cpu} mem={mem}")

    # DB counts
    for key, label in [
        ("neo4j_nodes", "neo4j nodes"),
        ("pg_job_observations", "pg:observations"),
        ("pg_job_candidates", "pg:candidates"),
        ("pg_discovered_sources", "pg:sources"),
    ]:
        if key in info:
            parts.append(f"{label}={info[key]}")

    # Queue depths
    for key, label in [
        ("mq_extract.jobs", "mq:extract"),
        ("mq_nuq.queue_scrape.prefetch", "mq:nuq-scrape"),
    ]:
        if key in info:
            parts.append(f"{label}={info[key]}")

    # Embed
    if "embed_model" in info:
        parts.append(f"embed={info['embed_model']}")

    return "[infra] " + " | ".join(parts) if parts else ""


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


_FC_KEEP = [
    "Worker taking job",
    "Job done",
    "Scraping URL",
    "deemed successful",
    "deemed failed",
    "map_url",
    "error",
    "Error",
    "failed",
]
_FC_DROP = [
    "bypassing authentication",
    "USE_DB_AUTHENTICATION",
    "robustInsert",
    "runWebScraper called",
    "running scrapeURL",
    "scrapeURL entered",
    "Selected engines",
    "Scraping via playwright",
    "Scraping via fetch",
    "Done with waitForJob",
    "Removed job from queue",
    "Request metrics",
    "request completed",
    "scrapeController",
    "log_job",
    "Connected to Redis",
    "Redis connected",
    "NUQ",
    "AUTUMN_SECRET_KEY",
    "Worker 10 listening",
    "WebSocket proxy",
    "Network info dump",
    "Number of CPUs",
    "Worker 10 started",
    "Attaching WebSocket",
    "NuQ reconciler",
    "NuQ prefetch",
    "Concurrency queue",
    "All services running",
    "All processes terminated",
    "Starting services",
    "Waiting for API",
    "Skipping container",
    "playwright-service",
    "nuq-postgres",
    "postgres",
    "Started consuming",
    "ENOTFOUND redis",
    "ioredis",
    "Redis error",
    "getaddrinfo ENOTFOUND",
]


def _fc_should_show(line: str) -> bool:
    if any(d in line for d in _FC_DROP):
        return False
    return any(k in line for k in _FC_KEEP)


def firecrawl_logger(log_path: Path, stop_event: threading.Event) -> None:
    """Tail Firecrawl API container logs, showing only scrape activity."""
    time.sleep(3)
    while not stop_event.is_set():
        try:
            proc = subprocess.Popen(
                "podman logs --since 2s -f firecrawl_api_1 2>/dev/null",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                if stop_event.is_set():
                    proc.terminate()
                    break
                clean = _strip_ansi(line).strip()
                if not clean or not _fc_should_show(clean):
                    continue
                ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
                msg = f"[fc] {clean}"
                try:
                    with open(log_path, "a") as f:
                        f.write(
                            f'{{"timestamp": "{ts}", "level": "INFO", '
                            f'"message": "{msg}", '
                            f'"logger": "firecrawl_tail"}}\n'
                        )
                        f.flush()
                    sys.stdout.write(f"{msg}\n")
                    sys.stdout.flush()
                except Exception:
                    pass
        except Exception:
            pass
        stop_event.wait(5)


def stats_logger(log_path: Path, stop_event: threading.Event, interval: float = 30.0) -> None:
    """Background daemon: periodically snapshot deep infra stats to log file."""
    time.sleep(interval)
    while not stop_event.is_set():
        info = deep_stats()
        msg = format_stats(info)
        if msg:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            try:
                with open(log_path, "a") as f:
                    f.write(
                        f'{{"timestamp": "{ts}", "level": "INFO", '
                        f'"message": "{msg}", "logger": "infra_monitor"}}\n'
                    )
                    f.flush()
            except Exception:
                pass
        stop_event.wait(interval)


_procs: list[subprocess.Popen] = []


def shutdown_handler(sig: int, _frame: object) -> None:
    """Intercept Ctrl+C: ask user, then stop everything."""
    console.print("\n\n[bold yellow]Stop ho pipeline and workers? [y/N][/bold yellow] ", end="")
    try:
        # Read from tty directly since stdin may be consumed by subprocess
        with open("/dev/tty") as tty:
            answer = tty.readline().strip().lower()
    except Exception:
        answer = ""

    if answer in ("y", "yes"):
        console.print("[red]Shutting down master and workers...[/red]")
        for p in _procs:
            if p.poll() is None:
                p.terminate()
        for p in _procs:
            if p.poll() is None:
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        stop_all()
        sys.exit(0)
    else:
        console.print("[dim]Continuing...[/dim]")


def main() -> None:
    global _procs

    parser = argparse.ArgumentParser(description="ho pipeline launcher")
    parser.add_argument(
        "--no-pipeline", action="store_true", help="Start infra only, don't run pipeline"
    )
    parser.add_argument("--no-cloud", action="store_true", help="Run fully offline: no Azure sync")
    parser.add_argument(
        "--worker-only", action="store_true", help="Start as a dedicated queue worker process"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of local parallel worker processes (default: 4)",
    )
    args = parser.parse_args()

    console.clear()
    console.print(
        Panel(
            Text("ho", style="bold cyan"),
            title="pipeline launcher",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Cleanup
    stop_all()
    time.sleep(1)

    # Define services
    services: list[tuple[str, Any, str]] = [
        ("llama-server (Embed)", lambda: check_http("http://localhost:8900/health"), ":8900"),
        ("redis", lambda: container_running("firecrawl-redis-1"), ":6379"),
        ("nuq-postgres", lambda: container_running("firecrawl-nuq-postgres-1"), ":5432"),
        ("searxng", lambda: check_http("http://localhost:8080"), ":8080"),
        ("neo4j", check_neo4j_ready, ":7687"),
        ("agent-memory-db", lambda: check_port("localhost", 5433), ":5433"),
        ("rabbitmq", lambda: container_running("firecrawl_rabbitmq_1"), ":5672"),
        ("playwright", lambda: container_running("firecrawl-playwright-service-1"), ":3000"),
        ("firecrawl api", lambda: check_port("localhost", 3002), ":3002"),
    ]

    # Kick off startup in background
    subprocess.Popen(
        [sys.executable, f"{PROJECT}/scripts/serve.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    run(
        f"{DOCKER_COMPOSE} up -d redis playwright-service nuq-postgres searxng neo4j api",
        silent=True,
    )

    run(
        "docker run -d --name firecrawl_rabbitmq_1 "
        "--network firecrawl_default --network-alias rabbitmq "
        "--restart unless-stopped "
        "--entrypoint /bin/bash rabbitmq:3-management "
        '-c "rm -f /var/lib/rabbitmq/.erlang.cookie; '
        'exec docker-entrypoint.sh rabbitmq-server"',
        silent=True,
    )

    # Live status table while containers come up
    failed: list[str] = []
    wait_start = time.monotonic()
    with Live(Table(), refresh_per_second=3, console=console) as live:
        for _ in range(90):
            elapsed = int(time.monotonic() - wait_start)
            t = Table(
                title=f"Waiting for services... {elapsed}s",
                box=box.SIMPLE,
                show_header=False,
                padding=(0, 2),
                expand=False,
            )
            t.add_column("")

            all_up = True
            for name, check_fn, port in services:
                ok = check_fn()
                if ok:
                    status = STATUS_UP
                else:
                    status = STATUS_WAIT
                    all_up = False
                row(t, name, status, port)

            live.update(t)

            if all_up:
                break
            time.sleep(1)
        else:
            failed = [name for name, check_fn, _ in services if not check_fn()]

    if failed:
        console.print("\n[red]Some services failed to start:[/red]")
        for f_name in failed:
            console.print(f"  [red]✗[/red] {f_name}")
        stop_all()
        sys.exit(1)

    console.print("\n[dim]All systems ready.[/dim]")

    if args.no_pipeline:
        console.print("\n[dim]Press Ctrl+C to stop.[/dim]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            stop_all()
            sys.exit(0)

    # Pipeline
    log_dir = PROJECT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "run.log"

    num_workers = max(1, args.workers)
    if args.worker_only:
        num_workers = 1

    console.print(
        f"\n[bold cyan]Pipeline starting with {num_workers} parallel process workers...[/bold cyan]"
    )
    console.print("[dim]Ctrl+C → stop prompt | All logs in logs/run.log[/dim]\n")

    env = os.environ.copy()
    env.setdefault("OVERNIGHT_LOOP", "true")
    # Load Azure relic creds (AZURE_STORAGE_ACCOUNT/KEY) so the orchestrator's
    # Azure-only company discovery can read the relic's companies blobs.
    _wd_env = Path(__file__).resolve().parent / ".watchdog.env"
    if _wd_env.exists():
        for _line in _wd_env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                env.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    # Full-force LLM matching: force the queue throttles so the workers
    # blast through the corpus (the cloud provider quota allows it).
    env["LLM_QUEUE_RPM"] = "240"
    env["LLM_QUEUE_MAX_IN_FLIGHT"] = "30"
    env["LLM_QUEUE_TPM"] = "400000"
    env["LLM_BUDGET_RADAR_RPM"] = "240"
    env["LLM_BUDGET_RADAR_TPM"] = "400000"

    # Background container stats logger
    stop_stats = threading.Event()
    stats_thread = threading.Thread(
        target=stats_logger,
        args=(log_path, stop_stats),
        daemon=True,
    )
    stats_thread.start()

    # Background Firecrawl log tail
    fc_thread = threading.Thread(
        target=firecrawl_logger,
        args=(log_path, stop_stats),
        daemon=True,
    )
    fc_thread.start()

    _procs = []
    if args.worker_only:
        w_env = env.copy()
        w_env["HO_WORKER_ONLY"] = "1"
        p = subprocess.Popen(
            [sys.executable, "-m", "src.radar.engine.orchestrator"],
            cwd=str(PROJECT),
            env=w_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _procs.append(p)
    else:
        # Master orchestrator
        p_main = subprocess.Popen(
            [sys.executable, "-m", "src.radar.engine.orchestrator"],
            cwd=str(PROJECT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _procs.append(p_main)

        # Worker processes
        w_env = env.copy()
        w_env["HO_WORKER_ONLY"] = "1"
        for _w_idx in range(1, num_workers):
            p_w = subprocess.Popen(
                [sys.executable, "-m", "src.radar.engine.orchestrator"],
                cwd=str(PROJECT),
                env=w_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _procs.append(p_w)

    # Install shutdown handler
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    def _stream_proc(p: subprocess.Popen, name: str) -> None:
        try:
            with open(log_path, "a") as log_file:
                assert p.stdout is not None
                for line in p.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_file.write(line)
                    log_file.flush()
        except Exception:
            pass

    threads: list[threading.Thread] = []
    for idx, p in enumerate(_procs):
        label = f"Worker-{idx}" if idx > 0 else "Master"
        t = threading.Thread(target=_stream_proc, args=(p, label), daemon=True)
        t.start()
        threads.append(t)

    _procs[0].wait()
    stop_stats.set()

    if _procs[0].returncode != 0 and _procs[0].returncode != -15:
        console.print(f"\n[red]Pipeline exited with code {_procs[0].returncode}[/red]")
        stop_all()
        sys.exit(_procs[0].returncode)


if __name__ == "__main__":
    main()
