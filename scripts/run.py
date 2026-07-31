#!/usr/bin/env python3
"""Full pipeline launcher: start all services, wait for health, run orchestrator.

Usage:
    make run          → full pipeline
    make dev          → infrastructure only (same as --no-pipeline)
"""

import argparse
import contextlib
import os
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

_proc: subprocess.Popen | None = None


def run(cmd: str, silent: bool = True, timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=silent, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)[:500]
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


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
        f"podman ps --filter name='{name}' --filter status=running --format '{{{{.Names}}}}'"
    )
    return code == 0


def row(t: Table, name: str, status: str, port: str = "") -> None:
    t.add_row(name, status, port)


def stop_all() -> None:
    console.print("\n[yellow]Stopping all services...[/yellow]")
    run("killall llama-server 2>/dev/null", silent=True)
    run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
    run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
    console.print("[green]All services stopped.[/green]")


def _podman_exec(container: str, cmd: str) -> str:
    code, out = run(
        f"podman exec {container} {cmd}",
        silent=True,
        timeout=10,
    )
    return out.strip() if code == 0 else ""


def deep_stats() -> dict[str, Any]:
    """Collect rich infra stats: DB counts, queue depth, embed info."""
    info: dict[str, Any] = {}

    # Neo4j node count
    if check_port("localhost", 7687):
        raw = _podman_exec(
            "firecrawl_neo4j_1",
            "cypher-shell -u neo4j -p password 'MATCH (n) RETURN count(n) AS nodes' 2>/dev/null",
        )
        for line in raw.split("\n"):
            if line.strip().isdigit():
                info["neo4j_nodes"] = int(line.strip())
                break

    # pgvector row counts
    if check_port("localhost", 5433):
        for table in ("job_observations", "job_candidates", "discovered_sources"):
            raw = _podman_exec(
                "firecrawl_agent-memory-db_1",
                f"psql -U postgres -d agent_memory -t "
                f"-c 'SELECT COUNT(*) FROM {table}' 2>/dev/null",
            )
            val = raw.strip()
            if val.isdigit():
                info[f"pg_{table}"] = int(val)

    # RabbitMQ queue depths
    if container_running("firecrawl_rabbitmq"):
        raw = _podman_exec(
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

            r = urllib.request.urlopen("http://localhost:8900/slots", timeout=5)
            data = _json.loads(r.read())
            for slot in data:
                info["embed_model"] = slot.get("model", "?")
                info["embed_slots"] = len(data)
        except Exception:
            pass

    # Container CPU/mem
    try:
        r = subprocess.run(
            "podman stats --no-stream "
            "--format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}' "
            "firecrawl_api_1 firecrawl_playwright-service_1 "
            "firecrawl_rabbitmq_1 firecrawl_redis_1 "
            "firecrawl_neo4j_1 2>/dev/null",
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


def shutdown_handler(sig: int, _frame: object) -> None:
    """Intercept Ctrl+C: ask user, then stop everything."""
    console.print("\n\n[bold yellow]Stop ho service? [y/N][/bold yellow] ", end="")
    try:
        # Read from tty directly since stdin may be consumed by subprocess
        with open("/dev/tty") as tty:
            answer = tty.readline().strip().lower()
    except Exception:
        answer = ""

    if answer in ("y", "yes"):
        console.print("[red]Shutting down...[/red]")
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        stop_all()
        sys.exit(0)
    else:
        console.print("[dim]Continuing...[/dim]")


def main() -> None:
    global _proc

    parser = argparse.ArgumentParser(description="ho pipeline launcher")
    parser.add_argument(
        "--no-pipeline", action="store_true", help="Start infra only, don't run pipeline"
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

    # ── Cleanup ──
    stop_all()
    time.sleep(1)

    # Define services
    services: list[tuple[str, Any, str]] = [
        ("llama-server (Embed)", lambda: check_http("http://localhost:8900/health"), ":8900"),
        ("redis", lambda: container_running("firecrawl_redis"), ":6379"),
        ("nuq-postgres", lambda: container_running("firecrawl_nuq-postgres"), ":5432"),
        ("searxng", lambda: check_http("http://localhost:8080"), ":8080"),
        ("neo4j", lambda: check_port("localhost", 7687), ":7687"),
        ("agent-memory-db", lambda: check_port("localhost", 5433), ":5433"),
        ("rabbitmq", lambda: container_running("firecrawl_rabbitmq"), ":5672"),
        ("playwright", lambda: container_running("firecrawl_playwright"), ":3000"),
        ("firecrawl api", lambda: check_port("localhost", 3002), ":3002"),
    ]

    # Kick off startup in background
    subprocess.Popen(
        [sys.executable, f"{PROJECT}/scripts/serve.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    run(
        f"{DOCKER_COMPOSE} up -d redis playwright-service "
        "nuq-postgres searxng neo4j agent-memory-db api",
        silent=True,
    )

    run(
        "podman run -d --name firecrawl_rabbitmq_1 "
        "--network firecrawl_default --network-alias rabbitmq "
        "--restart unless-stopped "
        "--entrypoint /bin/bash rabbitmq:3-management "
        '-c "rm -f /var/lib/rabbitmq/.erlang.cookie; '
        'exec docker-entrypoint.sh rabbitmq-server"',
        silent=True,
    )

    # ── Live status table while containers come up ──
    failed: list[str] = []
    with Live(Table(), refresh_per_second=3, console=console) as live:
        for _ in range(90):
            t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
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

    # ── Final status + initial deep stats ──
    console.print()
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("")
    for name, _fn, port in services:
        row(t, name, STATUS_UP, port)
    console.print(t)

    # Show initial DB state
    snapshot = deep_stats()
    console.print("[dim]Infra snapshot:[/dim]")
    for k, v in sorted(snapshot.items()):
        if not k.startswith("cpu_") and not k.startswith("mem_"):
            console.print(f"  [dim]{k}: {v}[/dim]")

    console.print("\n[dim]All systems ready.[/dim]")

    if args.no_pipeline:
        console.print("\n[dim]Press Ctrl+C to stop.[/dim]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            stop_all()
        return

    # ── Pipeline ──
    log_dir = PROJECT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "run.log"

    console.print("\n[bold cyan]Pipeline starting...[/bold cyan]")
    console.print("[dim]Ctrl+C → stop prompt | All logs in logs/run.log[/dim]\n")

    env = os.environ.copy()
    env.setdefault("OVERNIGHT_LOOP", "true")

    # Background container stats logger
    stop_stats = threading.Event()
    stats_thread = threading.Thread(
        target=stats_logger,
        args=(log_path, stop_stats),
        daemon=True,
    )
    stats_thread.start()

    _proc = subprocess.Popen(
        [sys.executable, "-m", "src.radar.orchestrator"],
        cwd=str(PROJECT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Install our shutdown handler
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        with open(log_path, "a") as log_file:
            assert _proc.stdout is not None
            for line in _proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
    except Exception:
        pass

    _proc.wait()
    stop_stats.set()

    if _proc.returncode != 0 and _proc.returncode != -15:  # -15 = SIGTERM
        console.print(f"\n[red]Pipeline exited with code {_proc.returncode}[/red]")
        stop_all()
        sys.exit(_proc.returncode)


if __name__ == "__main__":
    main()
