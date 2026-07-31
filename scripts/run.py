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
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

PROJECT = Path(__file__).resolve().parent.parent
DOCKER_COMPOSE = f"docker compose -f {PROJECT}/docker-compose.yaml"

console = Console()

STATUS_UP = "[green]RUNNING[/green]"
STATUS_WAIT = "[yellow]WAITING[/yellow]"
STATUS_DOWN = "[red]DOWN[/red]"


def run(cmd: str, silent: bool = True) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=silent, text=True, timeout=30)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ho pipeline launcher")
    parser.add_argument(
        "--no-pipeline", action="store_true", help="Start infra only, don't run pipeline"
    )
    parser.add_argument("--no-health", action="store_true", help="Skip health checks before infra")
    args = parser.parse_args()

    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]ho[/bold cyan] — pipeline launcher",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # ── Cleanup ──
    console.print("[yellow]Stopping stale containers...[/yellow]")
    run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
    run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
    run("killall llama-server 2>/dev/null", silent=True)
    time.sleep(1)
    console.print("[green]Cleanup complete[/green]\n")

    # ── Launch llama-server embedding ──
    console.print("[cyan]Starting embedding server (llama-server :8900)...[/cyan]")
    run("killall llama-server 2>/dev/null", silent=True)
    subprocess.Popen(
        [sys.executable, f"{PROJECT}/scripts/serve.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # ── Launch containers ──
    run(
        f"{DOCKER_COMPOSE} up -d redis playwright-service "
        "nuq-postgres searxng neo4j agent-memory-db",
        silent=False,
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

    # Wait for rabbitmq before launching API
    for _ in range(15):
        time.sleep(2)
        if container_running("firecrawl_rabbitmq"):
            code, _ = run(
                "podman exec firecrawl_rabbitmq_1 rabbitmqctl await_startup 2>/dev/null",
                silent=True,
            )
            if code == 0:
                break

    run(f"{DOCKER_COMPOSE} up -d api", silent=True)

    # ── Live status table ──
    all_up = False
    embed_ok = False
    with Live(Table(), refresh_per_second=2, console=console) as live:
        for _ in range(60):
            t = Table(title="Service Status", box=box.SIMPLE_HEAVY, border_style="cyan")
            t.add_column("Service", style="bold")
            t.add_column("Status")
            t.add_column("Port")

            embed_ok = check_http("http://localhost:8900/health")
            row(t, "llama-server (Embed)", STATUS_UP if embed_ok else STATUS_WAIT, ":8900")

            row(
                t,
                "redis",
                STATUS_UP if container_running("firecrawl_redis") else STATUS_WAIT,
                ":6379",
            )
            row(
                t,
                "nuq-postgres",
                STATUS_UP if container_running("firecrawl_nuq-postgres") else STATUS_WAIT,
                ":5432",
            )
            row(
                t,
                "searxng",
                STATUS_UP if check_http("http://localhost:8080") else STATUS_WAIT,
                ":8080",
            )
            row(
                t,
                "neo4j",
                STATUS_UP if check_port("localhost", 7687) else STATUS_WAIT,
                ":7687",
            )
            row(
                t,
                "agent-memory-db",
                STATUS_UP if check_port("localhost", 5433) else STATUS_WAIT,
                ":5433",
            )

            rabbit_ok = container_running("firecrawl_rabbitmq")
            row(t, "rabbitmq", STATUS_UP if rabbit_ok else STATUS_WAIT, ":5672")

            pw_ok = container_running("firecrawl_playwright")
            row(t, "playwright", STATUS_UP if pw_ok else STATUS_WAIT, ":3000")

            api_ok = check_port("localhost", 3002)
            row(t, "firecrawl api", STATUS_UP if api_ok else STATUS_WAIT, ":3002")

            live.update(t)

            all_up = (
                embed_ok
                and rabbit_ok
                and pw_ok
                and api_ok
                and container_running("firecrawl_redis")
                and container_running("firecrawl_nuq-postgres")
            )
            if all_up:
                break
            time.sleep(1)

    # ── Final status ──
    t = Table(title="All Systems Ready", box=box.SIMPLE_HEAVY, border_style="cyan")
    t.add_column("Service", style="bold")
    t.add_column("Status")
    t.add_column("Port")
    row(t, "llama-server (Embed)", STATUS_UP if embed_ok else STATUS_DOWN, ":8900")
    row(t, "redis", STATUS_UP, ":6379")
    row(t, "nuq-postgres", STATUS_UP, ":5432")
    row(t, "searxng", STATUS_UP, ":8080")
    row(t, "neo4j", STATUS_UP, ":7687")
    row(t, "agent-memory-db", STATUS_UP, ":5433")
    row(t, "rabbitmq", STATUS_UP, ":5672")
    row(t, "playwright", STATUS_UP, ":3000")
    row(t, "firecrawl api", STATUS_UP, ":3002")
    console.print()
    console.print(t)

    if args.no_pipeline:
        console.print("\n[dim]Infrastructure running. Press Ctrl+C to stop.[/dim]")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")
            run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
            run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
            run("killall llama-server 2>/dev/null", silent=True)
        return

    # ── Pipeline ──
    log_dir = PROJECT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "run.log"

    console.print("\n[bold cyan]── Starting pipeline ──[/bold cyan]\n")

    env = os.environ.copy()
    env.setdefault("OVERNIGHT_LOOP", "true")

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.radar.orchestrator"],
        cwd=str(PROJECT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _on_signal(sig, _frame):
        proc.send_signal(sig)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    with open(log_path, "a") as log_file:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)

    proc.wait()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
