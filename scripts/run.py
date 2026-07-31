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
from rich.text import Text

PROJECT = Path(__file__).resolve().parent.parent
DOCKER_COMPOSE = f"docker compose -f {PROJECT}/docker-compose.yaml"

console = Console()

STATUS_UP = "[green]READY[/green]"
STATUS_WAIT = "[yellow]·[/yellow]"
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


def stop_all() -> None:
    run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
    run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
    run("killall llama-server 2>/dev/null", silent=True)


def main() -> None:
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

    # ── Cleanup + start ──
    stop_all()
    time.sleep(1)

    # llama-server (embedding)
    subprocess.Popen(
        [sys.executable, f"{PROJECT}/scripts/serve.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Containers
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

    # Wait for rabbitmq
    for _ in range(20):
        time.sleep(2)
        if container_running("firecrawl_rabbitmq"):
            code, _ = run(
                "podman exec firecrawl_rabbitmq_1 rabbitmqctl await_startup 2>/dev/null",
                silent=True,
            )
            if code == 0:
                break

    # ── Live status table ──
    embed_ok = False
    failed: list[str] = []

    with Live(Table(), refresh_per_second=3, console=console) as live:
        for _ in range(90):
            t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
            t.add_column("")

            services = [
                (
                    "llama-server (Embed)",
                    lambda: check_http("http://localhost:8900/health"),
                    ":8900",
                ),
                ("redis", lambda: container_running("firecrawl_redis"), ":6379"),
                ("nuq-postgres", lambda: container_running("firecrawl_nuq-postgres"), ":5432"),
                ("searxng", lambda: check_http("http://localhost:8080"), ":8080"),
                ("neo4j", lambda: check_port("localhost", 7687), ":7687"),
                ("agent-memory-db", lambda: check_port("localhost", 5433), ":5433"),
                ("rabbitmq", lambda: container_running("firecrawl_rabbitmq"), ":5672"),
                ("playwright", lambda: container_running("firecrawl_playwright"), ":3000"),
                ("firecrawl api", lambda: check_port("localhost", 3002), ":3002"),
            ]

            all_up = True
            for name, check_fn, port in services:
                ok = check_fn()
                if ok:
                    status = STATUS_UP
                else:
                    status = STATUS_WAIT
                    all_up = False
                row(t, name, status, port)

            embed_ok = check_http("http://localhost:8900/health")
            live.update(t)

            if all_up and embed_ok:
                break
            time.sleep(1)
        else:
            failed = [name for name, check_fn, _ in services if not check_fn()]
            console.print("\n[red]Some services failed to start:[/red]")
            for f_name in failed:
                console.print(f"  [red]✗[/red] {f_name}")

    if failed:
        stop_all()
        sys.exit(1)

    # ── Final status ──
    console.print()
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column("")
    for name, _fn, port in services:
        row(t, name, STATUS_UP, port)
    console.print(t)
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

    console.print("\n[bold cyan]Pipeline starting...[/bold cyan]\n")

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

    try:
        with open(log_path, "a") as log_file:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
    except KeyboardInterrupt:
        pass

    proc.wait()
    if proc.returncode != 0:
        console.print(f"\n[red]Pipeline exited with code {proc.returncode}[/red]")
        stop_all()
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
