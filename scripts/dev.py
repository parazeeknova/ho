#!/usr/bin/env python3
"""Rich dev launcher: start llama-server + firecrawl with live status."""

import subprocess
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


def run(cmd: str, silent: bool = True) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=silent, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr)[:500]
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)


def check_http(url: str) -> bool:
    try:
        import urllib.request

        urllib.request.urlopen(url, timeout=3)
        return True
    except Exception:
        return False


def container_running(name: str) -> bool:
    code, _ = run(
        f"podman ps --filter name='{name}' --filter status=running --format '{{{{.Names}}}}'"
    )
    return code == 0


def main() -> None:
    console.clear()
    console.print(
        Panel.fit(
            "[bold cyan]ho[/bold cyan] — dev environment launcher",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # ── Cleanup ──
    with Live(
        Text("Cleaning up stale processes..."), refresh_per_second=4, console=console
    ) as live:
        run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
        run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
        run("killall llama-server 2>/dev/null", silent=True)
        time.sleep(1)
        live.update("[green]✓ Clean[/green]")

    # ── llama-server ──
    console.print("\n[bold]Starting llama-server...[/bold]")
    subprocess.Popen(
        [f"{PROJECT}/scripts/serve.sh"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(8):
        if check_http("http://localhost:8899/health"):
            break
        time.sleep(1)
    llama_ok = check_http("http://localhost:8899/health")

    # ── Firecrawl infra ──
    console.print("[bold]Starting firecrawl infrastructure...[/bold]")
    run(f"{DOCKER_COMPOSE} up -d redis playwright-service nuq-postgres", silent=False)

    # ── Rabbitmq ──
    console.print("[bold]Starting rabbitmq (podman)...[/bold]")
    run(
        "podman rm -f firecrawl_rabbitmq_1 2>/dev/null; "
        "podman run -d --name firecrawl_rabbitmq_1 "
        "--network firecrawl_default --network-alias rabbitmq "
        "--entrypoint /bin/bash rabbitmq:3-management "
        '-c "rm -f /var/lib/rabbitmq/.erlang.cookie; exec docker-entrypoint.sh rabbitmq-server"',
        silent=True,
    )
    time.sleep(3)

    # ── Firecrawl API ──
    console.print("[bold]Starting firecrawl api (takes ~20s)...[/bold]")
    run(f"{DOCKER_COMPOSE} up -d api", silent=False)
    for _ in range(20):
        if check_http("http://localhost:3002"):
            break
        time.sleep(2)
    api_ok = check_http("http://localhost:3002")

    # ── Status ──
    table = Table(title="Status", box=box.SIMPLE_HEAVY, border_style="cyan")
    table.add_column("Service", style="bold")
    table.add_column("Status")
    table.add_column("Port")

    def row(name: str, ok: bool, port: str) -> None:
        status = "[green]RUNNING[/green]" if ok else "[red]DOWN[/red]"
        table.add_row(name, status, port)

    row("llama-server", llama_ok, ":8899")
    row("firecrawl api", api_ok, ":3002")
    row("redis", container_running("firecrawl_redis"), ":6379")
    row("rabbitmq", container_running("firecrawl_rabbitmq"), ":5672")
    row("playwright", container_running("firecrawl_playwright"), ":3000")
    row("nuq-postgres", container_running("firecrawl_nuq-postgres"), ":5432")

    console.print()
    console.print(table)
    console.print("\n[dim]Press Ctrl+C to stop all services.[/dim]")

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
        run("killall llama-server 2>/dev/null", silent=True)


if __name__ == "__main__":
    main()
