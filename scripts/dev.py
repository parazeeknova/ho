#!/usr/bin/env python3
"""Rich dev launcher: start llama-server + firecrawl + agent-memory with live status."""

import contextlib
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

STATUS_DOWNLOADING = "[yellow]DOWNLOADING[/yellow]"
STATUS_INIT = "[yellow]INITIALIZING[/yellow]"
STATUS_UP = "[green]RUNNING[/green]"
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


def model_exists() -> bool:
    """Check if any GGUF file exists locally (in Models dir or HF cache)."""
    models_dir = Path.home() / "Models"
    if list(models_dir.glob("*.gguf")):
        return True
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        return bool(list(hf_cache.glob("models--*/snapshots/*/*.gguf")))
    return False


def container_running(name: str) -> bool:
    code, _ = run(
        f"podman ps --filter name='{name}' --filter status=running --format '{{{{.Names}}}}'"
    )
    return code == 0


def build_table() -> Table:
    t = Table(title="Status", box=box.SIMPLE_HEAVY, border_style="cyan")
    t.add_column("Service", style="bold")
    t.add_column("Status")
    t.add_column("Port")
    return t


def row(table: Table, name: str, status: str, port: str) -> None:
    table.add_row(name, status, port)


def status_for(ok: bool, downloading: bool = False, initializing: bool = False) -> str:
    if ok:
        return STATUS_UP
    if downloading:
        return STATUS_DOWNLOADING
    if initializing:
        return STATUS_INIT
    return STATUS_DOWN


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
    with Live(Table(), refresh_per_second=4, console=console) as live:
        t = build_table()
        row(t, "Cleanup", "[yellow]stopping stale...[/yellow]", "")
        live.update(t)

        run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
        run("podman rm -f firecrawl_rabbitmq_1 2>/dev/null", silent=True)
        run("killall llama-server 2>/dev/null", silent=True)
        time.sleep(1)

        t = build_table()
        row(t, "Cleanup", "[green]✓ Clean[/green]", "")
        live.update(t)

    # ── Start everything under a live-updating status table ─────────────

    llama_started = False
    llama_ok = embed_ok = False
    pgvector_ok = False
    pgvector_container = False

    infra_started = False
    api_started = False
    api_ok = False
    have_model = model_exists()

    deadman = 240  # max total startup seconds

    with Live(build_table(), refresh_per_second=2, console=console) as live:
        for elapsed in range(deadman):
            t = build_table()

            # ── Launch llama-server processes (first iteration only) ──
            if not llama_started:
                subprocess.Popen(
                    [sys.executable, f"{PROJECT}/scripts/serve.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                llama_started = True

            llama_ok = True
            embed_ok = check_http("http://localhost:8900/health")

            row(t, "GeneralCompute (gemma-4-31B-it)", STATUS_UP, "Cloud API")
            row(
                t,
                "llama-server (Embed)",
                status_for(
                    embed_ok,
                    downloading=not embed_ok and not have_model and elapsed < 120,
                    initializing=not embed_ok and have_model,
                ),
                ":8900",
            )

            # ── Launch agent-memory-db (first iteration only) ──
            if elapsed == 2:
                run(f"{DOCKER_COMPOSE} up -d agent-memory-db", silent=True)

            pgvector_container = container_running("agent-memory-db")
            pgvector_ok = pgvector_container and check_port("localhost", 5433)
            row(
                t,
                "agent-memory-db",
                status_for(pgvector_ok, initializing=pgvector_container and not pgvector_ok),
                ":5433",
            )

            # ── Launch firecrawl infra (first iteration only) ──
            if not infra_started:
                subprocess.Popen(
                    f"{DOCKER_COMPOSE} up -d redis playwright-service nuq-postgres searxng",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                infra_started = True

            row(t, "redis", status_for(container_running("firecrawl_redis")), ":6379")
            row(t, "nuq-postgres", status_for(container_running("firecrawl_nuq-postgres")), ":5432")
            row(t, "searxng", status_for(check_http("http://localhost:8080")), ":8080")

            # ── rabbitmq ──
            if elapsed == 3:
                run(
                    "podman rm -f firecrawl_rabbitmq_1 2>/dev/null; "
                    "podman run -d --name firecrawl_rabbitmq_1 "
                    "--network firecrawl_default --network-alias rabbitmq "
                    "--entrypoint /bin/bash rabbitmq:3-management "
                    '-c "rm -f /var/lib/rabbitmq/.erlang.cookie; '
                    'exec docker-entrypoint.sh rabbitmq-server"',
                    silent=True,
                )
            row(
                t,
                "rabbitmq",
                status_for(elapsed > 5 and container_running("firecrawl_rabbitmq")),
                ":5672",
            )

            row(t, "playwright", status_for(container_running("firecrawl_playwright")), ":3000")

            # ── firecrawl api ──
            if elapsed == 6:
                subprocess.Popen(
                    f"{DOCKER_COMPOSE} up -d api",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                api_started = True
            api_ok = check_http("http://localhost:3002")
            row(
                t,
                "firecrawl api",
                status_for(api_ok, initializing=api_started and not api_ok),
                ":3002",
            )

            live.update(t)

            # ── All up? Break early ──
            all_up = (
                llama_ok
                and embed_ok
                and pgvector_ok
                and api_ok
                and container_running("firecrawl_redis")
                and container_running("firecrawl_rabbitmq")
            )
            if all_up:
                break

            time.sleep(1)

    # ── Final status ──
    t = build_table()
    row(t, "GeneralCompute (gemma-4-31B-it)", STATUS_UP, "Cloud API")
    row(t, "llama-server (Embed)", status_for(embed_ok), ":8900")
    row(t, "agent-memory-db", status_for(pgvector_ok), ":5433")
    row(t, "firecrawl api", status_for(api_ok), ":3002")
    row(t, "redis", status_for(container_running("firecrawl_redis")), ":6379")
    row(t, "rabbitmq", status_for(container_running("firecrawl_rabbitmq")), ":5672")
    row(t, "playwright", status_for(container_running("firecrawl_playwright")), ":3000")
    row(t, "nuq-postgres", status_for(container_running("firecrawl_nuq-postgres")), ":5432")
    row(t, "searxng", status_for(check_http("http://localhost:8080")), ":8080")

    console.print()
    console.print(t)
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
