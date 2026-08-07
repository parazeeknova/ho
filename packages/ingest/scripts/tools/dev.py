#!/usr/bin/env python3
"""Rich dev launcher: start llama-server + ingest infra (searxng, neo4j,
agent-memory-db) with live status. Firecrawl stack removed — not started."""

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

PROJECT = Path(__file__).resolve().parents[2]
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

    # Cleanup
    with Live(Table(), refresh_per_second=4, console=console) as live:
        t = build_table()
        row(t, "Cleanup", "[yellow]stopping stale...[/yellow]", "")
        live.update(t)

        run(f"{DOCKER_COMPOSE} down 2>/dev/null", silent=True)
        run("killall llama-server 2>/dev/null", silent=True)
        time.sleep(1)

        t = build_table()
        row(t, "Cleanup", "[green]✓ Clean[/green]", "")
        live.update(t)

    # Start everything under a live-updating status table

    llama_started = False
    embed_ok = False
    pgvector_ok = False
    pgvector_container = False
    neo4j_ok = searxng_ok = False
    have_model = model_exists()

    deadman = 240  # max total startup seconds

    with Live(build_table(), refresh_per_second=2, console=console) as live:
        for elapsed in range(deadman):
            t = build_table()

            # Launch llama-server process (first iteration only)
            if not llama_started:
                subprocess.Popen(
                    [sys.executable, f"{PROJECT}/scripts/serve.py"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                llama_started = True

            embed_ok = check_http("http://localhost:8900/health")
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

            # Launch the compose services (first iteration only)
            if elapsed == 2:
                subprocess.Popen(
                    f"{DOCKER_COMPOSE} up -d searxng neo4j agent-memory-db",
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            searxng_ok = check_http("http://localhost:8080")
            row(t, "searxng", status_for(searxng_ok), ":8080")
            neo4j_ok = check_port("localhost", 7687)
            row(t, "neo4j", status_for(neo4j_ok), ":7687")

            pgvector_container = container_running("agent-memory-db")
            pgvector_ok = pgvector_container and check_port("localhost", 5433)
            row(
                t,
                "agent-memory-db",
                status_for(pgvector_ok, initializing=pgvector_container and not pgvector_ok),
                ":5433",
            )

            live.update(t)

            # All up? Break early
            if embed_ok and searxng_ok and neo4j_ok and pgvector_ok:
                break

            time.sleep(1)

    # Final status
    t = build_table()
    row(t, "llama-server (Embed)", status_for(embed_ok), ":8900")
    row(t, "searxng", status_for(searxng_ok), ":8080")
    row(t, "neo4j", status_for(neo4j_ok), ":7687")
    row(t, "agent-memory-db", status_for(pgvector_ok), ":5433")
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
