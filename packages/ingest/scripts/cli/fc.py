"""Ingest infra + Neo4j container manager.

The Firecrawl stack (api / playwright-service / rabbitmq / redis /
nuq-postgres) was removed — this manages only what the pipeline still needs:
searxng, torproxy, neo4j, agent-memory-db.

Usage:
    uv run python scripts/cli/fc.py up                        # start infra
    uv run python scripts/cli/fc.py down                      # stop infra
    uv run python scripts/cli/fc.py logs                      # tail compose logs
    uv run python scripts/cli/fc.py status                    # compose ps
    uv run python scripts/cli/fc.py clean                     # down -v
    uv run python scripts/cli/fc.py dev-down                  # infra + llama-server
    uv run python scripts/cli/fc.py tor-up
    uv run python scripts/cli/fc.py graph-up|graph-stop|graph-reset|graph-shell
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(PROJECT / "docker-compose.yaml")]
SERVICES = ["searxng", "torproxy", "neo4j", "agent-memory-db"]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT, check=check)


def up() -> None:
    _run([*COMPOSE, "up", "-d", *SERVICES])


def down() -> None:
    _run([*COMPOSE, "down"])


def logs() -> None:
    _run([*COMPOSE, "logs", "-f"])


def status() -> None:
    _run([*COMPOSE, "ps"])


def clean() -> None:
    _run([*COMPOSE, "down", "-v"])
    print("All container volumes cleared.")


def dev_down() -> None:
    down()
    subprocess.run(["killall", "llama-server"], check=False)


def tor_up() -> None:
    _run([*COMPOSE, "up", "-d", "torproxy"])


def graph_up() -> None:
    _run([*COMPOSE, "up", "-d", "neo4j"])


def graph_stop() -> None:
    _run([*COMPOSE, "stop", "neo4j"])


def graph_reset() -> None:
    _run([*COMPOSE, "down", "-v", "neo4j"])
    _run([*COMPOSE, "up", "-d", "neo4j"])


def graph_shell() -> None:
    _run([*COMPOSE, "exec", "neo4j", "cypher-shell", "-u", "neo4j", "-p", "password"])


ACTIONS = {
    "up": up,
    "down": down,
    "logs": logs,
    "status": status,
    "clean": clean,
    "dev-down": dev_down,
    "tor-up": tor_up,
    "graph-up": graph_up,
    "graph-stop": graph_stop,
    "graph-reset": graph_reset,
    "graph-shell": graph_shell,
}


def main() -> int:
    action = sys.argv[1] if len(sys.argv) > 1 else "up"
    fn = ACTIONS.get(action)
    if fn is None:
        print(f"unknown action '{action}'; choices: {', '.join(ACTIONS)}")
        return 1
    fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
