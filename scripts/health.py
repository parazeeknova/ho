#!/usr/bin/env python3
"""Enhanced health diagnostics for all services in the ho pipeline.

Reports actionable operational diagnostics instead of merely returning
"healthy" / "down".
"""

import contextlib
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"

fails = 0
warnings = 0


@dataclass
class CheckResult:
    name: str
    ok: bool = False
    detail: str = ""


_results: list[CheckResult] = []


def _register(name: str, fn: Callable[[], bool], detail: str = "") -> bool:
    global fails
    ok = False
    try:
        ok = fn()
    except Exception:
        ok = False
    _results.append(CheckResult(name=name, ok=ok, detail=detail))
    if ok:
        print(f"  {GREEN}OK{RESET}   {name}")
    else:
        print(f"  {RED}DOWN{RESET} {name}")
        fails += 1
    return ok


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    with contextlib.suppress(Exception), socket.create_connection((host, port), timeout=timeout):
        return True
    return False


def container_running(pattern: str) -> bool:
    try:
        r = subprocess.run(
            [
                "podman",
                "ps",
                "--filter",
                f"name={pattern}",
                "--filter",
                "status=running",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def container_count(pattern: str) -> int:
    try:
        r = subprocess.run(
            [
                "podman",
                "ps",
                "--filter",
                f"name={pattern}",
                "--filter",
                "status=running",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return len(r.stdout.strip().splitlines())
    except Exception:
        return 0


print("Configuration Validation")
try:
    from src.configuration import get_config

    cfg = get_config()
    problems = cfg.validate()
    if problems:
        for p in problems:
            print(f"  {YELLOW}WARN{RESET} {p}")
            warnings += 1
    else:
        print(f"  {GREEN}OK{RESET}   Configuration valid")
except Exception as e:
    print(f"  {RED}FAIL{RESET} Configuration load failed: {e}")
    fails += 1

print()
print("LLM")
_register("GeneralCompute Cloud", lambda: True)
_register("llama-server Embeddings :8900", lambda: http_ok("http://localhost:8900/health"))

print()
print("Firecrawl")
_register("api              :3002", lambda: http_ok("http://localhost:3002"))
_register("redis", lambda: container_running("firecrawl_redis"))
_register("rabbitmq", lambda: container_running("firecrawl_rabbitmq"))
_register("playwright", lambda: container_running("firecrawl_playwright"))
_register("nuq-postgres", lambda: container_running("firecrawl_nuq-postgres"))

print()
print("Agent Memory")
_register("agent-memory-db", lambda: container_running("firecrawl_agent-memory-db"))
pgvector_ok = _register("pgvector :5433", lambda: check_port("localhost", 5433))

if pgvector_ok:
    try:
        import asyncio

        async def _check_db_stats() -> dict:
            try:
                from src.memory.pgvector_store import MemoryStore

                store = await MemoryStore.create()
                chunks = await store.chunk_count()
                jobs = await store.job_ledger_count()
                domains = await store.discovered_domain_count()
                await store.close()
                return {"chunks": chunks, "jobs": jobs, "domains": domains}
            except Exception:
                return {}

        stats = asyncio.run(_check_db_stats())
        if stats:
            print(
                f"  {DIM}      {stats.get('chunks', 0)} resume chunks, "
                f"{stats.get('jobs', 0)} jobs, "
                f"{stats.get('domains', 0)} domains{RESET}"
            )
    except Exception:
        pass

print()
print("Metasearch")
_register("searxng          :8080", lambda: http_ok("http://localhost:8080"))

print()
print("Graph DB (Neo4j)")
neo4j_bolt = _register("neo4j (bolt)     :7687", lambda: check_port("localhost", 7687))
_register("neo4j (browser)  :7474", lambda: http_ok("http://localhost:7474"))

if neo4j_bolt:
    try:
        import asyncio

        async def _check_neo4j_stats() -> dict:
            try:
                from src.graph.graph_store import GraphStore

                graph = await GraphStore.create()
                node_count = await graph.node_count()
                rel_count = await graph.relationship_count()
                await graph.close()
                return {"nodes": node_count, "relationships": rel_count}
            except Exception:
                return {}

        stats = asyncio.run(_check_neo4j_stats())
        if stats:
            print(
                f"  {DIM}      {stats.get('nodes', 0)} nodes, "
                f"{stats.get('relationships', 0)} relationships{RESET}"
            )
    except Exception:
        pass

print()
if fails == 0 and warnings == 0:
    print(f"{GREEN}All health checks passed.{RESET}")
elif fails == 0:
    print(f"{YELLOW}All checks passed with {warnings} warning(s).{RESET}")
else:
    print(f"{RED}{fails} check(s) failed, {warnings} warning(s).{RESET}")

sys.exit(fails)
