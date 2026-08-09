#!/usr/bin/env python3
"""One-command setup of the user memory base for a fresh checkout.

Checks Postgres + the embedding server (starting it if needed), indexes the
resume into resume_embeddings, builds the persona into persona_embeddings +
resume_summary (grilling interactively when persona.json is missing), and prints
a summary of what is now in memory.

Usage:
    uv run python scripts/init_memory.py                       # full setup
    uv run python scripts/init_memory.py --no-resume           # persona only
    uv run python scripts/init_memory.py --grill               # force re-grill
    uv run python scripts/init_memory.py --resume-url <url>    # explicit resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent  # packages/autofill
REPO = ROOT.parent.parent  # repo root
for _p in (REPO, REPO / "packages" / "ingest", REPO / "packages"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

load_dotenv()
os.environ["LOG_LEVEL"] = "WARNING"  # quiet JSON log spam in setup scripts

import ux  # noqa: E402
from src.configuration import get_config  # noqa: E402
from src.logging import get_logger  # noqa: E402
from src.memory.pgvector_store import MemoryStore  # noqa: E402

logger = get_logger("init_memory")

# Sentinel returned by _input() when the user interrupts / closes stdin
# (Ctrl+C / Ctrl+D) — callers can treat it as "user aborted, quit cleanly".
_INPUT_ABORT = object()


def _input(prompt: str = "") -> str:
    """input() that converts Ctrl+C/Ctrl+D into the _INPUT_ABORT sentinel so
    an interactive prompt can never crash init-memory with a traceback."""
    try:
        return input(prompt)
    except EOFError, KeyboardInterrupt:
        raise SystemExit(_INPUT_ABORT) from None


def _embed_health_url() -> str:
    return get_config().embed.url.rsplit("/v1", 1)[0] + "/health"


async def embed_server_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
            resp = await client.get(_embed_health_url())
            return resp.status_code == 200
    except Exception:
        return False


async def ensure_embed_server(auto_start: bool) -> bool:
    if await embed_server_ready():
        ux.chip("ok", "Embedding server up on :8900 (Qwen3-Embedding-0.6B)")
        return True
    ux.chip("err", f"Embedding server DOWN at {_embed_health_url()}")
    if not shutil.which("llama-server"):
        ux.bullet("Install llama.cpp (the llama-server binary) first, e.g. via:")
        ux.bullet("  brew install llama.cpp        # macos", style="cyan")
        ux.bullet(
            "  pip install llama-cpp-python  # alternative, see scripts/serve.py",
            style="cyan",
        )
        return False
    if not auto_start:
        ux.bullet("Start it with `uv run python scripts/serve.py` and re-run init_memory.")
        return False
    ux.bullet("Start it now via scripts/serve.py? [Y/n]", style="white")
    answer = _input("").strip().lower()
    if answer not in ("", "y", "yes"):
        return False
    log = Path("/tmp/opencode") if Path("/tmp/opencode").exists() else Path("/tmp")
    log = log / "embed_server.log"
    with open(log, "a") as out:
        proc = subprocess.Popen(
            [sys.executable, str(REPO / "packages" / "ingest" / "scripts" / "serve.py")],
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    ux.chip("info", f"Spawned embed server (pid {proc.pid}); waiting for health...")
    with ux.console.status("Waiting for embedding server...", spinner="dots"):
        for _ in range(60):
            if await embed_server_ready():
                ux.chip("ok", "Embedding server is up.")
                return True
            time.sleep(1)
    ux.chip("err", f"Embed server did not become healthy; check {log}")
    return False


async def index_resume(resume_url: str | None, resume_path: str | None) -> None:
    cmd = [sys.executable, str(REPO / "packages" / "ingest" / "scripts" / "index_resume.py")]
    if resume_url:
        cmd += ["--url", resume_url]
    if resume_path:
        cmd += ["--path", resume_path]
    try:
        result = subprocess.run(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        # Ctrl+C during the resume download/index: abort init-memory cleanly
        # instead of the child printing a traceback and us "continuing".
        raise SystemExit(_INPUT_ABORT) from None
    if result.returncode != 0:
        if result.returncode == 130:
            # The resume indexer was interrupted (Ctrl+C): abort the whole run.
            raise SystemExit(_INPUT_ABORT)
        ux.chip(
            "warn",
            "Resume indexing failed; continuing (fix RESUME_URL/RESUME_PATH and re-run).",
        )


def _compose_file() -> Path:
    return REPO / "packages" / "ingest" / "docker-compose.yaml"


async def _start_postgres() -> bool:
    """Auto-start the agent-memory Postgres + redis containers and wait for
    Postgres.

    The app database is the ``agent-memory-db`` service (host port 5433,
    pgvector image, persistent volume). ``redis`` (port 6379) is also brought
    up because the LLM governor uses it as the shared token/RPM budget store —
    without it every LLM call logs "Redis budget unavailable, local-only".
    """
    import shutil

    if not shutil.which("docker"):
        return False
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "docker",
                "compose",
                "-f",
                str(_compose_file()),
                "up",
                "-d",
                "agent-memory-db",
                "redis",
            ],
            cwd=REPO,
            capture_output=True,
            timeout=180,
        )
        if result.returncode != 0:
            logger.warning(
                "docker compose up failed",
                stderr=(result.stderr or b"").decode()[-400:],
            )
            return False
    except Exception as e:
        logger.warning("docker compose up failed", error=str(e))
        return False
    for _ in range(45):
        try:
            store = await MemoryStore.create()
            await store.close()
            return True
        except Exception:
            await asyncio.sleep(1)
    return False


async def _ensure_redis() -> None:
    """Confirm redis (the LLM governor budget store) is reachable; warn only."""
    import redis.asyncio as aioredis

    url = os.environ.get("LLM_BUDGET_REDIS_URL", "redis://127.0.0.1:6379/1")
    try:
        r = aioredis.from_url(url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        ux.chip("ok", f"Redis up ({url.split('@')[-1]})")
    except Exception as e:
        ux.chip("warn", f"Redis unavailable ({e}); LLM governor will run local-only")


async def run_script(name: str, *extra: str) -> int:
    script = REPO / "packages" / "autofill" / "scripts" / name
    if not script.exists():
        script = REPO / "packages" / "ingest" / "scripts" / name
    return subprocess.run([sys.executable, str(script), *extra], cwd=REPO).returncode


def has_env(source: str) -> bool:
    return bool(os.environ.get(source))


def _missing_persona_items(persona_json: Path) -> set[str]:
    """Wizard questions and identity contact fields not yet answered.

    Lets init-memory notice that the grill gained new questions (e.g. the
    identity facts tier, the twitter field) instead of silently rebuilding
    from the old file.
    """
    try:
        from grill_persona import (  # type: ignore[import-not-found]
            CONTACT_FIELDS,
            CORE_QUESTIONS,
            _missing_generated_questions,
        )

        data = json.loads(persona_json.read_text())
    except Exception:
        return set()
    missing: set[str] = set()
    answered = {(a.get("category") or "").strip() for a in data.get("answers", [])}
    expected = {category for category, _ in CORE_QUESTIONS}
    missing |= expected - answered
    # Previously-LLM-generated questions that were never answered (LLM was down
    # when this persona was built). Each is labeled by its question text.
    missing |= set(_missing_generated_questions(data))
    identity = data.get("identity") or {}
    for field in CONTACT_FIELDS:
        if not (identity.get(field) or "").strip():
            missing.add(field)
    return missing


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build the user memory base for a fresh checkout.")
    parser.add_argument("--no-resume", action="store_true", help="Skip resume indexing")
    parser.add_argument("--resume-url", help="Resume download URL (overrides RESUME_URL)")
    parser.add_argument("--resume-path", help="Local resume file path (overrides RESUME_PATH)")
    parser.add_argument("--no-grill", action="store_true", help="Never run the interactive wizard")
    parser.add_argument(
        "--grill",
        action="store_true",
        help="Force the interactive wizard even if persona.json exists",
    )
    parser.add_argument(
        "--no-embed-start",
        action="store_true",
        help="Never auto-spawn the embed server",
    )
    args = parser.parse_args()

    ux.chip("info", "USER MEMORY INITIALIZATION  -  resume + persona RAG base")

    # 1. Postgres
    ux.section(1, 4, "Postgres")
    try:
        store = await MemoryStore.create()
    except Exception as e:
        ux.chip("err", f"Could not connect to Postgres: {e}")
        ux.bullet("Starting agent-memory-db + redis (docker compose) automatically...")
        if await _start_postgres():
            ux.chip("ok", "Postgres started automatically")
            store = await MemoryStore.create()
        else:
            ux.chip("err", "Could not start Postgres automatically")
            ux.bullet("Start it manually:")
            ux.bullet(
                "  docker compose -f "
                f"{_compose_file().relative_to(REPO)} up -d agent-memory-db redis",
                style="cyan",
            )
            ux.bullet(
                "  (First-time setup only, after the volume is initialized:)\n"
                "  PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres "
                "-d agent_memory -f packages/ingest/scripts/sql/init-pgvector.sql",
                style="cyan",
            )
            sys.exit(1)
    await store.close()
    ux.chip("ok", "Postgres connected (localhost:5433/agent_memory)")
    await _ensure_redis()

    # 2. Embedding server
    ux.section(2, 4, "Embedding server")
    if not await ensure_embed_server(auto_start=not args.no_embed_start):
        ux.chip("err", "Aborting: memory build needs the embedding server.")
        sys.exit(1)

    # 3. Resume -> resume_embeddings
    ux.section(3, 4, "Resume")
    if args.no_resume:
        ux.chip("info", "Skipping resume indexing (--no-resume).")
    elif args.resume_url or args.resume_path or has_env("RESUME_URL") or has_env("RESUME_PATH"):
        await index_resume(args.resume_url, args.resume_path)
    else:
        ux.chip(
            "warn",
            "No resume source found; set RESUME_URL/RESUME_PATH in .env "
            "or pass --resume-url/--resume-path.",
        )

    # 4. Persona -> persona_embeddings + resume_summary in persona.json
    ux.section(4, 4, "Persona")
    persona_json = REPO / "data" / "persona.json"
    if args.grill:
        await run_script("grill_persona.py")
    elif persona_json.exists():
        missing = _missing_persona_items(persona_json)
        if missing:
            ux.bullet(
                "persona.json exists but is missing: "
                + ", ".join(sorted(missing))
                + ". Run the interactive grill to answer them? [Y/n]",
                style="white",
            )
            answer = _input("").strip().lower()
            if answer in ("", "y", "yes"):
                await run_script("grill_persona.py")
            else:
                ux.chip("info", "Skipping grill; persona.json left as-is.")
        else:
            # persona.json is complete. Offer a clear 3-way choice: rebuild
            # memory from the existing persona (y), run the interactive wizard
            # to build a NEW persona from scratch (g), or skip (n).
            ux.bullet(
                "persona.json exists (complete). "
                "[Y] rebuild memory / [G] build a new one (wizard) / [N] skip",
                style="white",
            )
            answer = _input("").strip().lower()
            if answer in ("g", "grill", "new", "fresh"):
                ux.chip("info", "Launching interactive wizard to build a new persona...")
                await run_script("grill_persona.py")
            elif answer in ("", "y", "yes"):
                await run_script("build_persona.py")
            else:
                ux.chip("info", "Skipping persona rebuild.")
    elif args.no_grill:
        ux.chip(
            "warn",
            "persona.json missing and --no-grill given; "
            "run `uv run python scripts/grill_persona.py` later.",
        )
    else:
        await run_script("grill_persona.py")

    # Summary
    ux.divider()
    store = await MemoryStore.create()
    try:
        resume_count = await store.chunk_count()
        persona_count = await store.persona_chunk_count()
    finally:
        await store.close()
    rows = [
        ("resume_embeddings", f"{resume_count} chunks"),
        ("persona_embeddings", f"{persona_count} chunks"),
    ]
    if persona_json.exists():
        persona = json.loads(persona_json.read_text())
        rows.append(("persona.json", f"{len(persona.get('answers', []))} answers"))
        if persona.get("resume_summary"):
            rows.append(("resume_summary", "stored in persona.json"))
    else:
        rows.append(("persona.json", "MISSING (run grill_persona.py)"))
    ux.summary_table("MEMORY SUMMARY", rows)

    ux.next_steps(
        [
            "uv run python -m autofill.src.core.cli apply <greenhouse-url>",
            "uv run python -m src.radar.engine.orchestrator",
        ]
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit as _se:
        if getattr(_se, "code", None) is _INPUT_ABORT:
            # Ctrl+C / Ctrl+D at a prompt: confirm before quitting.
            import sys as _sys

            try:
                ans = _input("\n[ho] Quit init-memory? (y/N) ").strip().lower()
            except SystemExit:
                ans = "y"
            if ans in ("y", "yes"):
                print(
                    "[ho] Exiting. Memory unchanged unless a step already completed.",
                    flush=True,
                )
            else:
                print("[ho] Continuing setup...", flush=True)
                asyncio.run(main())
            _sys.exit(0)
        raise
    except KeyboardInterrupt:
        # Ctrl+C: confirm before quitting so an accidental press doesn't abort
        # mid-setup, then exit cleanly (no traceback).
        import sys as _sys

        try:
            ans = _input("\n[ho] Quit init-memory? (y/N) ").strip().lower()
        except SystemExit:
            ans = "y"
        if ans in ("y", "yes"):
            print("[ho] Exiting. Memory unchanged unless a step already completed.", flush=True)
        else:
            print("[ho] Continuing setup...", flush=True)
            asyncio.run(main())
        _sys.exit(0)
