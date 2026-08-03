#!/usr/bin/env python3
"""One-command end-to-end loop: radar pipeline + autofill bridge + autofill worker.

Runs the whole hiring loop from a single invocation:

  1. starts the radar pipeline (master + worker processes) so jobs keep being
     discovered, gated and LLM-matched;
  2. every ``--bridge-interval`` seconds drains the accepted candidates into
     the autofill queue (see src/radar/engine/autofill_bridge.py) and prints
     the queue state;
  3. runs the autofill worker (OVERNIGHT_LOOP=true -> autosubmit) which claims
     those jobs and fills/submits them via the browser, marking each job
     applied (applied_at), failed (error_count/last_error) or deferred.

The loop is self-managing: crashed children are restarted (up to 5 times each)
and the run stops once no new jobs were bridged and the queue is empty for
``--idle-cycles`` consecutive bridge cycles (or after ``--max-minutes``).

Usage:
    uv run python scripts/loop.py
    uv run python scripts/loop.py --no-radar          # fill existing queue only
    uv run python scripts/loop.py --no-fill           # ingest only, no browser
    uv run python scripts/loop.py --max-minutes 240
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "autofill"))

from autofill.db import AutofillDB
from src.radar.engine.autofill_bridge import drain_once, print_summary, queue_balance

PROJECT = Path(__file__).resolve().parent.parent
REPO = PROJECT.parent
LOG_DIR = PROJECT / "logs"

# Same LLM throttle overrides run.py forces so workers blast through the corpus.
RADAR_ENV_OVERRIDES = {
    "LLM_QUEUE_RPM": "240",
    "LLM_QUEUE_MAX_IN_FLIGHT": "30",
    "LLM_QUEUE_TPM": "400000",
    "LLM_BUDGET_RADAR_RPM": "240",
    "LLM_BUDGET_RADAR_TPM": "400000",
}


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OVERNIGHT_LOOP", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    _paths = [str(PROJECT), str(REPO / "packages" / "autofill")]
    if env.get("PYTHONPATH"):
        _paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(_paths)
    _wd_env = PROJECT / ".watchdog.env"
    if _wd_env.exists():
        for _line in _wd_env.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                env.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    return env


class Child:
    def __init__(self, name: str, argv: list[str], env: dict[str, str]) -> None:
        self.name = name
        self.argv = argv
        self.env = env
        self.proc: subprocess.Popen[str] | None = None
        self.restarts = 0

    async def start(self) -> None:
        LOG_DIR.mkdir(exist_ok=True)
        self.proc = subprocess.Popen(
            self.argv,
            cwd=str(PROJECT),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print(f"[loop] started {self.name} (pid {self.proc.pid})", flush=True)

        async def _stream() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            with (
                open(LOG_DIR / f"{self.name}.log", "ab") as log_file,
                open(LOG_DIR / "loop.log", "ab") as loop_log,
            ):
                for line in self.proc.stdout:
                    sys.stdout.write(f"[{self.name}] {line}")
                    sys.stdout.flush()
                    log_file.write(line.encode())
                    log_file.flush()
                    loop_log.write(line.encode())
                    loop_log.flush()

        asyncio.get_running_loop().create_task(_stream())

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, sig: int = signal.SIGINT) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.send_signal(sig)
                print(f"[loop] sent interrupt to {self.name}", flush=True)
            except ProcessLookupError:
                pass

    async def wait(self, timeout: float = 45.0) -> None:
        if self.proc is not None:
            try:
                await asyncio.to_thread(self.proc.wait, timeout)
            except Exception:
                self.proc.kill()
                await asyncio.to_thread(self.proc.wait)


async def _spawn_radar(children: list[Child], workers: int) -> None:
    env = _base_env()
    env.update(RADAR_ENV_OVERRIDES)
    master = Child(
        "radar-master",
        [sys.executable, "-m", "src.radar.engine.orchestrator"],
        env,
    )
    await master.start()
    children.append(master)
    for idx in range(1, workers):
        w_env = env.copy()
        w_env["HO_WORKER_ONLY"] = "1"
        child = Child(
            f"radar-worker-{idx}",
            [sys.executable, "-m", "src.radar.engine.orchestrator"],
            w_env,
        )
        await child.start()
        children.append(child)


async def _spawn_worker(children: list[Child]) -> None:
    env = _base_env()
    child = Child(
        "autofill-worker",
        [sys.executable, "-m", "autofill.worker"],
        env,
    )
    await child.start()
    children.append(child)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-radar", action="store_true", help="Do not start the radar pipeline")
    parser.add_argument("--no-fill", action="store_true", help="Do not start the autofill worker")
    parser.add_argument("--radar-workers", type=int, default=2, help="Extra radar worker procs")
    parser.add_argument(
        "--bridge-interval", type=int, default=120, help="Seconds between bridge drains"
    )
    parser.add_argument("--bridge-batch", type=int, default=50, help="Max candidates per drain")
    parser.add_argument(
        "--idle-cycles", type=int, default=3, help="Stop after N bridge cycles with nothing to do"
    )
    parser.add_argument("--max-minutes", type=int, default=0, help="Hard stop after N minutes")
    args = parser.parse_args()

    children: list[Child] = []
    started_at = time.monotonic()
    try:
        if not args.no_radar:
            await _spawn_radar(children, args.radar_workers)
        if not args.no_fill:
            await _spawn_worker(children)
        if not children:
            print("[loop] nothing to run (--no-radar and --no-fill both given)")
            return
    except Exception as exc:
        print(f"[loop] failed to start children: {exc}")
        for child in children:
            child.stop()
        return

    db = await AutofillDB.create()
    idle_cycles = 0
    try:
        while True:
            await asyncio.sleep(args.bridge_interval)

            for child in children:
                if not child.is_alive():
                    if child.restarts >= 5:
                        print(f"[loop] {child.name} crashed and exceeded restarts", flush=True)
                        continue
                    child.restarts += 1
                    print(f"[loop] restarting {child.name} (attempt {child.restarts})", flush=True)
                    await child.start()

            enqueued = await drain_once(db, args.bridge_batch)
            balance = await queue_balance(db)
            print(
                f"[loop] bridge +{enqueued} (accepted in corpus {balance['drainable_accepted']}) "
                f"| applied {balance['applied']}, open {balance['open']}, "
                f"deferred {balance['deferred']}, skipped {balance['skipped']}, "
                f"errored {balance['errored']}",
                flush=True,
            )

            if enqueued == 0 and balance["open"] == 0:
                idle_cycles += 1
                if idle_cycles >= args.idle_cycles:
                    print(f"[loop] queue idle for {idle_cycles} cycles; finishing", flush=True)
                    break
            else:
                idle_cycles = 0

            if args.max_minutes and (time.monotonic() - started_at) / 60 >= args.max_minutes:
                print("[loop] --max-minutes reached; finishing", flush=True)
                break
    except KeyboardInterrupt:
        print("[loop] interrupted; shutting down...", flush=True)
    finally:
        for child in children:
            child.stop()
        for child in children:
            await child.wait()
        print("[loop] final queue state:", flush=True)
        print_summary(await queue_balance(db))
        await db.close()
        print("[loop] done.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
