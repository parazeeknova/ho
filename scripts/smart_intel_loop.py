"""Periodic local smart-intel refresh loop.

Runs smart_intel.py every REFRESH_EVERY seconds on the local box (NOT the
relic). Kept alive by the watchdog. Exports intel/smart_intel.json+csv.

    setsid nohup env PYTHONPATH=$PWD uv run python3 scripts/smart_intel_loop.py \
        > logs/smart_intel_loop.out 2>&1 &
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

REFRESH_EVERY = 1800  # 30 min
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def run_once() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT
    env.setdefault(
        "LD_LIBRARY_PATH", "/nix/store/61a1nwx3w6rqyaisj5rn1sal1981apm7-zlib-1.3.2/lib"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            os.path.join(PROJECT, "scripts", "smart_intel.py"),
            "--write",
            cwd=PROJECT,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=600)
    except Exception as e:
        print(f"smart_intel refresh error: {e}", flush=True)


async def main() -> None:
    while True:
        t0 = time.monotonic()
        try:
            await run_once()
            print(
                f"smart_intel refreshed in {time.monotonic()-t0:.1f}s @ "
                f"{time.strftime('%H:%M:%S')}",
                flush=True,
            )
        except Exception as e:
            print(f"smart_intel refresh failed: {e}", flush=True)
        await asyncio.sleep(REFRESH_EVERY)


if __name__ == "__main__":
    asyncio.run(main())
