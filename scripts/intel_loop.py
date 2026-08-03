"""Nightly vector-intel refresh loop: every REFRESH_EVERY seconds, rebuild
the vector-intel exports (recommendations.json/csv) from the current
obs_embeddings. Runs detached so the pipeline is untouched.

    setsid nohup env PYTHONPATH=$PWD uv run python3 scripts/intel_loop.py > logs/intel_loop.out 2>&1 &
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

REFRESH_EVERY = 1800  # 30 min

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def run_intel() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT
    env.setdefault(
        "LD_LIBRARY_PATH", "/nix/store/61a1nwx3w6rqyaisj5rn1sal1981apm7-zlib-1.3.2/lib"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            os.path.join(PROJECT, "scripts", "vector_intel.py"),
            "--write",
            "--top-k",
            "8",
            cwd=PROJECT,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=600)
    except Exception as e:
        print(f"intel refresh error: {e}", flush=True)


async def main() -> None:
    while True:
        t0 = time.monotonic()
        try:
            await run_intel()
            print(
                f"intel refreshed in {time.monotonic()-t0:.1f}s @ {time.strftime('%H:%M:%S')}",
                flush=True,
            )
        except Exception as e:
            print(f"intel refresh failed: {e}", flush=True)
        await asyncio.sleep(REFRESH_EVERY)


if __name__ == "__main__":
    asyncio.run(main())
