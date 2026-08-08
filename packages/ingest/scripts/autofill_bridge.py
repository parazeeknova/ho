#!/usr/bin/env python3
"""CLI for the radar -> autofill bridge (see src/radar/engine/autofill_bridge.py).

Usage:
    uv run python scripts/autofill_bridge.py                # drain once + summary
    uv run python scripts/autofill_bridge.py --drain 500
    uv run python scripts/autofill_bridge.py --summary
    uv run python scripts/autofill_bridge.py --watch --interval 300
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "autofill"))

from autofill.src.core.db import AutofillDB
from src.radar.engine.autofill_bridge import drain_once, print_summary, queue_balance


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--drain",
        type=int,
        nargs="?",
        const=100,
        default=None,
        help="Drain up to N accepted candidates (default 100)",
    )
    parser.add_argument("--summary", action="store_true", help="Only print queue summary")
    parser.add_argument(
        "--watch", action="store_true", help="Drain in a loop every --interval seconds"
    )
    parser.add_argument("--interval", type=int, default=300, help="Watch interval in seconds")
    args = parser.parse_args()

    db = await AutofillDB.create()
    try:
        if args.watch:
            while True:
                await drain_once(db, args.drain or 100)
                print_summary(await queue_balance(db))
                await asyncio.sleep(args.interval)
        elif not args.summary:
            await drain_once(db, args.drain or 100)
            print_summary(await queue_balance(db))
        else:
            print_summary(await queue_balance(db))
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
