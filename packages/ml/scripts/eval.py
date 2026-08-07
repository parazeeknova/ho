"""Offline evaluation: nDCG@K, precision@K, application/screening/interview/offer rates.

Usage:
  uv run python -m ml.scripts.eval --model ranker_v1 --k 10
"""

from __future__ import annotations

import argparse
import asyncio


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ranker_v1")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    print(f"Eval {args.model} @K={args.k} — stub (needs trained model + labeled holdout)")


if __name__ == "__main__":
    asyncio.run(main())
