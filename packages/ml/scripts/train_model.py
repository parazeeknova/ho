"""Offline trainer: Postgres → Parquet snapshot → LightGBM.

Usage:
  uv run python -m ml.scripts.train_model --type ranker --version v1
  uv run python -m ml.scripts.train_model --type clf --version v1
"""

from __future__ import annotations

import argparse
import asyncio


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["ranker", "clf"], default="ranker")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    print(f"Training {args.type} {args.version} — stub (collect data first)")

    # Real training:
    # 1. Query decision_events with temporal windows (train/val/test by time)
    # 2. Join candidate_snapshots + job_snapshots + graph features
    # 3. Build feature matrix via ml.features
    # 4. For ranker: group by impression_id, LambdaMART
    # 5. For clf: binary per funnel stage, calibrate, save
    # 6. Snapshot dataset to artifacts/datasets/dataset_{hash}.parquet
    # 7. Register in model_registry


if __name__ == "__main__":
    asyncio.run(main())
