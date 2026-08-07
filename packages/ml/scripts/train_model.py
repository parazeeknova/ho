"""Offline trainer: Postgres → dataset → Parquet → LightGBM (ranker + classifiers).

Real training pipeline (not a stub):
  1. Build the dataset from decision_events (impressions + outcome labels).
  2. Temporal split (train / val / test) — never random.
  3. Train LambdaMART ranker + binary funnel classifiers.
  4. Calibrate classifiers (isotonic).
  5. Snapshot dataset to Parquet (dataset_hash registered).
  6. Register models in model_registry.

Usage:
  uv run python packages/ml/scripts/train_model.py --type ranker --version v1
  uv run python packages/ml/scripts/train_model.py --type clf --version v1
  uv run python packages/ml/scripts/train_model.py --type all --version v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from ml.config import get_ml_config
from ml.dataset import (
    build_dataset,
    fetch_all_job_events,
    fetch_impressions,
)
from ml.model_registry import dataset_hash_for_rows, register_model


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["ranker", "clf", "all"], default="all")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    from ml import FEATURE_VERSION
    from ml.ltr import train_classifiers, train_ranker

    try:
        from src.memory.pgvector_store import MemoryStore
    except ImportError:
        store = None
    else:
        store = await MemoryStore.create()

    if store is None:
        print("No store — nothing to train on yet. Run a few sweeps first.")
        return

    # Build the dataset once (shared by ranker + classifiers).
    impressions = await fetch_impressions(store, FEATURE_VERSION)
    job_events = await fetch_all_job_events(store)
    ds = build_dataset(impressions, job_events)
    print(f"Impressions: {len(ds.impression_ids())} | ranked rows: {len(ds.rows)}")
    if ds.is_empty():
        print("No data — run sweeps so job_ranked events + outcomes accumulate first.")
        await store.close()
        return

    # Snapshot dataset to Parquet for reproducibility.
    cfg = get_ml_config()
    ds_path = Path(cfg.dataset_dir)
    ds_path.mkdir(parents=True, exist_ok=True)
    dhash = dataset_hash_for_rows([r.as_dict() for r in ds.rows])
    parquet_path = str(ds_path / f"dataset_{dhash}.parquet")
    ds.to_parquet(parquet_path)
    print(f"Dataset snapshot: {parquet_path} (rows={len(ds.rows)}, hash={dhash})")

    results: dict[str, Any] = {}
    if args.type in ("ranker", "all"):
        results["ranker"] = await train_ranker(store, FEATURE_VERSION, version=args.version)
        r = results["ranker"]
        print(f"Ranker: {r.get('status')} | {json.dumps(r.get('metrics', {}))}")
        if r.get("status") == "trained" and "model" in r:
            r.pop("model", None)  # don't serialize the lgb object
            await register_model(
                store,
                "lgb_ranker",
                args.version,
                FEATURE_VERSION,
                "now",
                "now",
                dhash,
                r.get("metrics", {}),
                parquet_path,
            )
            print(f"Registered lgb_ranker {args.version} (dataset {dhash})")
    if args.type in ("clf", "all"):
        results["clf"] = await train_classifiers(store, FEATURE_VERSION, version=args.version)
        c = results["clf"]
        print(f"Classifiers: {c.get('status')}")
        for stage, meta in (c.get("stages") or {}).items():
            if isinstance(meta, dict):
                print(f"  {stage}: {meta.get('status')} {json.dumps(meta.get('metrics', {}))}")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
