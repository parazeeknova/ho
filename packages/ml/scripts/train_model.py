"""Offline trainer: Postgres → Parquet snapshot → LightGBM (ranker + classifiers).

Usage:
  uv run python -m ml.scripts.train_model --type ranker --version v1
  uv run python -m ml.scripts.train_model --type clf --version v1
"""

from __future__ import annotations

import argparse
import asyncio
import json

from ml.config import get_ml_config


async def _fetch_events(store, window_days: int) -> list[dict]:
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM decision_events
            WHERE created_at > NOW() - make_interval(days => $1)
            ORDER BY created_at ASC
            """,
            window_days,
        )
        return [dict(r) for r in rows]


async def _build_dataset(store, feature_version: str) -> list[dict]:
    """Join decision_events with candidate/job snapshots + graph features into
    training rows. Temporal split (train/val/test by time) applied downstream."""
    cfg = get_ml_config()
    events = await _fetch_events(store, cfg.training.train_window_days)
    out: list[dict] = []
    for e in events:
        features = e.get("features") or {}
        # Only rows with feature_version match (avoid leakage from future models)
        if e.get("feature_version") != feature_version and feature_version:
            continue
        if not features:
            continue
        out.append(
            {
                "job_id": e["job_id"],
                "event_type": e["event_type"],
                "impression_id": e.get("impression_id"),
                "timestamp": e["created_at"],
                "features": features,
                "reward": e.get("reward"),
            }
        )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["ranker", "clf"], default="ranker")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()

    from ml import FEATURE_VERSION
    from ml.model_registry import dataset_hash_for_rows, register_model

    try:
        from src.memory.pgvector_store import MemoryStore
    except ImportError:
        store = None
    else:
        store = await MemoryStore.create()

    if store is None:
        print("No store — nothing to train on yet. Run a few sweeps first.")
        return

    rows = await _build_dataset(store, FEATURE_VERSION)
    print(f"Training rows: {len(rows)}")
    if not rows:
        await store.close()
        return

    dhash = dataset_hash_for_rows(rows)
    print(f"dataset_hash={dhash}")

    from ml.ltr import train_classifiers, train_ranker

    if args.type == "ranker":
        result = await train_ranker(store, FEATURE_VERSION, version=args.version)
    else:
        result = await train_classifiers(store, FEATURE_VERSION, version=args.version)
    print(
        json.dumps({k: (str(v)[:100] if k == "model" else v) for k, v in result.items()}, indent=2)
    )
    if result.get("status") == "trained":
        await register_model(
            store,
            "lgb_ranker" if args.type == "ranker" else "clf",
            args.version,
            FEATURE_VERSION,
            "now",
            "now",
            dhash,
            {"rows": len(rows)},
            "",
        )
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
