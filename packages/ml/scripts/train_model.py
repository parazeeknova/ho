"""Offline trainer: Postgres → dataset → Parquet → LightGBM (ranker + classifiers).

Real training pipeline (not a stub):
  1. Build the dataset from decision_events (impressions + outcome labels),
     with outcome maturity/censoring (label_maturity_days).
  2. Impression-level temporal split (train / val / test) — never random.
  3. Train LambdaMART ranker + binary funnel classifiers.
  4. Calibrate classifiers (isotonic) on VALIDATION predictions — never train.
  5. Persist model + calibrator artifacts to disk (joblib / LightGBM text).
  6. Snapshot dataset to Parquet (dataset_hash registered).
  7. Register models in model_registry as status='candidate' (P0: never active).

Usage:
  uv run python packages/ml/scripts/train_model.py --type ranker --version v1
  uv run python packages/ml/scripts/train_model.py --type clf --version v1
  uv run python packages/ml/scripts/train_model.py --type all --version v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.src.config import get_ml_config
from ml.src.ranking.dataset import (
    build_dataset,
    fetch_all_job_events,
    fetch_impressions,
)
from ml.src.ranking.model_registry import dataset_hash_for_rows, register_model


def _persist_model(model: Any, path: Path) -> None:
    """Persist a LightGBM Booster to native text (small, loadable, versioned)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    print(f"  model artifact -> {path}")


def _persist_calibrator(calibrator: Any, path: Path) -> None:
    """Persist a fitted sklearn calibrator via joblib."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, str(path))
    print(f"  calibrator artifact -> {path}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["ranker", "clf", "all"], default="all")
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--status",
        default="candidate",
        choices=["trained", "validated", "shadow", "candidate"],
        help="Initial registry status (default candidate; never auto-active).",
    )
    ap.add_argument(
        "--maturity-days",
        type=float,
        default=None,
        help="Outcome maturity window (days). Rows younger than this are censored.",
    )
    args = ap.parse_args()

    from ml import FEATURE_VERSION
    from ml.src.ranking.ltr import train_classifiers, train_ranker

    try:
        from src.memory.pgvector_store import MemoryStore
    except ImportError:
        store = None
    else:
        store = await MemoryStore.create()

    if store is None:
        print("No store — nothing to train on yet. Run a few sweeps first.")
        return

    cfg = get_ml_config()
    maturity_days = (
        args.maturity_days
        if args.maturity_days is not None
        else getattr(cfg.training, "label_maturity_days", 7.0)
    )

    # Build the dataset once (shared by ranker + classifiers).
    impressions = await fetch_impressions(store, FEATURE_VERSION)
    job_events = await fetch_all_job_events(store)
    ds = build_dataset(impressions, job_events, label_maturity_days=maturity_days)
    print(
        f"Impressions: {len(set(ds.impression_ids()))} | ranked rows: {len(ds.rows)}"
        f" | censored: {sum(1 for r in ds.rows if r.censored)} (maturity {maturity_days}d)"
    )
    if ds.is_empty():
        print("No data — run sweeps so job_ranked events + outcomes accumulate first.")
        await store.close()
        return

    # Snapshot dataset to Parquet for reproducibility.
    ds_path = Path(cfg.dataset_dir)
    ds_path.mkdir(parents=True, exist_ok=True)
    dhash = dataset_hash_for_rows([r.as_dict() for r in ds.rows])
    parquet_path = str(ds_path / f"dataset_{dhash}.parquet")
    ds.to_parquet(parquet_path)
    print(f"Dataset snapshot: {parquet_path} (rows={len(ds.rows)}, hash={dhash})")

    models_dir = Path(cfg.artifact_dir) / "models"
    calibrators_dir = Path(cfg.artifact_dir) / "calibrators"
    models_dir.mkdir(parents=True, exist_ok=True)
    calibrators_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(UTC).isoformat()

    results: dict[str, Any] = {}
    if args.type in ("ranker", "all"):
        results["ranker"] = await train_ranker(store, FEATURE_VERSION, version=args.version)
        r = results["ranker"]
        print(f"Ranker: {r.get('status')} | {json.dumps(r.get('metrics', {}))}")
        if r.get("status") == "trained" and "model" in r:
            model = r.pop("model", None)
            artifact = str(models_dir / f"lgb_ranker_{args.version}_{dhash}.txt")
            _persist_model(model, Path(artifact))
            model_id = await register_model(
                store,
                "lgb_ranker",
                args.version,
                FEATURE_VERSION,
                now_iso,
                now_iso,
                dhash,
                r.get("metrics", {}),
                model_artifact_path=artifact,
                status=args.status,
            )
            print(f"Registered lgb_ranker {args.version} as '{args.status}' ({model_id})")
    if args.type in ("clf", "all"):
        results["clf"] = await train_classifiers(store, FEATURE_VERSION, version=args.version)
        c = results["clf"]
        print(f"Classifiers: {c.get('status')}")
        for stage, meta in (c.get("stages") or {}).items():
            if not isinstance(meta, dict):
                continue
            print(f"  {stage}: {meta.get('status')} {json.dumps(meta.get('metrics', {}))}")
            if meta.get("status") != "trained":
                continue
            model = meta.pop("model", None)
            cal = meta.pop("calibrator", None)
            model_artifact = str(models_dir / f"{stage}_{args.version}_{dhash}.txt")
            _persist_model(model, Path(model_artifact))
            cal_artifact = ""
            if cal is not None:
                cal_artifact = str(calibrators_dir / f"{stage}_{args.version}_{dhash}.joblib")
                _persist_calibrator(cal, Path(cal_artifact))
            await register_model(
                store,
                f"clf_{stage}",
                args.version,
                FEATURE_VERSION,
                now_iso,
                now_iso,
                dhash,
                meta.get("metrics", {}),
                model_artifact_path=model_artifact,
                calibrator_artifact_path=cal_artifact,
                status=args.status,
            )
            print(f"  Registered clf_{stage} {args.version} as '{args.status}'")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
