"""Periodic ML training hook for the loop driver.

The radar pipeline emits decision_events (job_ranked impressions). Once enough
outcome data accumulates (applications -> screening -> interview -> offer),
this trains the LightGBM LambdaMART ranker + funnel classifiers offline and
registers them in model_registry.

It is idempotent and guarded:
  - trains at most once every ML_TRAIN_INTERVAL_SECS (default 6h),
  - skips when there is no supervised signal yet (no positive labels /
    censored rows), so it never spams "no_data" model rows,
  - registers as status='candidate' at most; the first trained model is
    promoted to 'active' so `bun status` shows a real model, later models
    stay candidate until promoted by an explicit eval/promotion step.

Run in-process by the loop (not a child process): training is CPU-bound and
short-lived on the small datasets early on; when data grows, it can move to a
background worker without changing the API.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml import FEATURE_VERSION
from ml.src.config import get_ml_config
from ml.src.ranking.dataset import (
    build_dataset,
    fetch_all_job_events,
    fetch_impressions,
    uncensored,
)
from ml.src.ranking.model_registry import (
    get_active_model,
    register_model,
    transition_model,
)
from src.logging import get_logger

logger = get_logger("ml_trainer")

# How often the loop attempts training (not a per-run hammer).
TRAIN_INTERVAL_SECS = int(__import__("os").environ.get("ML_TRAIN_INTERVAL_SECS", "21600"))
# Minimum number of labeled (uncensored, non-zero-label) rows to bother.
MIN_LABELED_ROWS = int(__import__("os").environ.get("ML_MIN_LABELED_ROWS", "20"))
VERSION = "v1"

_last_trained: float = 0.0
_last_attempt: float = 0.0


def _persist_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def _persist_calibrator(calibrator: Any, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, str(path))


async def maybe_train(store: Any) -> str:
    """Train + register if it's time and there's signal. Returns a status string."""
    global _last_trained, _last_attempt
    now = time.monotonic()
    if now - _last_attempt < 60:
        return "skipped (recent attempt)"
    _last_attempt = now
    if now - _last_trained < TRAIN_INTERVAL_SECS:
        return f"skipped (trained {int(TRAIN_INTERVAL_SECS / 60)}m ago)"

    if store is None:
        return "skipped (no store)"

    cfg = get_ml_config()
    maturity_days = getattr(cfg.training, "label_maturity_days", 7.0)

    # Gate on data sufficiency BEFORE calling the trainer: building the dataset
    # is cheap, training is not. We need impressions AND at least some labeled
    # (uncensored, non-zero) rows to produce a non-"no_data" model.
    impressions = await fetch_impressions(store, FEATURE_VERSION)
    if not impressions:
        return "skipped (no impressions yet)"
    job_events = await fetch_all_job_events(store)
    ds = build_dataset(impressions, job_events, label_maturity_days=maturity_days)
    if ds.is_empty():
        return "skipped (empty dataset)"
    uncensored_rows = uncensored(ds.rows)
    n_pos = sum(1 for r in uncensored_rows if r.ordinal_relevance > 0)
    n_labeled = len(uncensored_rows)
    if n_labeled < MIN_LABELED_ROWS or n_pos < 2:
        return (
            f"skipped (not enough signal: {n_labeled} labeled, {n_pos} positive; "
            "need outcomes to mature)"
        )

    return await _train(store, impressions, job_events, cfg)


async def _train(store: Any, impressions: list[dict[str, Any]], job_events: dict, cfg: Any) -> str:
    global _last_trained
    from ml.src.ranking.ltr import train_classifiers, train_ranker

    models_dir = Path(cfg.artifact_dir) / "models"
    calibrators_dir = Path(cfg.artifact_dir) / "calibrators"
    models_dir.mkdir(parents=True, exist_ok=True)
    calibrators_dir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(UTC).isoformat()

    dhash = ""
    from ml.src.ranking.model_registry import dataset_hash_for_rows

    dhash = dataset_hash_for_rows([dict(r) for r in impressions])[:12]

    trained_any = False
    notes: list[str] = []

    # Ranker
    try:
        r = await train_ranker(store, FEATURE_VERSION, version=VERSION)
        if r.get("status") == "trained" and "model" in r:
            model = r.pop("model")
            artifact = str(models_dir / f"lgb_ranker_{VERSION}_{dhash}.txt")
            _persist_model(model, Path(artifact))
            mid = await register_model(
                store,
                "lgb_ranker",
                VERSION,
                FEATURE_VERSION,
                now_iso,
                now_iso,
                dhash,
                r.get("metrics", {}),
                model_artifact_path=artifact,
                status="candidate",
            )
            notes.append(f"ranker {mid} ({r.get('metrics', {}).get('val_ndcg@10', 'n/a')})")
            trained_any = True
        else:
            notes.append(f"ranker: {r.get('status', 'no_data')}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"ranker error: {e}")

    # Classifiers
    try:
        c = await train_classifiers(store, FEATURE_VERSION, version=VERSION)
        for stage, meta in (c.get("stages") or {}).items():
            if not isinstance(meta, dict):
                continue
            if meta.get("status") != "trained":
                notes.append(f"{stage}: {meta.get('status', 'no_data')}")
                continue
            model = meta.pop("model", None)
            cal = meta.pop("calibrator", None)
            model_artifact = str(models_dir / f"{stage}_{VERSION}_{dhash}.txt")
            _persist_model(model, Path(model_artifact))
            cal_artifact = ""
            if cal is not None:
                cal_artifact = str(calibrators_dir / f"{stage}_{VERSION}_{dhash}.joblib")
                _persist_calibrator(cal, Path(cal_artifact))
            mid = await register_model(
                store,
                f"clf_{stage}",
                VERSION,
                FEATURE_VERSION,
                now_iso,
                now_iso,
                dhash,
                meta.get("metrics", {}),
                model_artifact_path=model_artifact,
                calibrator_artifact_path=cal_artifact,
                status="candidate",
            )
            notes.append(f"{stage} {mid}")
            trained_any = True
    except Exception as e:  # noqa: BLE001
        notes.append(f"classifiers error: {e}")

    if trained_any:
        _last_trained = time.monotonic()
        # First real model -> active so the pipeline starts using it. Later
        # models stay candidate until an explicit promotion step compares them.
        active = await get_active_model(store, "lgb_ranker")
        if active is None:
            newest = await _newest_candidate(store, "lgb_ranker")
            if newest:
                await transition_model(store, newest, "active", require_current="candidate")
                notes.append("-> promoted first ranker to active")
        logger.info("ML training completed", models=", ".join(notes))
        return "trained: " + "; ".join(notes)

    logger.info("ML training attempted, no usable model", notes=", ".join(notes))
    return "no_model: " + "; ".join(notes)


async def _newest_candidate(store: Any, model_type: str) -> str | None:
    async with store._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT model_id FROM model_registry WHERE model_type=$1 AND status='candidate' "
            "ORDER BY created_at DESC LIMIT 1",
            model_type,
        )
        return row["model_id"] if row else None


async def run_maybe_train(store: Any) -> str:
    """Async entry for the loop driver; returns a short status string."""
    try:
        return await maybe_train(store)
    except Exception as e:  # noqa: BLE001
        logger.warning("ML training hook failed", exception=str(e))
        return f"error: {e}"
