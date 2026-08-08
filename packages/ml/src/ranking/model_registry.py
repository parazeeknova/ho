"""Model registry — versioned artifacts with dataset hash and promotion states.

Every training run records model_id, model_type, version, feature_version,
training window, dataset_hash, metrics, artifact paths, and status.

Promotion lifecycle (P0 fix — never auto-activate a freshly trained model):
    trained -> validated -> shadow -> candidate -> active -> retired

The training script registers a new model as status='candidate' at most; the
promotion step compares shadow metrics against the current active model and
only then marks it active.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

# Promotion state machine. A model may only move to the NEXT state (or retire).
PROMOTION_ORDER = ["trained", "validated", "shadow", "candidate", "active", "retired"]


async def register_model(
    store: Any,
    model_type: str,
    version: str,
    feature_version: str,
    training_start: str,
    training_end: str,
    dataset_hash: str,
    metrics: dict[str, Any],
    model_artifact_path: str = "",
    calibrator_artifact_path: str = "",
    status: str = "candidate",
) -> str:
    """Register a trained model as 'candidate' (never 'active' directly).

    ``model_artifact_path`` and ``calibrator_artifact_path`` point at the
    persisted model/calibrator files (joblib / LightGBM text), NOT the dataset
    Parquet — the serving layer must be able to load a real model artifact.
    """
    assert status in PROMOTION_ORDER, f"invalid status {status}"
    model_id = f"{model_type}_{version}_{int(time.time())}"
    async with store._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO model_registry (
                model_id, model_type, version, feature_version,
                training_start, training_end, dataset_hash, metrics, status,
                artifact_path, calibrator_artifact_path
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (model_id) DO NOTHING
            """,
            model_id,
            model_type,
            version,
            feature_version,
            training_start,
            training_end,
            dataset_hash,
            json.dumps(metrics),
            status,
            model_artifact_path,
            calibrator_artifact_path,
        )
    return model_id


async def get_active_model(store: Any, model_type: str) -> dict[str, Any] | None:
    async with store._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM model_registry WHERE model_type=$1 AND status='active' ORDER BY created_at DESC LIMIT 1",
            model_type,
        )
        return dict(row) if row else None


async def get_model(store: Any, model_id: str) -> dict[str, Any] | None:
    async with store._pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM model_registry WHERE model_id=$1", model_id)
        return dict(row) if row else None


async def transition_model(
    store: Any, model_id: str, to_status: str, require_current: str | None = None
) -> bool:
    """Move a model to the next promotion state, guarding against invalid jumps.

    ``require_current`` optionally pins the expected current state so a
    concurrent promotion cannot race.
    """
    assert to_status in PROMOTION_ORDER, f"invalid status {to_status}"
    async with store._pool.acquire() as conn:
        row = await conn.fetchrow("SELECT status FROM model_registry WHERE model_id=$1", model_id)
        if not row:
            return False
        cur = row["status"]
        if require_current is not None and cur != require_current:
            return False
        if cur == to_status:
            return True
        # Allow forward moves or retire-from-anywhere.
        if to_status != "retired" and cur not in PROMOTION_ORDER:
            return False
        if to_status != "retired":
            cur_idx = PROMOTION_ORDER.index(cur)
            to_idx = PROMOTION_ORDER.index(to_status)
            if to_idx <= cur_idx:
                return False
        await conn.execute(
            "UPDATE model_registry SET status=$2 WHERE model_id=$1", model_id, to_status
        )
        return True


def dataset_hash_for_rows(rows: list[dict[str, Any]]) -> str:
    raw = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
