"""Model registry — versioned artifacts with dataset hash.

Every training run records model_id, model_type, version, feature_version,
training window, dataset_hash, metrics, and artifact path. The serving layer
loads the latest row where status='active'.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


async def register_model(
    store: Any,
    model_type: str,
    version: str,
    feature_version: str,
    training_start: str,
    training_end: str,
    dataset_hash: str,
    metrics: dict[str, Any],
    artifact_path: str,
) -> str:
    model_id = f"{model_type}_{version}_{int(time.time())}"
    async with store._pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO model_registry (
                model_id, model_type, version, feature_version,
                training_start, training_end, dataset_hash, metrics, artifact_path, status
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'active')
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
            artifact_path,
        )
    return model_id


async def get_active_model(store: Any, model_type: str) -> dict[str, Any] | None:
    async with store._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM model_registry WHERE model_type=$1 AND status='active' ORDER BY created_at DESC LIMIT 1",
            model_type,
        )
        return dict(row) if row else None


def dataset_hash_for_rows(rows: list[dict[str, Any]]) -> str:
    raw = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
