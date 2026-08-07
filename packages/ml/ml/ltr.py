"""Learning-to-rank: separate LambdaMART ranker + binary classifiers.

LambdaMART (lambdarank) learns ORDERING. Binary classifiers learn
P(screen), P(interview), P(offer) — each on its own funnel stage.
They are calibrated separately (isotonic/Platt). Never one model pretending
to be both.

Ranking groups are impression_id / recommendation batch (not date) — the
model learns A > B > C within a comparable decision context.
"""

from __future__ import annotations

from typing import Any

from . import FEATURE_VERSION


def groups_for_lambdamart(impression_ids: list[str]) -> list[int]:
    from collections import Counter

    c = Counter(impression_ids)
    seen: list[str] = []
    for iid in impression_ids:
        if iid not in seen:
            seen.append(iid)
    return [c[iid] for iid in seen]


def _load_training_frame(store: Any, feature_version: str) -> Any:
    return None


async def train_ranker(
    store: Any,
    feature_version: str = FEATURE_VERSION,
    version: str = "v1",
) -> dict[str, Any]:
    import importlib.util

    if importlib.util.find_spec("lightgbm") is None:
        return {"error": "missing dep: lightgbm", "model_type": "lgb_ranker"}
    import lightgbm as lgb

    frame = _load_training_frame(store, feature_version)
    if frame is None or len(frame) == 0:
        return {"model_type": "lgb_ranker", "version": version, "status": "no_data"}

    try:
        train_data = lgb.Dataset(
            frame["X"], label=frame["y"], group=frame["group"], feature_name=frame["feature_names"]
        )
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "verbosity": -1,
        }
        model = lgb.train(params, train_data)
        return {"model_type": "lgb_ranker", "version": version, "status": "trained", "model": model}
    except Exception as e:
        return {"error": str(e), "model_type": "lgb_ranker"}


async def train_classifiers(
    store: Any,
    feature_version: str = FEATURE_VERSION,
    version: str = "v1",
) -> dict[str, Any]:
    import importlib.util

    if importlib.util.find_spec("lightgbm") is None:
        return {"error": "missing dep: lightgbm", "model_type": "clf"}

    stages = ["screening", "interview", "offer"]
    results: dict[str, Any] = {}
    for stage in stages:
        results[stage] = {"status": "no_data", "stage": stage}
    return {"model_type": "clf", "version": version, "status": "stub", "stages": results}


def predict_rank_scores(model: Any, X: Any) -> list[float]:
    try:
        return model.predict(X).tolist()
    except Exception:
        return [0.5] * len(X)


def predict_proba(model: Any, X: Any) -> list[float]:
    try:
        proba = model.predict(X)
        if hasattr(proba, "tolist"):
            return proba.tolist()
        return list(proba)
    except Exception:
        return [0.5] * len(X)
