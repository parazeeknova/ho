"""Learning-to-rank: separate LambdaMART ranker + binary classifiers.

LambdaMART (lambdarank) learns ORDERING (grouped by impression_id).
Binary classifiers learn P(applied), P(screening), P(interview), P(offer)
per funnel stage. They are calibrated separately (isotonic). Never one model
pretending to be both.

Training uses the real dataset builder (ml.dataset) — not a stub.
"""

from __future__ import annotations

from typing import Any

from . import FEATURE_VERSION
from .dataset import (
    Dataset,
    build_dataset,
    classifier_matrices,
    fetch_all_job_events,
    fetch_impressions,
    lgbm_matrices,
    temporal_split,
)


async def _load_dataset(store: Any, feature_version: str) -> Dataset:
    """Real dataset loader from decision_events (impressions + outcomes)."""
    impressions = await fetch_impressions(store, feature_version)
    job_events = await fetch_all_job_events(store)
    return build_dataset(impressions, job_events)


def _check_lgb() -> bool:
    import importlib.util

    return importlib.util.find_spec("lightgbm") is not None


def train_lambdamart(X, y, group, feature_names):
    import lightgbm as lgb

    from .dataset import to_np

    Xn, yn, group = to_np(X, y, group)
    ds = lgb.Dataset(Xn, label=yn, group=group, feature_name=feature_names)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "verbosity": -1,
    }
    return lgb.train(params, ds, num_boost_round=200)


def train_binary(X, y, feature_names):
    import lightgbm as lgb

    from .dataset import to_np

    Xn, yn = to_np(X, y)
    ds = lgb.Dataset(Xn, label=yn, feature_name=feature_names)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "verbosity": -1,
    }
    return lgb.train(params, ds, num_boost_round=200)


def calibrate_binary(model, X, y):
    from sklearn.isotonic import IsotonicRegression

    scores = model.predict(X)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y)
    return iso


async def train_ranker(
    store: Any,
    feature_version: str = FEATURE_VERSION,
    version: str = "v1",
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    if not _check_lgb():
        return {"error": "missing dep: lightgbm", "model_type": "lgb_ranker"}

    ds = await _load_dataset(store, feature_version)
    if ds.is_empty():
        return {"model_type": "lgb_ranker", "version": version, "status": "no_data"}

    # Temporal split: train on oldest, hold out newest for honest evaluation.
    train_rows, val_rows, _ = temporal_split(ds.rows, 0.7, val_fraction)
    if not train_rows:
        return {"model_type": "lgb_ranker", "version": version, "status": "no_data"}

    X, y, group = lgbm_matrices(train_rows)
    if len(set(y)) < 2:
        return {
            "model_type": "lgb_ranker",
            "version": version,
            "status": "no_data",
            "reason": "no positive labels",
        }

    try:
        model = train_lambdamart(X, y, group, list(NUMERIC_FEATURE_NAMES))
        # Evaluate on held-out val impressions (nDCG) if any.
        metrics: dict[str, Any] = {"trained_on_rows": len(train_rows)}
        if val_rows:
            Xv, yv, gv = lgbm_matrices(val_rows)
            preds = model.predict(Xv)

            # per-impression nDCG@10
            scores = _grouped_ndcg(yv, preds, gv, 10)
            metrics["val_ndcg@10"] = round(scores, 4)
        return {
            "model_type": "lgb_ranker",
            "version": version,
            "status": "trained",
            "model": model,
            "metrics": metrics,
            "feature_names": list(NUMERIC_FEATURE_NAMES),
            "trained_rows": len(train_rows),
        }
    except Exception as e:
        return {"error": str(e), "model_type": "lgb_ranker"}


async def train_classifiers(
    store: Any,
    feature_version: str = FEATURE_VERSION,
    version: str = "v1",
    val_fraction: float = 0.15,
) -> dict[str, Any]:
    if not _check_lgb():
        return {"error": "missing dep: lightgbm", "model_type": "clf"}

    ds = await _load_dataset(store, feature_version)
    if ds.is_empty():
        return {"model_type": "clf", "version": version, "status": "no_data"}

    train_rows, val_rows, _ = temporal_split(ds.rows, 0.7, val_fraction)
    if not train_rows:
        return {"model_type": "clf", "version": version, "status": "no_data"}

    stages = ["applied", "screening", "interview", "offer"]
    results: dict[str, Any] = {}
    trained_any = False
    for stage in stages:
        X, y = classifier_matrices(train_rows, stage)
        if sum(y) == 0:
            results[stage] = {"status": "no_positive", "stage": stage}
            continue
        try:
            model = train_binary(X, y, list(NUMERIC_FEATURE_NAMES))
            calibrator = calibrate_binary(model, X, y)
            metrics: dict[str, Any] = {"trained_rows": len(train_rows), "positives": sum(y)}
            if val_rows:
                Xv, yv = classifier_matrices(val_rows, stage)
                preds = model.predict(Xv)
                cal_preds = calibrator.predict(preds)
                import contextlib

                from sklearn.metrics import log_loss, roc_auc_score

                with contextlib.suppress(Exception):
                    metrics["val_auc"] = round(roc_auc_score(yv, preds), 4)
                with contextlib.suppress(Exception):
                    metrics["val_logloss"] = round(log_loss(yv, cal_preds), 4)
            results[stage] = {
                "status": "trained",
                "model": model,
                "calibrator": calibrator,
                "metrics": metrics,
                "feature_names": list(NUMERIC_FEATURE_NAMES),
            }
            trained_any = True
        except Exception as e:
            results[stage] = {"status": "error", "error": str(e), "stage": stage}

    return {
        "model_type": "clf",
        "version": version,
        "status": "trained" if trained_any else "no_data",
        "stages": results,
    }


def _grouped_ndcg(y, preds, groups, k: int) -> float:
    """Mean nDCG@k across groups (impressions)."""
    from .evaluation import ndcg_at_k

    total = 0.0
    cnt = 0
    idx = 0
    for g in groups:
        g = max(g, 1)
        ys = y[idx : idx + g]
        ps = preds[idx : idx + g]
        order = sorted(range(len(ys)), key=lambda i: ps[i], reverse=True)
        ranked = [ys[i] for i in order]
        total += ndcg_at_k(ranked, ys, k)
        cnt += 1
        idx += g
    return total / cnt if cnt else 0.0


NUMERIC_FEATURE_NAMES = [
    "semantic_fit",
    "matching_skills_count",
    "missing_skills_count",
    "skill_coverage",
    "exp_gap",
    "source_confidence",
    "freshness_score",
    "salary_fit",
    "location_fit",
    "is_remote",
    "sponsors_visa",
    "underdog_score",
    "funding_stage_ordinal",
    "funding_recency_days",
    "osint_count",
    "company_pagerank",
    "company_betweenness",
    "founder_hiring_signal",
    "tech_overlap",
    "graph_hiring_likelihood",
    "hour_of_day",
]


def predict_rank_scores(model: Any, X: Any) -> list[float]:
    try:
        preds = model.predict(X)
        return preds.tolist() if hasattr(preds, "tolist") else list(preds)
    except Exception:
        return [0.5] * len(X) if hasattr(X, "__len__") else [0.5]


def predict_proba(model: Any, X: Any) -> list[float]:
    try:
        proba = model.predict(X)
        return proba.tolist() if hasattr(proba, "tolist") else list(proba)
    except Exception:
        return [0.5] * len(X) if hasattr(X, "__len__") else [0.5]
