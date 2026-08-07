"""Offline evaluation — temporal holdout, ranking + calibration + funnel metrics."""

from __future__ import annotations

import math
from typing import Any


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[float], ideal: list[float], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(ideal, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevances: list[float], k: int, threshold: float = 0.5) -> float:
    top = relevances[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r >= threshold) / len(top)


def evaluate_ranking(y_true: list[float], y_pred: list[float], k: int = 10) -> dict[str, float]:
    order = sorted(range(len(y_true)), key=lambda i: y_pred[i], reverse=True)
    ranked_true = [y_true[i] for i in order]
    return {
        f"ndcg@{k}": ndcg_at_k(ranked_true, y_true, k),
        f"precision@{k}": precision_at_k(ranked_true, k),
        "mrr": next((1.0 / (i + 1) for i, r in enumerate(ranked_true) if r > 0.5), 0.0),
    }


def funnel_metrics(events: list[dict[str, Any]]) -> dict[str, float]:
    total = len([e for e in events if e.get("event_type") == "job_ranked"])
    applied = len([e for e in events if e.get("event_type") == "job_applied"])
    screening = len([e for e in events if e.get("event_type") in ("screening", "screening_email")])
    interview = len([e for e in events if e.get("event_type") == "interview"])
    offer = len([e for e in events if e.get("event_type") == "offer"])
    denom = max(applied, 1)
    return {
        "application_rate": applied / max(total, 1),
        "screening_rate": screening / denom,
        "interview_rate": interview / denom,
        "offer_rate": offer / denom,
        "interviews_per_100": interview / denom * 100,
        "offers_per_100": offer / denom * 100,
    }


def temporal_split(
    events: list[dict[str, Any]], train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events = sorted(events, key=lambda e: e.get("timestamp", 0))
    n = len(events)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return events[:t1], events[t1:t2], events[t2:]
