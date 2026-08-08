"""Offline evaluation — temporal holdout, ranking + calibration + funnel metrics."""

from __future__ import annotations

import math
from typing import Any


def expected_calibration_error(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> float:
    """ECE: |accuracy(bin) - mean_prob(bin)| weighted by bin size. 0 = perfect."""
    n = len(y_true)
    if n == 0:
        return 0.0
    bins = [0.0] * n_bins
    counts = [0] * n_bins
    correct = [0] * n_bins
    for y, p in zip(y_true, y_prob):
        b = min(int(p * n_bins), n_bins - 1)
        counts[b] += 1
        correct[b] += y
        bins[b] += p
    total = 0.0
    for i in range(n_bins):
        if counts[i] == 0:
            continue
        acc = correct[i] / counts[i]
        conf = bins[i] / counts[i]
        total += abs(acc - conf) * counts[i] / n
    return total


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / n


def reliability_curve(
    y_true: list[int], y_prob: list[float], n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin [confidence, accuracy, count] for a reliability diagram."""
    out: list[dict[str, float]] = []
    counts = [0] * n_bins
    bins = [0.0] * n_bins
    correct = [0] * n_bins
    for y, p in zip(y_true, y_prob):
        b = min(int(p * n_bins), n_bins - 1)
        counts[b] += 1
        correct[b] += y
        bins[b] += p
    for i in range(n_bins):
        if counts[i] == 0:
            continue
        out.append(
            {
                "bin": i / n_bins,
                "confidence": round(bins[i] / counts[i], 4),
                "accuracy": round(correct[i] / counts[i], 4),
                "count": counts[i],
            }
        )
    return out


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


def promotion_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    require_beats: tuple[str, ...] = ("ndcg@10", "precision@10", "interview_rate"),
    calib_worsen_ok: float = 0.02,
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    """Decide whether a shadow model earns promotion over the current baseline.

    The reviewer's P1 rule: the candidate must beat the baseline on the
    business metrics (nDCG@10, precision@10, interview rate) while calibration
    error must not worsen beyond ``calib_worsen_ok``. Never promote on
    val_ndcg@10 alone.

    Returns a verdict dict with `promote` (bool) and per-metric deltas.
    """
    verdicts: dict[str, str] = {}
    wins = 0
    for m in require_beats:
        b = baseline.get(m, 0.0)
        c = candidate.get(m, 0.0)
        delta = c - b
        verdicts[m] = (
            "candidate"
            if delta > min_improvement
            else ("tie" if abs(delta) <= min_improvement else "baseline")
        )
        if delta > min_improvement:
            wins += 1
    # Calibration must not worsen beyond tolerance.
    b_ece = baseline.get("ece", 0.0)
    c_ece = candidate.get("ece", baseline.get("ece", 0.0))
    calib_ok = c_ece <= b_ece + calib_worsen_ok
    promote = wins >= len(require_beats) and calib_ok
    return {
        "promote": promote,
        "wins": wins,
        "required": len(require_beats),
        "metrics": verdicts,
        "calibration_ok": calib_ok,
        "calibration_delta": round(c_ece - b_ece, 4),
        "rule": (
            "candidate must beat baseline on ALL of "
            f"{', '.join(require_beats)} AND not worsen calibration "
            f"(ECE +{calib_worsen_ok})"
        ),
    }


def temporal_split(
    events: list[dict[str, Any]], train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events = sorted(events, key=lambda e: e.get("timestamp", 0))
    n = len(events)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return events[:t1], events[t1:t2], events[t2:]
