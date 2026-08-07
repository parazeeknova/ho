"""Calibration — isotonic / Platt for P(outcome) classifiers."""

from __future__ import annotations

import numpy as np


def isotonic_calibrate(y_true: list[int], y_score: list[float]) -> Any:
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_score, y_true)
    return ir


def platt_calibrate(y_true: list[int], y_score: list[float]) -> Any:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(np.array(y_score).reshape(-1, 1), y_true)
    return lr


def calibration_curve(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = [(bins[i] <= p < bins[i + 1]) for p in y_prob]
        cnt = sum(mask)
        if cnt == 0:
            continue
        actual = sum(y for y, m in zip(y_true, mask) if m) / cnt
        pred = sum(p for p, m in zip(y_prob, mask) if m) / cnt
        out.append(
            {"bin": i, "predicted": round(pred, 3), "actual": round(actual, 3), "count": cnt}
        )
    return out
