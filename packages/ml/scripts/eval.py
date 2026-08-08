"""Offline evaluation: baseline (current) ranker vs LTR shadow on historical impressions.

Metrics (business ground truth, not just nDCG):
  nDCG@10, precision@10, MRR
  application rate, screening rate, interview rate, offer rate
  interviews/100 applications, offers/100 applications
  calibration: ECE, Brier, reliability curve (on classifier probas)

Temporal holdout: impressions are split at the impression level (never rows),
the newest impressions form the test set. Both the current (production order)
ranker and the LTR model are scored on the test impressions, and a
promotion_decision enforces the reviewer's rule: LTR must beat baseline on
nDCG@10 AND precision@10 AND interview rate while calibration must not worsen.

Usage:
  uv run python packages/ml/scripts/eval.py --k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from ml import FEATURE_VERSION
from ml.src.config import get_ml_config
from ml.src.ranking.dataset import (
    build_dataset,
    fetch_all_job_events,
    fetch_impressions,
    lgbm_matrices,
    temporal_split,
    uncensored,
)
from ml.src.ranking.evaluation import (
    evaluate_ranking,
    promotion_decision,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    try:
        from src.memory.pgvector_store import MemoryStore
    except ImportError:
        store = None
    else:
        store = await MemoryStore.create()

    if store is None:
        print("No store.")
        return

    cfg = get_ml_config()
    maturity_days = getattr(cfg.training, "label_maturity_days", 7.0)

    impressions = await fetch_impressions(store, FEATURE_VERSION)
    job_events = await fetch_all_job_events(store)
    ds = build_dataset(impressions, job_events, label_maturity_days=maturity_days)
    if ds.is_empty():
        print("No impressions yet — run sweeps first.")
        await store.close()
        return

    train_rows, val_rows, test_rows = temporal_split(ds.rows, 0.7, 0.15)
    print(f"rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    report: dict[str, Any] = {
        "n": len(test_rows),
        "k": args.k,
        "maturity_days": maturity_days,
        "censored": sum(1 for r in ds.rows if r.censored),
        "current": {},
        "ltr": {},
        "funnel": {},
        "promotion": {},
    }

    # Funnel metrics across all rows (not just test) so rates aren't tiny.
    report["funnel"] = _funnel_from_rows(ds.rows)

    if test_rows:
        # Current policy ordering (by stored rank asc) per impression.
        current_order = sorted(
            test_rows,
            key=lambda r: (r.impression_id, r.raw_features.get("rank", 99) or 99),
        )
        current_rel = [float(r.ordinal_relevance) for r in current_order]
        report["current"] = evaluate_ranking(current_rel, current_rel, args.k)
        report["current"]["label_positive"] = sum(1 for r in current_rel if r > 0)

        # LTR shadow: train on train rows (ordinal labels), score test rows.
        from ml.src.ranking.ltr import _check_lgb, train_lambdamart

        try:
            if not _check_lgb():
                report["ltr"] = {"status": "lightgbm_missing"}
            else:
                Xtr, ytr, gtr = lgbm_matrices(uncensored(train_rows), label="ordinal")
                if len(set(ytr)) < 2:
                    report["ltr"] = {"status": "no_positive_labels"}
                else:
                    model = train_lambdamart(Xtr, ytr, gtr, _FEATURE_NAMES)
                    test_ordered = sorted(
                        test_rows,
                        key=lambda r: model.predict([r.features])[0],
                        reverse=True,
                    )
                    test_rel = [float(r.ordinal_relevance) for r in test_ordered]
                    preds = [float(model.predict([r.features])[0]) for r in test_ordered]
                    report["ltr"] = evaluate_ranking(test_rel, preds, args.k)
                    report["ltr"]["label_positive"] = sum(1 for r in test_rel if r > 0)
                    # Business funnel on the test rows as labeled.
                    report["ltr"]["applied"] = sum(1 for r in test_rows if r.applied)
                    report["ltr"]["screening"] = sum(1 for r in test_rows if r.screening)
                    report["ltr"]["interview"] = sum(1 for r in test_rows if r.interview)
                    report["ltr"]["offer"] = sum(1 for r in test_rows if r.offer)
                    # Promotion decision vs the current baseline.
                    report["promotion"] = promotion_decision(report["current"], report["ltr"])
        except Exception as e:
            report["ltr"] = {"status": f"error: {e}"}
    else:
        report["note"] = "no test impressions (need more data)"

    print(json.dumps(report, indent=2, default=str))
    await store.close()


def _funnel_from_rows(rows: list[Any]) -> dict[str, float]:
    total = max(len(rows), 1)
    applied = sum(1 for r in rows if r.applied)
    screening = sum(1 for r in rows if r.screening)
    interview = sum(1 for r in rows if r.interview)
    offer = sum(1 for r in rows if r.offer)
    denom = max(applied, 1)
    return {
        "application_rate": round(applied / total, 4),
        "screening_rate": round(screening / denom, 4),
        "interview_rate": round(interview / denom, 4),
        "offer_rate": round(offer / denom, 4),
        "interviews_per_100": round(interview / denom * 100, 2),
        "offers_per_100": round(offer / denom * 100, 2),
    }


_FEATURE_NAMES = [
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


if __name__ == "__main__":
    asyncio.run(main())
