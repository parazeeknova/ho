"""Offline evaluation: current ranker vs LTR shadow on historical impressions.

Metrics (business ground truth, not just nDCG):
  nDCG@10, precision@10, MRR
  application rate, screening rate, interview rate, offer rate
  interviews/100 applications, offers/100 applications

Temporal holdout: the most recent test-window impressions are scored by both
the current (hand-weighted) ranker and the LTR model; the comparison decides
whether LTR earns the right to influence production.

Usage:
  uv run python packages/ml/scripts/eval.py --k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from ml import FEATURE_VERSION
from ml.dataset import build_dataset, fetch_all_job_events, fetch_impressions, temporal_split
from ml.evaluation import evaluate_ranking, funnel_metrics


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

    impressions = await fetch_impressions(store, FEATURE_VERSION)
    job_events = await fetch_all_job_events(store)
    ds = build_dataset(impressions, job_events)
    if ds.is_empty():
        print("No impressions yet — run sweeps first.")
        await store.close()
        return

    train_rows, val_rows, test_rows = temporal_split(ds.rows, 0.7, 0.15)
    print(f"rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    # Current hand-weighted ranker proxy: rank by the stored `rank` (production
    # order) — that IS the current policy's output on the test impressions.
    # LTR: if a trained model exists, predict on test features; else report
    # that shadow model is unavailable.
    from ml.ltr import lgbm_matrices, train_lambdamart

    report: dict[str, Any] = {
        "n": len(test_rows),
        "k": args.k,
        "current": {},
        "ltr": {},
        "funnel": funnel_metrics([r.as_dict() for r in test_rows]),
    }

    if test_rows:
        # Current policy ordering (by stored rank asc) per impression.
        current_order = sorted(
            test_rows,
            key=lambda r: (r.impression_id, r.raw_features.get("rank", 99) or 99),
        )
        current_rel = [r.relevance for r in current_order]
        report["current"] = evaluate_ranking(current_rel, current_rel, args.k)
        report["current"]["label_positive"] = sum(1 for r in current_rel if r > 0)

        # LTR shadow: train on train rows, score test rows.
        try:
            from ml.ltr import _check_lgb

            if _check_lgb():
                Xtr, ytr, gtr = lgbm_matrices(train_rows)
                if len(set(ytr)) >= 2:
                    model = train_lambdamart(Xtr, ytr, gtr, _FEATURE_NAMES)
                    test_ordered = sorted(
                        test_rows,
                        key=lambda r: model.predict([r.features])[0],
                        reverse=True,
                    )
                    test_rel = [r.relevance for r in test_ordered]
                    report["ltr"] = evaluate_ranking(
                        test_rel,
                        [model.predict([r.features])[0] for r in test_ordered],
                        args.k,
                    )
                    report["ltr"]["label_positive"] = sum(1 for r in test_rel if r > 0)
                else:
                    report["ltr"] = {"status": "no_positive_labels"}
            else:
                report["ltr"] = {"status": "lightgbm_missing"}
        except Exception as e:
            report["ltr"] = {"status": f"error: {e}"}
    else:
        report["note"] = "no test impressions (need more data)"

    print(json.dumps(report, indent=2, default=str))
    await store.close()


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
