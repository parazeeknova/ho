"""Learning-to-rank: separate LambdaMART ranker + binary classifiers.

LambdaMART (lambdarank) learns ORDERING. Binary classifiers learn
P(screen), P(interview), P(offer) — each on its own funnel stage.
They are calibrated separately (isotonic/Platt). Never one model pretending
to be both.
"""

from __future__ import annotations

from typing import Any


def groups_for_lambdamart(impression_ids: list[str]) -> list[int]:
    """Group sizes for LightGBM ranker — one group per impression/batch."""
    from collections import Counter

    c = Counter(impression_ids)
    # Preserve insertion order of first occurrence
    seen: list[str] = []
    for iid in impression_ids:
        if iid not in seen:
            seen.append(iid)
    return [c[iid] for iid in seen]


async def train_ranker(
    store: Any,
    feature_version: str,
    version: str = "v1",
) -> dict[str, Any]:
    """Train LambdaMART ranker on pairwise preference + outcome data."""
    try:
        import lightgbm as lgb
    except ImportError:
        return {"error": "lightgbm not installed"}

    # Load training data: impressions with ranked jobs + pairwise labels
    # For now, stub — real training needs accumulated decision_events.
    # This is the training entry point; Phase 1 collects data before it fires.
    return {"model_type": "lgb_ranker", "version": version, "status": "stub"}


async def train_classifiers(
    store: Any,
    feature_version: str,
    version: str = "v1",
) -> dict[str, Any]:
    """Train binary classifiers P(screen), P(interview), P(offer)."""
    try:
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return {"error": "lightgbm/sklearn not installed"}
    return {"model_type": "clf", "version": version, "status": "stub"}
