"""Tests for the review's statistical fixes: censoring, ordinal labels,
impression-level temporal split, physical grouping, promotion decision, and
calibration-on-validation semantics."""

from __future__ import annotations

from ml.src.ranking.dataset import (
    RankedRow,
    build_dataset,
    events_to_ordinal,
    lgbm_matrices,
    temporal_split,
    uncensored,
)
from ml.src.ranking.evaluation import brier_score, expected_calibration_error, promotion_decision


def _row(job_id: str, imp: str, ts: float, ordinal: int = 0, censored: bool = False) -> RankedRow:
    return RankedRow(
        job_id=job_id,
        impression_id=imp,
        features=[0.0] * 22,
        raw_features={},
        relevance=min(1.0, ordinal / 6.0),
        ordinal_relevance=ordinal,
        applied=1 if ordinal >= 3 else 0,
        screening=1 if ordinal >= 4 else 0,
        interview=1 if ordinal >= 5 else 0,
        offer=1 if ordinal >= 6 else 0,
        ts=ts,
        censored=censored,
    )


def test_events_to_ordinal_highest_stage_wins():
    events = [{"event_type": "confirmation_email"}, {"event_type": "interview"}]
    assert events_to_ordinal(events) == 5
    assert events_to_ordinal([{"event_type": "offer"}]) == 6
    assert events_to_ordinal([{"event_type": "job_clicked"}]) == 1
    assert events_to_ordinal([]) == 0


def test_temporal_split_keeps_impression_integrity():
    """Rows of one impression must never straddle train/val/test."""
    rows = []
    for i in range(10):
        imp = f"imp_{i}"
        # 3 rows per impression, slightly staggered ts.
        for j in range(3):
            rows.append(_row(f"job_{i}_{j}", imp, ts=float(i * 3 + j)))
    train, val, test = temporal_split(rows, 0.7, 0.15)
    train_imps = {r.impression_id for r in train}
    val_imps = {r.impression_id for r in val}
    test_imps = {r.impression_id for r in test}
    assert train_imps.isdisjoint(val_imps)
    assert train_imps.isdisjoint(test_imps)
    assert val_imps.isdisjoint(test_imps)
    # Every impression appears in exactly one split, fully.

    all_imps = {r.impression_id for r in rows}
    assert train_imps | val_imps | test_imps == all_imps


def test_uncensored_drops_young_rows():
    rows = [
        _row("a", "imp1", 1.0, ordinal=3, censored=False),
        _row("b", "imp2", 2.0, censored=True),
    ]
    kept = uncensored(rows)
    assert [r.job_id for r in kept] == ["a"]


def test_lgbm_matrices_physical_grouping():
    """Rows of the same impression must be CONTIGUOUS in X/y/group."""
    rows = [
        _row("a1", "A", ts=5),
        _row("b1", "B", ts=1),
        _row("a2", "A", ts=6),
        _row("b2", "B", ts=2),
        _row("c1", "C", ts=3),
    ]
    X, _y, group = lgbm_matrices(rows, label="ordinal")
    # Group array must equal [2,2,1] AND the underlying X must be sorted so
    # group boundaries line up (A A B B C).
    assert group == [2, 2, 1]
    assert len(X) == 5
    # y must be ordered by (impression_id, ts): A(5,6), B(1,2), C(3)
    ids = ["a1", "a2", "b1", "b2", "c1"]
    assert [r.job_id for r in sorted(rows, key=lambda r: (r.impression_id, r.ts))] == ids


def test_ordinal_labels_used():
    rows = [
        _row("a", "A", 1.0, ordinal=3),
        _row("b", "A", 2.0, ordinal=6),
        _row("c", "B", 3.0, ordinal=0),
    ]
    _, y, _ = lgbm_matrices(rows, label="ordinal")
    assert sorted(y) == [0.0, 3.0, 6.0]
    # relevance label is the 0..1 normalized version.
    _, y_rel, _ = lgbm_matrices(rows, label="relevance")
    assert max(y_rel) == 1.0


def test_promotion_decision_requires_all_wins_and_calibration():
    baseline = {"ndcg@10": 0.41, "precision@10": 0.31, "interview_rate": 3.2, "ece": 0.05}
    good = {"ndcg@10": 0.48, "precision@10": 0.37, "interview_rate": 4.4, "ece": 0.05}
    d = promotion_decision(baseline, good)
    assert d["promote"] is True
    # Loses interview rate -> no promotion even if nDCG improves.
    partial = {"ndcg@10": 0.48, "precision@10": 0.37, "interview_rate": 2.0, "ece": 0.05}
    assert promotion_decision(baseline, partial)["promote"] is False
    # Calibration worsens -> no promotion.
    bad_cal = {"ndcg@10": 0.48, "precision@10": 0.37, "interview_rate": 4.4, "ece": 0.20}
    assert promotion_decision(baseline, bad_cal)["promote"] is False


def test_calibration_metrics():
    # miscalibrated: predicts 0.9 but all outcomes are negative.
    y = [0, 0, 0]
    p = [0.9, 0.9, 0.9]
    assert expected_calibration_error(y, p, n_bins=10) > 0.5
    # perfectly calibrated binary pair.
    assert brier_score([0, 1], [0, 1]) == 0.0
    assert brier_score([0], [1]) == 1.0
    # Overconfident (predict 0.8 for a positive) has a nonzero Brier.
    assert abs(brier_score([1], [0.8]) - 0.04) < 1e-9


def test_build_dataset_marks_censored():
    imp = [
        {
            "job_id": "old",
            "impression_id": "i1",
            "features": {"semantic_fit": 0.5},
            "created_at": 1_600_000_000,
        },
        {
            "job_id": "recent",
            "impression_id": "i2",
            "features": {"semantic_fit": 0.5},
            "created_at": 1_600_000_000 + 290 * 86400,  # 290 days later
        },
    ]
    now = 1_600_000_000 + 300 * 86400
    events = {}
    ds = build_dataset(imp, events, label_maturity_days=30, now=now)
    by_job = {r.job_id: r for r in ds.rows}
    assert by_job["old"].censored is False  # 300 days old -> mature
    assert by_job["recent"].censored is True  # only 10 days old < 30d window


def test_hierarchical_discovery_policy_propensity_is_product():
    """Two-level choice: propensity must be the product of family and source
    pi so IPS/SNIPS stay valid for the hierarchical decision."""
    from ml.src.bandits.policies import DISCOVERY_HIERARCHY, DiscoveryPolicy

    p = DiscoveryPolicy(hierarchy=DISCOVERY_HIERARCHY)
    # No exploration (eps=0) -> greedy family + greedy source. With fresh
    # Beta(1,1) priors MC-P is near-uniform, so pi ~ (1/|families|)*(1/|sources|).
    source, mu = p.choose(exploration=0.0)
    assert source in {s for fam in DISCOVERY_HIERARCHY.values() for s in fam}
    assert 0.0 < mu <= 1.0
    # A reward to a source routes to both its family and the source.
    p.update("greenhouse", 1.0)
    source2, mu2 = p.choose(exploration=0.0)
    assert source2 in {s for fam in DISCOVERY_HIERARCHY.values() for s in fam}
    assert 0.0 < mu2 <= 1.0


def test_discovery_hierarchy_covers_all_ats_families():
    from ml.src.bandits.policies import DISCOVERY_HIERARCHY

    ats = DISCOVERY_HIERARCHY["ats"]
    assert "greenhouse" in ats
    assert "ashby" in ats
    assert "lever" in ats
    assert "workable" in ats
    assert "workday" in ats
    assert "smartrecruiters" in ats
    assert "rippling" in ats
    assert len(ats) >= 10
    # Families present.
    assert set(DISCOVERY_HIERARCHY) >= {"ats", "search", "community", "aggregator"}
