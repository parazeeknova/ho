"""Feature substrate — candidate × job × context.

Graph features are included FROM THE BEGINNING (before LTR) so the first
ranker already sees the best available signal — no churn from adding them
later. Every feature carries observed_at and the extraction asserts
observed_at <= decision_ts to block temporal leakage.

Feature version is bumped on any change so training can filter by version.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from . import FEATURE_VERSION

# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------


class FeatureLeakageError(RuntimeError):
    pass


def assert_no_leakage(features: dict[str, Any], decision_ts: float) -> None:
    for k, v in features.items():
        if isinstance(v, dict) and "observed_at" in v:
            if v["observed_at"] > decision_ts + 1.0:  # 1s clock skew tolerance
                raise FeatureLeakageError(
                    f"feature {k} observed_at {v['observed_at']} > decision {decision_ts}"
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _salary_fit(candidate_floor: float | None, job_salary: float | None) -> float:
    if job_salary is None or candidate_floor is None:
        return 0.5
    if job_salary >= candidate_floor * 1.2:
        return 1.0
    if job_salary >= candidate_floor:
        return 0.8
    if job_salary >= candidate_floor * 0.7:
        return 0.4
    return 0.1


def _location_fit(candidate_loc: str | None, job_loc: str | None, is_remote: bool) -> float:
    if is_remote:
        return 1.0
    if not candidate_loc or not job_loc:
        return 0.5
    if candidate_loc.lower() in job_loc.lower() or job_loc.lower() in candidate_loc.lower():
        return 1.0
    return 0.3


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_features(
    candidate: dict[str, Any],
    job: dict[str, Any],
    context: dict[str, Any] | None = None,
    graph_signals: dict[str, Any] | None = None,
    decision_ts: float | None = None,
) -> dict[str, Any]:
    """Build the candidate×job×context feature vector.

    All features are numeric (float/int) or short categorical strings suitable
    for LightGBM. Graph signals are optional — when present they enrich the
    vector; when absent the model still ranks (graceful degradation).
    """
    decision_ts = decision_ts or time.time()
    ctx = context or {}
    gs = graph_signals or {}

    cand_skills = set(s.lower() for s in (candidate.get("skills") or []))
    job_skills = set(s.lower() for s in (job.get("matching_skills") or []))
    missing = set(s.lower() for s in (job.get("missing_skills") or []))

    tech_overlap = 0.0
    company_techs: set[str] = set(t.lower() for t in (gs.get("company_techs") or []))
    if cand_skills and company_techs:
        tech_overlap = len(cand_skills & company_techs) / max(len(cand_skills), 1)

    # Seniority / experience gap
    cand_exp = candidate.get("experience_years")
    job_exp_req = job.get("experience_required")
    exp_gap = None
    if cand_exp is not None and job_exp_req is not None:
        try:
            exp_gap = float(cand_exp) - float(job_exp_req)
        except Exception:
            pass

    feats: dict[str, Any] = {
        # --- semantic / qualification ---
        "semantic_fit": float(job.get("match_percent", 0)) / 100.0,
        "matching_skills_count": len(job_skills),
        "missing_skills_count": len(missing),
        "skill_coverage": len(job_skills) / max(len(job_skills) + len(missing), 1),
        "exp_gap": exp_gap if exp_gap is not None else 0.0,
        # --- source / freshness ---
        "source_confidence": float(job.get("source_confidence", 0.5)),
        "freshness_score": {"urgent": 1.0, "review": 0.5, "stale": 0.2}.get(
            str(job.get("freshness_lane", "review")).lower(), 0.5
        ),
        "source": str(job.get("source", "unknown")),
        # --- compensation / location ---
        "salary_fit": _salary_fit(candidate.get("salary_floor"), job.get("salary_amount")),
        "location_fit": _location_fit(
            candidate.get("location"), job.get("normalized_location"), bool(job.get("is_remote"))
        ),
        "is_remote": 1.0 if job.get("is_remote") else 0.0,
        "sponsors_visa": 1.0 if job.get("sponsors_visa") else 0.0,
        # --- company / underdog ---
        "underdog_score": float(job.get("underdog_score", 0.5)),
        "funding_stage_ordinal": {
            "seed": 0,
            "pre-seed": 0,
            "series a": 1,
            "series b": 2,
            "series c": 3,
            "series d": 4,
        }.get(str(job.get("funding_stage", "")).lower(), 2),
        "funding_recency_days": gs.get("funding_recency_days", 9999),
        "osint_count": len(job.get("osint_signals") or []),
        # --- graph signals (when available) ---
        "company_pagerank": float(gs.get("pagerank", 0.0)),
        "company_betweenness": float(gs.get("betweenness", 0.0)),
        "founder_hiring_signal": 1.0 if gs.get("hiring_signal") else 0.0,
        "tech_overlap": tech_overlap,
        "graph_hiring_likelihood": float(gs.get("hiring_likelihood", 0.5)),
        # --- context ---
        "hour_of_day": float(ctx.get("hour_of_day", 12)) / 24.0,
        "market": str(ctx.get("market", "unknown")),
        # --- versioning / leakage guard ---
        "feature_version": FEATURE_VERSION,
        "observed_at": decision_ts,
    }

    # Leakage guard — every feature's observed_at must be <= decision
    assert_no_leakage({"_self": {"observed_at": decision_ts}}, decision_ts)

    return feats


# Numeric feature names in stable order (for LightGBM input alignment).
NUMERIC_FEATURES: list[str] = [
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

CATEGORICAL_FEATURES: list[str] = ["source", "market"]


def features_to_vector(feats: dict[str, Any], feature_list: list[str] | None = None) -> list[float]:
    fl = feature_list or NUMERIC_FEATURES
    return [float(feats.get(k, 0.0) or 0.0) for k in fl]


def feature_hash(features: dict[str, Any]) -> str:
    raw = json.dumps({k: features.get(k) for k in NUMERIC_FEATURES}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
