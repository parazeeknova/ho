"""Real training-dataset builder: decision_events → impressions → labels → Parquet.

Pipeline (per the review's "training semantics" priority):

    decision_events
          │
          ▼
    dataset builder
          │
          ├── temporal cutoff (only events before now)
          ├── impression grouping (job_ranked rows per impression_id)
          ├── outcome attribution (latest reward event per job_id)
          ├── label construction (ranked relevance + funnel-stage positives)
          ├── feature extraction (numeric vector, feature_version-filtered)
          ├── leakage guard (feature.observed_at <= impression.created_at)
          └── train/val/test temporal split
          │
          ▼
        Parquet
          │
    ┌─────┴──────┐
    ▼           ▼
 LambdaRank   classifiers
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .features import NUMERIC_FEATURES, features_to_vector

# Rewards (from reward.py) mapped to a 0..1 relevance for LambdaMART ranking.
# Ranking relevance is the max positive-outcome reward observed for the job,
# normalized. Negative outcomes floor at 0. This is the LABEL, separate from
# the raw reward (policy uses raw reward; ranking uses relevance).
_POSITIVE_REWARD_STAGES = {
    "screening": 5.0,
    "screening_email": 5.0,
    "interview": 15.0,
    "offer": 100.0,
    "application_confirmed": 0.75,
    "confirmation_email": 0.75,
    "application_submitted": 0.5,
    "job_applied": 0.5,
    "job_saved": 0.25,
    "job_clicked": 0.10,
}

# Ordinal relevance grades (the reviewer's P2 ask): LambdaMART learns ordering,
# so give it an ordinal target rather than pretending the reward scale itself
# is meaningful. 0 = no outcome, 1 = clicked, 2 = saved, 3 = applied,
# 4 = screened, 5 = interviewed, 6 = offer. The classifier/EV layer owns the
# actual probabilities and utilities.
_ORDINAL_STAGE = {
    "job_clicked": 1,
    "job_saved": 2,
    "job_applied": 3,
    "application_submitted": 3,
    "application_confirmed": 3,
    "confirmation_email": 3,
    "screening": 4,
    "screening_email": 4,
    "interview": 5,
    "offer": 6,
}


def reward_to_relevance(reward: float | None) -> float:
    """Map a raw reward to a 0..1 LambdaMART relevance grade."""
    if reward is None or reward <= 0:
        return 0.0
    # Rejection is a negative outcome but still informative: it means the job
    # was pursued and failed, which is worse than never applied.
    # We keep negative as 0 for ranking (don't rank rejected higher), but the
    # classifier stages use the raw events separately.
    return min(1.0, reward / 100.0)


def events_to_ordinal(events: list[dict[str, Any]]) -> int:
    """Highest ordinal stage reached across a job's outcome events (0..6)."""
    best = 0
    for ev in events:
        stage = _ORDINAL_STAGE.get(ev.get("event_type", ""))
        if stage is not None and stage > best:
            best = stage
    return best


@dataclass
class RankedRow:
    """One job within an impression, ready for LambdaMART or classifier."""

    job_id: str
    impression_id: str
    features: list[float]
    raw_features: dict[str, Any]
    # Ranking label (0..1 relevance)
    relevance: float
    # Ordinal ranking label (0..6): 0 none, 1 clicked, 2 saved, 3 applied,
    # 4 screened, 5 interviewed, 6 offer. LambdaMART learns ordering on this.
    ordinal_relevance: int
    # Funnel-stage labels (1 if the outcome was at/above that stage)
    applied: int
    screening: int
    interview: int
    offer: int
    reward: float | None = None
    ts: float = 0.0
    # Outcome censoring: True when the impression is younger than
    # label_maturity_days, so no outcome has had time to mature. Censored rows
    # must be EXCLUDED from supervised label training — labeling a 2-day-old
    # application as a negative because no interview has appeared is bias.
    censored: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "impression_id": self.impression_id,
            "features": self.features,
            "relevance": self.relevance,
            "ordinal_relevance": self.ordinal_relevance,
            "applied": self.applied,
            "screening": self.screening,
            "interview": self.interview,
            "offer": self.offer,
            "reward": self.reward,
            "ts": self.ts,
            "censored": self.censored,
        }


@dataclass
class Dataset:
    rows: list[RankedRow] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.rows

    def impression_ids(self) -> list[str]:
        return [r.impression_id for r in self.rows]

    def to_parquet(self, path: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.is_empty():
            return
        # Columnar: expand features into named columns + labels.
        data: dict[str, Any] = {f: [] for f in NUMERIC_FEATURES}
        data["job_id"] = []
        data["impression_id"] = []
        data["relevance"] = []
        data["ordinal_relevance"] = []
        data["applied"] = []
        data["screening"] = []
        data["interview"] = []
        data["offer"] = []
        data["ts"] = []
        data["censored"] = []
        for r in self.rows:
            for i, fname in enumerate(NUMERIC_FEATURES):
                data[fname].append(r.features[i] if i < len(r.features) else 0.0)
            data["job_id"].append(r.job_id)
            data["impression_id"].append(r.impression_id)
            data["relevance"].append(r.relevance)
            data["ordinal_relevance"].append(r.ordinal_relevance)
            data["applied"].append(r.applied)
            data["screening"].append(r.screening)
            data["interview"].append(r.interview)
            data["offer"].append(r.offer)
            data["ts"].append(r.ts)
            data["censored"].append(1 if r.censored else 0)
        table = pa.table(data)
        pq.write_table(table, path)


async def fetch_impressions(store: Any, feature_version: str) -> list[dict[str, Any]]:
    """Fetch job_ranked rows (features + impression + rank) with feature_version
    filter, oldest first. This is the ranking event surface."""
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT event_id, job_id, impression_id, candidate_id,
                   candidate_snapshot_id, job_snapshot_id,
                   features, rank, policy, model_version, feature_version,
                   exploration, propensity, source, created_at
            FROM decision_events
            WHERE event_type = 'job_ranked'
              AND impression_id IS NOT NULL
              AND (feature_version = $1 OR $1 = '')
            ORDER BY created_at ASC
            """,
            feature_version,
        )
        return [dict(r) for r in rows]


async def fetch_outcomes(store: Any) -> dict[str, dict[str, Any]]:
    """Latest reward/outcome event per job_id, for label attribution."""
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (job_id) job_id, event_type, reward, created_at
            FROM decision_events
            WHERE reward IS NOT NULL
            ORDER BY job_id, created_at DESC
            """
        )
        return {r["job_id"]: dict(r) for r in rows}


async def fetch_all_job_events(store: Any) -> dict[str, list[dict[str, Any]]]:
    """Every reward-bearing event per JOB KEY (for funnel-stage labels).

    Rewards carry the autofill job_id (job-xxxx) while ranked rows carry the
    radar canonical_id (url hash). To attach outcomes to ranked rows, key the
    map by BOTH the reward's job_id AND the canonical_id of the radar_candidate
    with the same direct_apply_url as the reward's autofill job."""
    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT job_id, event_type, reward, created_at
            FROM decision_events
            WHERE reward IS NOT NULL
            ORDER BY created_at ASC
            """
        )
        by_job: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_job.setdefault(r["job_id"], []).append(dict(r))
        # Bridge autofill job_id -> radar canonical_id via apply_url. The radar
        # re-hashes canonical_id each sweep, so also match ranked events whose
        # meta->>'url' equals the apply_link (the URL is now stored in meta).
        try:
            bridges = await conn.fetch(
                """
                SELECT a.job_id AS auto_id, c.canonical_id
                FROM autofill_queue a
                JOIN radar_candidates c ON c.direct_apply_url = a.apply_link
                WHERE a.job_id IN (SELECT DISTINCT job_id FROM decision_events WHERE reward IS NOT NULL)
                """
            )
            for b in bridges:
                auto_id = b["auto_id"]
                canon = b["canonical_id"]
                if auto_id in by_job:
                    by_job.setdefault(canon, []).extend(by_job[auto_id])
            # Meta-url bridge: reward's autofill job -> ranked events whose
            # meta->>'url' matches the apply_link.
            meta_bridges = await conn.fetch(
                """
                SELECT DISTINCT a.job_id AS auto_id, de.job_id AS ranked_job
                FROM autofill_queue a
                JOIN decision_events de
                  ON de.event_type = 'job_ranked' AND de.meta->>'url' = a.apply_link
                WHERE a.job_id IN (SELECT DISTINCT job_id FROM decision_events WHERE reward IS NOT NULL)
                """
            )
            for b in meta_bridges:
                auto_id = b["auto_id"]
                ranked_job = b["ranked_job"]
                if auto_id in by_job and ranked_job not in by_job:
                    by_job.setdefault(ranked_job, []).extend(by_job[auto_id])
        except Exception:
            pass
        return by_job


def _stage_flags(events: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    """Compute applied/screening/interview/offer flags from a job's reward events."""
    applied = screening = interview = offer = 0
    for ev in events:
        t = ev.get("event_type", "")
        if t in (
            "job_applied",
            "application_submitted",
            "application_confirmed",
            "confirmation_email",
        ):
            applied = 1
        elif t in ("screening", "screening_email"):
            screening = 1
            applied = 1
        elif t == "interview":
            interview = 1
            screening = 1
            applied = 1
        elif t == "offer":
            offer = 1
            interview = 1
            screening = 1
            applied = 1
    return applied, screening, interview, offer


def _max_relevance(events: list[dict[str, Any]]) -> float:
    """Highest stage reward reached → relevance grade (0..1)."""
    best = 0.0
    for ev in events:
        r = ev.get("reward")
        if r is not None and r > best:
            best = r
    return reward_to_relevance(best)


def _leakage_ok(features: dict[str, Any], decision_ts: float) -> bool:
    """Feature observed_at must be <= the decision timestamp (leakage guard)."""
    observed = features.get("observed_at")
    return not (isinstance(observed, (int, float)) and observed > decision_ts + 1.0)


def build_dataset(
    impressions: list[dict[str, Any]],
    job_events: dict[str, list[dict[str, Any]]],
    label_maturity_days: float = 0.0,
    now: float | None = None,
) -> Dataset:
    """Assemble RankedRows from impression rows + per-job outcome events.

    - Groups are impressions (LambdaMART ranking unit).
    - relevance = max positive reward stage reached.
    - ordinal_relevance = highest ordinal stage reached (0..6) — the LambdaMART
      label, since LambdaMART learns ordering.
    - Funnel flags applied/screening/interview/offer for classifiers.
    - Feature vector via features_to_vector (stable NUMERIC_FEATURES order).
    - Censoring: when label_maturity_days > 0, rows younger than that window
      are marked censored (no outcome has had time to mature). Censored rows
      must be excluded from supervised label training.
    """
    now_ts = now if now is not None else time.time()
    ds = Dataset()
    for imp in impressions:
        job_id = imp.get("job_id") or ""
        impression_id = imp.get("impression_id") or ""
        raw_feats = imp.get("features") or {}
        # Older events (pre-jsonb-fix) stored features as a JSON *string*.
        if isinstance(raw_feats, str):
            try:
                raw_feats = json.loads(raw_feats)
            except Exception:
                raw_feats = {}
        if not isinstance(raw_feats, dict):
            raw_feats = {}
        # Leakage guard: reject rows whose features were observed after decision.
        decision_ts = _ts_of(imp.get("created_at"))
        if not _leakage_ok(raw_feats, decision_ts):
            continue
        events = job_events.get(job_id, [])
        applied, screening, interview, offer = _stage_flags(events)
        relevance = _max_relevance(events)
        ordinal = events_to_ordinal(events)
        censored = label_maturity_days > 0 and (now_ts - decision_ts) < label_maturity_days * 86400
        # Reward for the row = the outcome reward (for policy record), or the
        # original ranked row's reward if it had one (it won't).
        row = RankedRow(
            job_id=job_id,
            impression_id=impression_id,
            features=features_to_vector(raw_feats),
            raw_features=raw_feats,
            relevance=relevance,
            ordinal_relevance=ordinal,
            applied=applied,
            screening=screening,
            interview=interview,
            offer=offer,
            reward=events[-1].get("reward") if events else None,
            ts=decision_ts,
            censored=censored,
        )
        ds.rows.append(row)
    return ds


def _ts_of(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "timestamp"):
        with contextlib.suppress(Exception):
            return v.timestamp()
    return time.time()


def temporal_split(
    rows: list[RankedRow],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[list[RankedRow], list[RankedRow], list[RankedRow]]:
    """Temporal (NOT random) train/val/test split, at the IMPRESSION level.

    The split unit is the unique impression_id — an impression's entire group
    of rows goes into exactly one split. Splitting individual rows would let
    one impression leak across train and validation (rows of the same ranking
    are not independent). Impressions are ordered by their earliest timestamp
    and sliced into train/val/test, then expanded back into rows.
    """
    if not rows:
        return [], [], []
    # Group rows by impression_id, ordered by each impression's oldest ts.
    by_imp: dict[str, list[RankedRow]] = {}
    for r in rows:
        by_imp.setdefault(r.impression_id, []).append(r)
    ordered_imps = sorted(
        by_imp.values(),
        key=lambda grp: min(r.ts for r in grp),
    )
    n_imps = len(ordered_imps)
    if n_imps < 3:
        return list(rows), [], []
    t1 = int(n_imps * train_frac)
    t2 = int(n_imps * (train_frac + val_frac))
    train_imps = ordered_imps[:t1]
    val_imps = ordered_imps[t1:t2]
    test_imps = ordered_imps[t2:]
    out_train = [r for g in train_imps for r in g]
    out_val = [r for g in val_imps for r in g]
    out_test = [r for g in test_imps for r in g]
    # Preserve each split's internal ordering by ts (stable, deterministic).
    out_train.sort(key=lambda r: r.ts)
    out_val.sort(key=lambda r: r.ts)
    out_test.sort(key=lambda r: r.ts)
    return out_train, out_val, out_test


def lgbm_matrices(
    rows: list[RankedRow],
    label: str = "ordinal",
) -> tuple[list[list[float]], list[float], list[int]]:
    """X, y, group for LightGBM lambdarank from ranked rows.

    Rows are FIRST physically sorted by (impression_id, created_at) so that
    rows of the same impression are CONTIGUOUS — the group array LightGBM
    receives must line up with the actual row order. Without this invariant a
    future query change can silently interleave impressions (A1 B1 A2 B2) with
    group=[2,2] and produce plausible-but-garbage metrics.

    ``label`` selects the target: ``"ordinal"`` (0..6 grade, the reviewer's P2
    recommendation — LambdaMART learns ordering, not a fake reward scale) or
    ``"relevance"`` (0..1, backward compatible).
    """
    if not rows:
        return [], [], []
    ordered = sorted(rows, key=lambda r: (r.impression_id, r.ts))
    X = [r.features for r in ordered]
    if label == "relevance":
        y = [r.relevance for r in ordered]
    else:
        y = [float(r.ordinal_relevance) for r in ordered]
    group = groups_for_lambdamart([r.impression_id for r in ordered])
    return X, y, group


def uncensored(rows: list[RankedRow]) -> list[RankedRow]:
    """Drop rows whose outcome label is still censored (too young to mature).

    Training labels for censored impressions are biased: a 2-day-old job with
    only an application would be labeled "no interview" even though the
    interview simply hasn't had time to arrive. Exclude them from supervised
    label training; keep them only for impression-group bookkeeping.
    """
    return [r for r in rows if not r.censored]


def classifier_matrices(rows: list[RankedRow], stage: str) -> tuple[list[list[float]], list[int]]:
    """X, y for a funnel-stage binary classifier (applied/screening/interview/offer)."""
    X = [r.features for r in rows]
    key = {
        "applied": "applied",
        "screening": "screening",
        "interview": "interview",
        "offer": "offer",
    }.get(stage)
    if key is None:
        raise ValueError(f"unknown stage {stage}")
    y = [1 if getattr(r, key) else 0 for r in rows]
    return X, y


def to_np(X, y, group=None):
    """Convert dataset matrices to numpy arrays for LightGBM."""
    import numpy as np

    Xn = np.array(X, dtype=float)
    yn = np.array(y, dtype=float)
    if group is not None:
        return Xn, yn, [int(g) for g in group]
    return Xn, yn


def groups_for_lambdamart(impression_ids: list[str]) -> list[int]:
    from collections import Counter

    c = Counter(impression_ids)
    seen: list[str] = []
    for iid in impression_ids:
        if iid not in seen:
            seen.append(iid)
    return [c[iid] for iid in seen]
