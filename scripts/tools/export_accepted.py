"""Export accepted candidates from radar_candidates to CSV dumps.

Usage:
    uv run python scripts/export_accepted.py                   # jobs -> intel/accepted_jobs.csv
    uv run python scripts/export_accepted.py --mode outreach  # founders/funding/socials csv
    uv run python scripts/export_accepted.py --mode all       # everything
    uv run python scripts/export_accepted.py --eligibility near_miss
    uv run python scripts/export_accepted.py --out /tmp/x.csv # custom path
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]

FIELDNAMES_JOBS = [
    "company",
    "role",
    "location",
    "match_percent",
    "shortlist_probability",
    "verdict",
    "eligibility",
    "apply_url",
    "source",
    "role_family",
    "salary_amount",
    "salary_currency",
    "salary_period",
    "salary_raw",
    "posted_date",
    "first_seen_utc",
    "last_seen_utc",
    "freshness_lane",
    "is_remote",
    "matching_skills",
    "missing_skills",
]

FIELDNAMES_OUTREACH = [
    "company",
    "role",
    "match_percent",
    "shortlist_probability",
    "verdict",
    "eligibility",
    "apply_url",
    "posted_date",
    "funding_stage",
    "funding_info",
    "founders",
    "founder_socials",
    "company_news",
    "osint_signals",
    "jd_summary",
    "role_summary",
]

FIELDNAMES = [
    *FIELDNAMES_JOBS,
    "funding_stage",
    "funding_info",
    "founders",
    "founder_socials",
    "company_news",
    "osint_signals",
    "jd_summary",
    "role_summary",
]


def _skills(val: Any) -> str:
    if isinstance(val, list):
        return "|".join(str(x) for x in val)
    return ""


def _ts(epoch: Any) -> str:
    if not epoch:
        return ""
    import datetime as dt

    try:
        return dt.datetime.fromtimestamp(float(epoch), tz=dt.UTC).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(epoch)


def _row(r: Any) -> dict[str, Any]:
    return {
        "company": r["normalized_company"],
        "role": r["normalized_role"],
        "location": r["normalized_location"],
        "match_percent": r["match_percent"],
        "shortlist_probability": r["shortlist_probability"],
        "verdict": r["verdict"],
        "eligibility": r["eligibility"],
        "apply_url": r["direct_apply_url"],
        "source": r["source"],
        "role_family": r["role_family"],
        "salary_amount": r["salary_amount"],
        "salary_currency": r["salary_currency"],
        "salary_period": r["salary_period"],
        "salary_raw": r["salary_raw"],
        "posted_date": r["posted_date"],
        "first_seen_utc": _ts(r["first_seen"]),
        "last_seen_utc": _ts(r["last_seen"]),
        "freshness_lane": r["freshness_lane"],
        "is_remote": r["is_remote"],
        "matching_skills": _skills(r["matching_skills"]),
        "missing_skills": _skills(r["missing_skills"]),
        "funding_stage": r["funding_stage"],
        "funding_info": json.dumps(r["funding_info"]) if r["funding_info"] else "",
        "founders": _skills(r["founders"]),
        "founder_socials": _skills(r["founder_socials"]),
        "company_news": r["company_news"],
        "osint_signals": _skills(r["osint_signals"]),
        "jd_summary": r["jd_summary"],
        "role_summary": r["role_summary"],
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="Dump accepted radar candidates to CSV")
    ap.add_argument(
        "--eligibility",
        default="accepted",
        help="eligibility to export (accepted|near_miss|rejected|all), default accepted",
    )
    ap.add_argument(
        "--mode",
        default="jobs",
        choices=("jobs", "outreach", "all"),
        help="jobs: apply columns; outreach: founders/funding/socials/news; all: everything",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output CSV path (default intel/accepted_<mode>.csv)",
    )
    args = ap.parse_args()

    from src.memory.pgvector_store import MemoryStore

    if args.mode == "jobs":
        fieldnames = FIELDNAMES_JOBS
    elif args.mode == "outreach":
        fieldnames = FIELDNAMES_OUTREACH
    else:
        fieldnames = FIELDNAMES
    out = Path(args.out) if args.out else PROJECT / "intel" / f"accepted_{args.mode}.csv"

    store = await MemoryStore.create()
    try:
        if args.eligibility == "all":
            where, params = "", []
        else:
            where, params = "WHERE eligibility = $1", [args.eligibility]

        async with store._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT canonical_id, source, direct_apply_url, normalized_company,
                       normalized_role, normalized_location, freshness_lane,
                       eligibility, role_family, salary_amount, salary_currency,
                       salary_period, salary_raw, posted_date, first_seen, last_seen,
                       matching_skills, missing_skills, match_percent,
                       shortlist_probability, verdict, jd_summary, company_news,
                       role_summary, is_remote, founders, funding_stage,
                       founder_socials, osint_signals, funding_info
                FROM radar_candidates
                {where}
                ORDER BY match_percent DESC, first_seen DESC
                """,
                *params,
            )
    finally:
        await store.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            full = _row(r)
            w.writerow({k: full[k] for k in fieldnames})

    print(f"Exported {len(rows)} {args.eligibility} candidates ({args.mode}) -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
