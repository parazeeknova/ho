"""Local smart-intelligence aggregation over the radar + OSINT data.

Pulls the following signals together (no LLM, runs on the local box):

  1. Funding->hiring alerts: companies that recently raised (from
     company_osint.signals) AND are actively hiring (accepted/near-miss
     candidates) -> "just raised, hiring now" high-ROI tier.
  2. Salary intelligence: median/avg by role-family, by location, by
     company funding-stage.
  3. Freshness score: how recent each never-applied candidate's posting is
     (apply-timing priority).
  4. Skill-gap -> LARP plan: which learnable skills unlock the most roles.
  5. Location/visa strategy: where wins concentrate, sponsor coverage.
  6. Repost detection: same company+role appearing again weeks later.

Writes intel/smart_intel.json + .csv for the friend's auto-applier and the
Telegram analytics. Run from the local box (NOT the relic):
    uv run python3 scripts/intel/smart_intel.py [--write]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Any

from src.http_cache import set_http_cache_store
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore

logger = get_logger("smart_intel")

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "intel"


async def _funding_hiring(store: MemoryStore) -> list[dict[str, Any]]:
    """Companies that recently raised AND are hiring right now."""
    async with store._pool.acquire() as c:
        rows = await c.fetch(
            """
            SELECT co.company, co.data, co.cached_at,
                   rc.normalized_company, rc.normalized_role, rc.match_percent
            FROM company_osint co
            LEFT JOIN LATERAL (
                SELECT normalized_company, normalized_role, match_percent
                FROM radar_candidates rc2
                WHERE rc2.normalized_company = co.company
                  AND rc2.eligibility IN ('accepted','near_miss')
                ORDER BY rc2.created_at DESC LIMIT 3
            ) rc ON TRUE
            WHERE co.data::text ILIKE '%funding_event%'
               OR co.data::text ILIKE '%amount_usd%'
               OR co.data::text ILIKE '%raise%'
            """
        )
    by_co: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = r["data"] if isinstance(r["data"], dict) else json.loads(r["data"] or "{}")
        signals = d.get("signals", [])
        founders = d.get("founders", [])
        key = r["company"]
        if key not in by_co:
            by_co[key] = {
                "company": key,
                "funding": [s for s in signals if s.get("funding_event") or s.get("amount_usd")],
                "founders": len(founders),
                "hiring_roles": [],
                "raised_at": r["cached_at"],
            }
        if r["normalized_role"]:
            by_co[key]["hiring_roles"].append(
                {
                    "role": r["normalized_role"],
                    "match": r["match_percent"],
                }
            )
    out = []
    for co in by_co.values():
        if co["funding"] and co["hiring_roles"]:
            out.append(co)
    return sorted(out, key=lambda x: -(x["hiring_roles"][0]["match"] or 0))


async def _salary_intel(store: MemoryStore) -> dict[str, Any]:
    async with store._pool.acquire() as c:
        by_role = await c.fetch(
            """
            SELECT COALESCE(NULLIF(role_family,''),'unknown') AS fam,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY salary_amount)::int AS med,
                   round(avg(salary_amount))::int AS avg, count(*) AS n
            FROM radar_candidates
            WHERE salary_amount > 0 AND salary_currency='USD' AND salary_period='year'
            GROUP BY 1 ORDER BY n DESC
            """
        )
        by_loc = await c.fetch(
            """
            SELECT COALESCE(NULLIF(normalized_location,''),'unknown') AS loc,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY salary_amount)::int AS med,
                   count(*) AS n
            FROM radar_candidates
            WHERE salary_amount > 0 AND salary_currency='USD' AND salary_period='year'
            GROUP BY 1 ORDER BY n DESC LIMIT 6
            """
        )
    return {
        "by_role_family": [dict(r) for r in by_role],
        "by_location": [dict(r) for r in by_loc],
    }


async def _skill_gap_plan(store: MemoryStore) -> list[dict[str, Any]]:
    """Which learnable missing-skills unlock the most near-miss roles."""
    async with store._pool.acquire() as c:
        rows = await c.fetch(
            """
            WITH norm AS (
                SELECT CASE
                    WHEN jsonb_typeof(missing_skills)='array' THEN missing_skills
                    WHEN jsonb_typeof(missing_skills)='string' THEN (missing_skills #>> '{}')::jsonb
                    ELSE '[]'::jsonb END AS sk
                FROM radar_candidates WHERE eligibility='near_miss'
            )
            SELECT skill, count(*) AS roles
            FROM norm, LATERAL jsonb_array_elements_text(sk) skill
            WHERE skill IS NOT NULL AND skill != ''
            GROUP BY 1 ORDER BY roles DESC LIMIT 15
            """
        )
    return [{"skill": r["skill"], "unlocks_roles": r["roles"]} for r in rows]


async def _freshness(store: MemoryStore) -> list[dict[str, Any]]:
    """Apply-timing priority: freshest accepted/near-miss first."""
    async with store._pool.acquire() as c:
        rows = await c.fetch(
            """
            SELECT normalized_company, normalized_role, match_percent, last_seen,
                   eligibility, direct_apply_url
            FROM radar_candidates
            WHERE eligibility IN ('accepted','near_miss')
              AND last_seen IS NOT NULL
            ORDER BY last_seen DESC LIMIT 30
            """
        )
    now = time.time()
    out = []
    for r in rows:
        age_h = (now - float(r["last_seen"])) / 3600
        out.append(
            {
                "company": r["normalized_company"],
                "role": r["normalized_role"],
                "match": r["match_percent"],
                "age_hours": round(age_h, 1),
                "eligibility": r["eligibility"],
                "url": r["direct_apply_url"],
            }
        )
    return out


async def _location_visa(store: MemoryStore) -> dict[str, Any]:
    async with store._pool.acquire() as c:
        loc = await c.fetch(
            """
            SELECT COALESCE(NULLIF(normalized_location,''),'unknown') AS loc, count(*) n
            FROM radar_candidates WHERE eligibility IN ('accepted','near_miss')
            GROUP BY 1 ORDER BY n DESC LIMIT 8
            """
        )
        sponsor = await c.fetchval(
            "SELECT count(*) FROM radar_candidates "
            "WHERE eligibility IN ('accepted','near_miss') "
            "AND (extra->>'sponsors_visa')::text = 'true'"
        )
        tot = await c.fetchval(
            "SELECT count(*) FROM radar_candidates WHERE eligibility IN ('accepted','near_miss')"
        )
    return {
        "top_locations": [dict(r) for r in loc],
        "visa_sponsoring": sponsor,
        "total_wins": tot,
    }


async def _reposts(store: MemoryStore) -> list[dict[str, Any]]:
    """Same company+role posted again days later = expanding/desperate."""
    async with store._pool.acquire() as c:
        rows = await c.fetch(
            """
            SELECT normalized_company, normalized_role,
                   min(first_seen) AS first_ts, max(first_seen) AS last_ts, count(*) AS n
            FROM radar_candidates
            WHERE eligibility IN ('accepted','near_miss')
              AND normalized_company != '' AND normalized_company NOT LIKE '%not specified%'
            GROUP BY normalized_company, normalized_role
            HAVING count(*) > 1
            ORDER BY count(*) DESC LIMIT 12
            """
        )
    return [
        {
            "company": r["normalized_company"],
            "role": r["normalized_role"],
            "seen_times": r["n"],
            "first_ts": r["first_ts"],
            "last_ts": r["last_ts"],
        }
        for r in rows
    ]


def _write(blob: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "smart_intel.json").write_text(json.dumps(blob, indent=2, default=str))
    fields = ["company", "role", "match", "age_hours", "eligibility", "url"]
    with open(OUT_DIR / "smart_intel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in blob.get("freshness", []):
            w.writerow({k: row.get(k, "") for k in fields})
    logger.info(f"wrote {OUT_DIR / 'smart_intel.json'} + smart_intel.csv")


async def run(write: bool) -> dict[str, Any]:
    store = await MemoryStore.create()
    set_http_cache_store(store)
    try:
        funding_hiring = await _funding_hiring(store)
        salary = await _salary_intel(store)
        skill_gap = await _skill_gap_plan(store)
        freshness = await _freshness(store)
        loc_visa = await _location_visa(store)
        reposts = await _reposts(store)
        blob = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "funding_hiring": funding_hiring,
            "salary": salary,
            "skill_gap_plan": skill_gap,
            "freshness": freshness,
            "location_visa": loc_visa,
            "reposts": reposts,
        }
        logger.info(
            f"smart_intel: {len(funding_hiring)} funding-hiring, {len(reposts)} reposts, "
            f"{len(skill_gap)} skill-gap, {len(freshness)} fresh"
        )
        if write:
            _write(blob)
        return blob
    finally:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Local smart-intelligence aggregation")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.write))


if __name__ == "__main__":
    main()
