"""LLM planning pass: proposes new dork queries + source patterns from the
recent accepted-candidate record.

Reads the last N accepted candidates (company, role, source, skills,
funding) plus the current dork query list, asks the LLM for:

    { "queries": [5 new SearXNG dork queries],
      "sources": [3 new source patterns/URLs to watch],
      "rationale": "one short sentence per suggestion" }

Output is written to intel/planning_suggestions.json and NEVER applied
automatically: review it, then wire approved queries into your dork
config or board registry. The pass runs through the shared governor, so
it can never blow the LLM rate budget.

Run:  make intel MODE=planning          (or uv run python scripts/intel/planning_pass.py)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from src.http_cache import set_http_cache_store
from src.llm.context import ContextManager
from src.logging import get_logger
from src.memory.pgvector_store import MemoryStore
from src.radar.sources.dorking import _DORK_QUERIES

logger = get_logger("planning_pass")

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "intel"
OUT_FILE = OUT_DIR / "planning_suggestions.json"
TOP_N = 30

PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "5 new SearXNG dork queries to discover roles like the accepted ones",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3 new source patterns (ATS slugs, boards, GitHub indexes, feed URLs)",
        },
        "rationale": {
            "type": "string",
            "description": "Short rationale for each suggestion",
        },
    },
    "required": ["queries", "sources", "rationale"],
}


async def _recent_accepted(store: MemoryStore, top: int = TOP_N) -> list[dict[str, Any]]:
    async with store._pool.acquire() as c:
        rows = await c.fetch(
            """
            SELECT normalized_company, normalized_role, source, match_percent,
                   funding_stage, jd_summary, matching_skills
            FROM radar_candidates
            WHERE eligibility = 'accepted'
              AND normalized_company != ''
            ORDER BY created_at DESC
            LIMIT $1
            """,
            top,
        )
    out: list[dict[str, Any]] = []
    for r in rows:
        skills = r["matching_skills"]
        if isinstance(skills, str):
            try:
                skills = json.loads(skills)
            except Exception:
                skills = []
        out.append(
            {
                "company": r["normalized_company"],
                "role": r["normalized_role"],
                "source": r["source"],
                "match": r["match_percent"],
                "funding": r["funding_stage"] or "",
                "jd_summary": (r["jd_summary"] or "")[:200],
                "skills": skills if isinstance(skills, list) else [],
            }
        )
    return out


def _build_prompt(accepted: list[dict[str, Any]], current_queries: list[str]) -> str:
    lines = [
        "You are the radar planner for a job-intelligence pipeline hunting global",
        "high-pay underdog roles for a specific candidate (junior/mid software",
        "roles, remote-friendly, visa-sponsoring, early-stage).",
        "",
        f"Current SearXNG dork queries ({len(current_queries)}):",
    ]
    lines += [f"  - {q}" for q in current_queries[:25]]
    lines.append("")
    lines.append(f"Recent {len(accepted)} accepted roles (evidence of what works):")
    for j in accepted:
        lines.append(
            f"  - {j['company']} | {j['role']} | {j['source']} | match {j['match']}% "
            f"| funding: {j['funding'] or 'n/a'}"
        )
        if j["skills"]:
            lines.append(f"      skills: {', '.join(map(str, j['skills'][:8]))}")
        if j["jd_summary"]:
            lines.append(f"      jd: {j['jd_summary'][:160]}")
    lines.append("")
    lines.append(
        "Propose 5 NEW search queries and 3 NEW source patterns (ATS board "
        "slugs, GitHub index repos, company career URLs, feed patterns) that "
        "would surface MORE roles like these. Queries must be SearXNG-style "
        "dork syntax. Avoid repeating the current queries verbatim."
    )
    return "\n".join(lines)


def _write(blob: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(blob, indent=2, default=str))
    logger.info(f"wrote {OUT_FILE}")


async def run(write: bool) -> dict[str, Any]:
    store = await MemoryStore.create()
    set_http_cache_store(store)
    ctx = ContextManager()
    try:
        accepted = await _recent_accepted(store)
        logger.info(f"planning_pass: {len(accepted)} recent accepted roles")
        prompt = _build_prompt(accepted, list(_DORK_QUERIES))

        extracted = await ctx.json_chat(prompt, schema=PLANNING_SCHEMA)
        if not isinstance(extracted, dict):
            raise RuntimeError("planning LLM returned non-dict")

        queries = [str(q) for q in extracted.get("queries", [])[:5] if q]
        sources = [str(s) for s in extracted.get("sources", [])[:3] if s]
        blob = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "based_on_roles": len(accepted),
            "suggested_queries": queries,
            "suggested_sources": sources,
            "rationale": str(extracted.get("rationale", "")),
            "applied": False,  # never auto-applied; review manually
        }
        logger.info(
            f"planning_pass: {len(queries)} queries, {len(sources)} sources suggested (NOT applied)"
        )
        if write:
            _write(blob)
        return blob
    finally:
        await store.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM planning pass for radar sources")
    ap.add_argument("--write", action="store_true", help="write planning_suggestions.json")
    args = ap.parse_args()
    asyncio.run(run(args.write))


if __name__ == "__main__":
    main()
