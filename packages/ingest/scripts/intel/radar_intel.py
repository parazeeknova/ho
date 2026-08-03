#!/usr/bin/env python3
"""Radar Intelligence Engine - learning, fuzzy matching, recommendations.

The mission is heavy application volume. This engine:

1. LEARNS a skill graph from every gated candidate: the skills that
   actually match (your proven stack), the skills you miss (common
   gaps -> LARP candidates), and how skills co-occur across jobs.
   The more jobs it sees, the better it knows you and the fuzzier it
   can recommend (never hard-strict on learnable gaps).

2. FUZZY-CLASSIFIES every gated job into a tier:
     HARD_MATCH  >=80%  - go apply now
     FUZZY_MATCH 50-79% - missing 1-3 learnable skills, apply + LARP
     LARP_MATCH  <50%   - core overlap + LARP-able gaps (2-3 max)
     HARD_REJECT        - genuine no-gos (phd, clearance, 10yr+)

3. LARP-AWARE: distinguishes learnable gaps (otel, odin, drizzle,
   temporal, langchain...) from hard requirements (phd, clearance,
   citizenship, 10+ years). Missing "hard" skills cap the tier.

4. RECOMMENDS + exports: ranked list with LARP tags, plus CSV/JSON
   feeds for an auto-apply pipeline (your friend's project).

Usage:
    uv run python scripts/intel/radar_intel.py            # analyze + print + export
    uv run python scripts/intel/radar_intel.py --top 20   # top N recommendations
    uv run python scripts/intel/radar_intel.py --telegram # push digest to Telegram
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Hard requirements that can NEVER be LARPed - universal signals, not a
# skill list (skills number in the hundreds and we don't hardcode them).
_HARD_MISS_PATTERNS = [
    re.compile(r"\bph\.?d\b|\bdoctorate\b|\bpostdoc\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\+?\s*(?:years?|yrs?)\s+", re.IGNORECASE),
    re.compile(r"\bclearance\b|\bsecret\s+clearance\b", re.IGNORECASE),
    re.compile(r"\bcitizen(?:ship)?\b|\bcitizens?\s+only\b", re.IGNORECASE),
    re.compile(r"\bactive\s+top\s+secret\b", re.IGNORECASE),
]

# Degree/credential/experience words make a miss a HARD requirement.
# Universal grammar, not a per-skill guess.
_HARD_WORDS = re.compile(
    r"\b(?:ph\.?d|doctorate|masters|bachelor|clearance|certification|"
    r"citizenship|sponsorship|yrs?|years?|degree|license)\b",
    re.IGNORECASE,
)

# Titles that are hard-senior even if missing skills look learnable.
_HARD_TITLE = re.compile(
    r"\b(?:principal|staff|architect|director|head of|vp|senior staff)\b",
    re.IGNORECASE,
)


def _skill_list(val: Any) -> list[str]:
    """Normalize skills from list[str], str, or jsonb-typed values."""
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [val]
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


class SkillGraph:
    """Learned model of the candidate's skill landscape."""

    def __init__(self) -> None:
        self.matched: Counter[str] = Counter()
        self.missed: Counter[str] = Counter()
        self.cooccur: defaultdict[str, Counter] = defaultdict(Counter)
        self.larp_success: Counter[str] = Counter()
        self.total_jobs = 0

    def learn(self, matching, missing, accepted: bool) -> None:
        self.total_jobs += 1
        for s in _skill_list(matching):
            s = _norm_skill(s)
            if s and len(s) > 1:
                self.matched[s] += 1
        for s in _skill_list(missing):
            s = _norm_skill(s)
            if s and len(s) > 1:
                self.missed[s] += 1
                if accepted:
                    self.larp_success[s] += 1
        # co-occurrence of your matched skills
        ms = [_norm_skill(s) for s in _skill_list(matching)]
        for a in ms:
            for b in ms:
                if a and b and a != b:
                    self.cooccur[a][b] += 1

    def is_larp_friendly(self, skill: str) -> bool:
        """Data-driven LARP check - no hardcoded skill list.

        A missing skill is LARP-able unless it is a universal hard signal
        (phd, years, clearance, citizenship) OR a long descriptive phrase
        about experience/credentials. Everything else - any tech/tool name,
        known or unknown - is treated as learnable. The system learns over
        time: a skill missed on jobs that were accepted becomes confidently
        LARP-able (larp_success), and a skill missed across the market is
        common enough to be learnable.
        """
        s = skill.strip().lower()
        if not s or len(s) < 2:
            return False
        if any(p.search(s) for p in _HARD_MISS_PATTERNS):
            return False
        if _HARD_WORDS.search(s):
            return False
        # Learned: you already LARPed this on an accepted job.
        if self.larp_success[s] >= 2:
            return True
        # Composition heuristic: real tech/tool names are short, dense,
        # alphanumeric strings ("otel", "drizzle", "temporal", "aws").
        # Long descriptive phrases about experience are NOT LARP-able.
        words = s.split()
        if len(words) <= 3 and re.fullmatch(r"[\w./+#-]+", s.replace(" ", "")):
            return True
        # Learned: missed very often across the market -> common expectation.
        return self.missed[s] >= 5

    def larp_effort(self, skill: str) -> str:
        """Heuristic effort estimate based on the skill's word count/size."""
        s = skill.lower()
        words = len(s.split())
        if len(s) <= 4 and words == 1:
            return "2-4d"
        if len(s) <= 12 or words <= 2:
            return "1w"
        return "1-2w"


def _norm_skill(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip()


def classify(
    match_percent: int,
    missing: list,
    graph: SkillGraph,
) -> tuple[str, list[str], list[str]]:
    """Return (tier, larp_skills, hard_blocks)."""
    larp: list[str] = []
    hard: list[str] = []
    for s in _skill_list(missing):
        s = _norm_skill(s)
        if not s or len(s) < 2:
            continue
        if graph.is_larp_friendly(s):
            larp.append(s)
        else:
            hard.append(s)

    # Hard blocks cap the tier.
    if len(hard) >= 1 and match_percent < 60:
        return "HARD_REJECT", larp, hard
    if len(hard) >= 2:
        return "HARD_REJECT", larp, hard

    if match_percent >= 80 and len(hard) == 0:
        return "HARD_MATCH", larp, hard
    if match_percent >= 60 or len(larp) <= 3:
        return "FUZZY_MATCH", larp, hard
    return "LARP_MATCH", larp, hard


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--export", default=str(PROJECT / "intel"))
    args = ap.parse_args()

    from src.http_cache import set_http_cache_store
    from src.memory.pgvector_store import MemoryStore

    store = await MemoryStore.create()
    set_http_cache_store(store)
    graph = SkillGraph()

    async with store._pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT normalized_company, normalized_role, match_percent,
                      matching_skills, missing_skills, eligibility, verdict,
                      direct_apply_url, funding_stage
               FROM radar_candidates
               WHERE eligibility IN ('accepted','near_miss','rejected')
            """
        )
        jobs: list[dict[str, Any]] = []
        for r in rows:
            accepted = r["eligibility"] == "accepted"
            graph.learn(r["matching_skills"], r["missing_skills"], accepted)
            tier, larp, hard = classify(r["match_percent"], r["missing_skills"], graph)
            jobs.append(
                {
                    "company": r["normalized_company"],
                    "role": r["normalized_role"],
                    "match": r["match_percent"],
                    "verdict": r["verdict"],
                    "eligibility": r["eligibility"],
                    "apply_url": r["direct_apply_url"],
                    "funding_stage": r["funding_stage"],
                    "matching_skills": r["matching_skills"],
                    "missing_skills": r["missing_skills"],
                    "larp_skills": larp,
                    "hard_blocks": hard,
                    "tier": tier,
                }
            )

    print(
        f"Analyzed {len(jobs)} gated jobs | skill graph: "
        f"{len(graph.matched)} matched, {len(graph.missed)} missed"
    )

    # Recommendations
    ranked = sorted(
        jobs,
        key=lambda j: (
            0
            if j["tier"] == "HARD_MATCH"
            else 1
            if j["tier"] == "FUZZY_MATCH"
            else 2
            if j["tier"] == "LARP_MATCH"
            else 3,
            -j["match"],
        ),
    )
    show = [j for j in ranked if j["tier"] != "HARD_REJECT"][: args.top]

    print("\n=== TOP RECOMMENDATIONS ===")
    for i, j in enumerate(show, 1):
        larp_str = ", ".join(j["larp_skills"]) or "-"
        hard_str = ", ".join(j["hard_blocks"]) or "-"
        print(f"{i:2d}. [{j['tier']:<11}] {j['match']:3d}% {j['company']} | {j['role'][:40]}")
        print(f"     apply: {j['apply_url'][:70]}")
        print(f"     LARP:  {larp_str}")
        if j["hard_blocks"]:
            print(f"     hard:  {hard_str}")

    # Skill gaps you could close for more matches
    print("\n=== SKILL GAP INTELLIGENCE (top misses worth learning) ===")
    top_misses = graph.missed.most_common(12)
    for skill, count in top_misses:
        larp = graph.is_larp_friendly(skill)
        eff = graph.larp_effort(skill) if larp else "hard"
        print(f"  {skill:<28} missed in {count:4d} jobs | {eff}")

    # Export for the auto-applier
    outdir = Path(args.export)
    outdir.mkdir(exist_ok=True)
    csv_path = outdir / "recommendations.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tier",
                "match_percent",
                "company",
                "role",
                "apply_url",
                "larp_skills",
                "hard_blocks",
                "matching_skills",
                "missing_skills",
            ],
        )
        w.writeheader()
        for j in jobs:
            w.writerow(
                {
                    "tier": j["tier"],
                    "match_percent": j["match"],
                    "company": j["company"],
                    "role": j["role"],
                    "apply_url": j["apply_url"],
                    "larp_skills": "|".join(j["larp_skills"]),
                    "hard_blocks": "|".join(j["hard_blocks"]),
                    "matching_skills": "|".join(j["matching_skills"] or []),
                    "missing_skills": "|".join(j["missing_skills"] or []),
                }
            )
    json_path = outdir / "recommendations.json"
    json_path.write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).isoformat(),
                "skill_graph": {
                    "matched": dict(graph.matched.most_common(50)),
                    "missed": dict(graph.missed.most_common(50)),
                },
                "recommendations": show,
            },
            indent=2,
        )
    )
    print(f"\nExported {len(jobs)} jobs -> {csv_path} + {json_path}")

    # Telegram digest
    if args.telegram:
        try:
            from src.agent.telegram_agent import TelegramAgent

            ta = TelegramAgent()
            if ta.is_configured:
                lines = ["<b>🧠 Radar Intelligence Digest</b>", ""]
                lines.append(f"Analyzed <b>{len(jobs)}</b> gated jobs.")
                lines.append(
                    f"Graph: {len(graph.matched)} known skills, {len(graph.missed)} gaps learned."
                )
                lines.append("")
                lines.append("<b>Top LARP plays:</b>")
                larp_jobs = [
                    j
                    for j in ranked
                    if j["tier"] in ("FUZZY_MATCH", "LARP_MATCH") and j["larp_skills"]
                ][:8]
                for j in larp_jobs:
                    larp_str = ", ".join(j["larp_skills"][:3])
                    lines.append(f"▪ <b>{j['company']}</b> — {j['role'][:40]} ({j['match']}%)")
                    lines.append(f"   LARP: <i>{larp_str}</i>")
                    if j["apply_url"]:
                        lines.append(f'   <a href="{j["apply_url"]}">Apply →</a>')
                await ta._send_raw("\n".join(lines))
                print("Telegram digest sent")
        except Exception as exc:
            print(f"telegram digest failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
