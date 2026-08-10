"""AnalyticsAgent: Generates market intelligence from PostgreSQL and Neo4j."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.logging import get_logger

if TYPE_CHECKING:
    from src.graph.graph_store import GraphStore
    from src.llm.context import ContextManager
    from src.memory.pgvector_store import MemoryStore

logger = get_logger("analytics_agent")


class AnalyticsAgent:
    """Generates market reports and company hiring profiles."""

    def __init__(
        self,
        store: MemoryStore,
        graph: GraphStore,
        ctx: ContextManager,
    ) -> None:
        self.store = store
        self.graph = graph
        self.ctx = ctx
        self._interactive = False

    async def generate_resilient_report(self, interactive: bool = False) -> list[str]:
        self._interactive = interactive
        sections: list[list[str]] = []
        section_funcs = [
            ("Pipeline Health", self._section_health),
            ("Acceptance Overview", self._section_radar_stats),
            ("Top Companies To Chase", self._section_top_companies),
            ("Sector Signal (vector)", self._section_sector_signal),
            ("Most In-Demand Skills", self._section_radar_skills),
            ("Near-Miss Skill Gaps", self._section_radar_arbitrage),
            ("Rejection Breakdown", self._section_radar_rejections),
            ("Salary Statistics", self._section_radar_salaries),
            ("Freshness Lanes", self._section_radar_freshness),
            ("Funding Hiring Signal", self._section_funding_hiring),
            ("Repost Signal", self._section_reposts),
        ]

        for name, func in section_funcs:
            try:
                result = await func()
                if result:
                    sections.append(result)
            except Exception as e:
                logger.warning(
                    "Analytics section failed",
                    section=name,
                    exception=str(e),
                )

        return ["\n".join(s) for s in sections]

    # smarter sections

    async def _section_health(self) -> list[str]:
        """Live pipeline velocity: how fast are matches landing right now."""
        lines = ["**Pipeline Velocity**"]
        try:
            rows = await self.store.get_recent_accepts(hours=24)
            total24 = len(rows)
            h1 = [r for r in rows if (time.time() - (r.get("ts") or 0)) < 3600]
            n_h1 = len(h1)
            lines.append(f"  Accepted last 24h: **{total24}**")
            lines.append(f"  Accepted last 1h:  **{n_h1}**")
            if rows:
                lines.append(f"  Rate: ~{total24 / 24:.1f}/hr ({total24 / 24 / 60:.2f}/min)")
            near = await self.store.get_near_miss_count()
            lines.append(f"  Near-miss (LARP-able): {near}")
        except Exception:
            lines.append("  *Velocity data unavailable.*")
        lines.append("")
        return lines

    async def _section_top_companies(self) -> list[str]:
        """Accepted companies ranked — where your applications actually land."""
        lines = ["**Top Companies To Chase**"]
        try:
            top = await self.store.get_top_companies(limit=8)
            if top:
                for idx, c in enumerate(top, 1):
                    stage = c.get("funding_stage") or "seed"
                    lines.append(
                        f"  {idx}. {c['company']} — "
                        f"{c['accepted']} accepted, match {c.get('avg_match', 0)}% "
                        f"[{stage}]"
                    )
            else:
                lines.append("  *No accepted companies yet.*")
        except Exception:
            lines.append("  *Company data unavailable.*")
        lines.append("")
        return lines

    async def _section_sector_signal(self) -> list[str]:
        """Vector-discovered sector clusters from accepted vs market (if embedded)."""
        lines = ["**Sector Signal**"]
        try:
            signal = await self.store.get_sector_signal(limit=6)
            if signal:
                for idx, s in enumerate(signal, 1):
                    lines.append(
                        f"  {idx}. {s['label']} — {s['count']} accepted, {s['pct']}% of accepted"
                    )
            else:
                lines.append("  *Embed a few sweeps first; sector signal needs vectors.*")
        except Exception:
            lines.append("  *Sector data unavailable.*")
        lines.append("")
        return lines

    async def _section_radar_stats(self) -> list[str]:
        lines = ["**Radar Gate Stats**"]
        try:
            stats = await self.store.get_radar_gate_stats()
            lines.append(f"  Total candidates: {stats.get('total', 0)}")
            lines.append(f"  Accepted: {stats.get('accepted', 0)}")
            lines.append(f"  Near-miss: {stats.get('near_miss', 0)}")
            lines.append(f"  Rejected: {stats.get('rejected', 0)}")
            lines.append(f"  Urgent lane: {stats.get('urgent', 0)}")
            lines.append(f"  Review lane: {stats.get('review', 0)}")
        except Exception as e:
            logger.warning("Radar stats failed", exception=str(e))
            lines.append("  *Radar data not yet available.*")
        lines.append("")
        return lines

    async def _section_radar_skills(self) -> list[str]:
        lines = ["**Most In-Demand Skills**"]
        try:
            top = await self.store.get_radar_top_skills(limit=12)
            if top:
                for idx, item in enumerate(top, 1):
                    lines.append(
                        f"  {idx}. {item['skill']} ({item['count']} matches)",
                    )
            else:
                lines.append("  *No skills data from radar yet.*")
        except Exception:
            lines.append("  *Radar skills unavailable.*")
        lines.append("")
        return lines

    async def _section_radar_arbitrage(self) -> list[str]:
        lines = [
            "**Near-Miss Skill Gaps**",
            "  *(Skills that appear most in near-miss roles)*",
        ]
        try:
            arbitrage = await self.store.get_radar_skill_arbitrage()
            if arbitrage:
                for idx, item in enumerate(arbitrage[:8], 1):
                    lines.append(
                        f"  {idx}. {item['skill']} (blocked {item['miss_count']} roles)",
                    )
            else:
                lines.append("  *No near-miss data yet.*")
        except Exception:
            lines.append("  *Near-miss data unavailable.*")
        lines.append("")
        return lines

    async def _section_radar_rejections(self) -> list[str]:
        lines = ["**Rejection Breakdown**"]
        try:
            stats = await self.store.get_radar_gate_stats()
            top = stats.get("top_rejection_reasons", [])
            if top:
                for rr in top[:6]:
                    reason = rr.get("reason", "?").replace("_", " ")
                    lines.append(f"  • {reason}: {rr.get('count', 0)}")
            else:
                lines.append("  *No rejections recorded yet.*")
        except Exception:
            lines.append("  *Rejection data unavailable.*")
        lines.append("")
        return lines

    async def _section_radar_salaries(self) -> list[str]:
        lines = ["**Salary Statistics**"]
        try:
            s = await self.store.get_salary_stats()
            if s.get("count", 0) > 0:
                lines.append(f"  Median: {s.get('median', 0):,}")
                lines.append(f"  Average: {s.get('avg', 0):,}")
                lines.append(f"  Roles with salary: {s.get('count', 0)}")
            else:
                lines.append("  *Not enough salary data yet.*")
        except Exception:
            lines.append("  *Salary data unavailable.*")
        lines.append("")
        return lines

    async def _section_radar_freshness(self) -> list[str]:
        lines = ["**Posting Freshness Lanes**"]
        try:
            stats = await self.store.get_radar_gate_stats()
            total = stats.get("total", 0)
            if total > 0:
                pct_urgent = stats.get("urgent", 0) / total * 100
                pct_review = stats.get("review", 0) / total * 100
                lines.append(f"  Urgent: {stats.get('urgent', 0)} ({pct_urgent:.0f}%)")
                lines.append(f"  Review: {stats.get('review', 0)} ({pct_review:.0f}%)")
            else:
                lines.append("  *No freshness data yet.*")
        except Exception:
            lines.append("  *Freshness data unavailable.*")
        lines.append("")
        return lines

    def _load_smart_intel(self) -> dict:
        """Load intel/smart_intel.json (written by scripts/intel/smart_intel.py)."""
        import json
        from pathlib import Path

        p = Path(__file__).resolve().parent.parent.parent / "intel" / "smart_intel.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return {}

    async def _section_funding_hiring(self) -> list[str]:
        """Companies that just raised AND are hiring now (highest-ROI tier)."""
        lines = ["**Funding + Hiring Signal**"]
        data = self._load_smart_intel()
        fh = data.get("funding_hiring") or []
        if fh:
            for idx, c in enumerate(fh[:8], 1):
                fund = (c.get("funding") or [{}])[0]
                amount = fund.get("amount_usd")
                amt = f"${amount / 1e6:.1f}M" if amount else ""
                stage = fund.get("stage", "")
                roles = ", ".join(r["role"][:24] for r in c.get("hiring_roles", [])[:2])
                lines.append(f"  {idx}. {c['company']} — raised {amt} {stage} → hiring: {roles}")
        else:
            lines.append("  *No funding-hiring signals yet (Azure funding tracker feeding).*")
        lines.append("")
        return lines

    async def _section_reposts(self) -> list[str]:
        """Same role reposted repeatedly = active/hungry hiring, apply again."""
        lines = ["**Repost Signal — Re-Hiring Now**"]
        data = self._load_smart_intel()
        reps = data.get("reposts") or []
        if reps:
            for idx, r in enumerate(reps[:8], 1):
                lines.append(f"  {idx}. {r['company']} — {r['role'][:30]} seen {r['seen_times']}x")
        else:
            lines.append("  *No repost signal yet.*")
        lines.append("")
        return lines
