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

    # ── smarter sections ───────────────────────────────────────────────

    async def _section_health(self) -> list[str]:
        """Live pipeline velocity: how fast are matches landing right now."""
        lines = ["<b>Pipeline Velocity</b>"]
        try:
            rows = await self.store.get_recent_accepts(hours=24)
            total24 = len(rows)
            h1 = [r for r in rows if (time.time() - (r.get("ts") or 0)) < 3600]
            n_h1 = len(h1)
            lines.append(f"  Accepted last 24h: <b>{total24}</b>")
            lines.append(f"  Accepted last 1h:  <b>{n_h1}</b>")
            if rows:
                lines.append(
                    f"  Rate: ~{total24 / 24:.1f}/hr ({total24 / 24 / 60:.2f}/min)"
                )
            near = await self.store.get_near_miss_count()
            lines.append(f"  Near-miss (LARP-able): {near}")
        except Exception:
            lines.append("  <i>Velocity data unavailable.</i>")
        lines.append("")
        return lines

    async def _section_top_companies(self) -> list[str]:
        """Accepted companies ranked — where your applications actually land."""
        lines = ["<b>Top Companies To Chase</b>"]
        try:
            top = await self.store.get_top_companies(limit=8)
            if top:
                for idx, c in enumerate(top, 1):
                    stage = c.get("funding_stage") or "seed"
                    lines.append(
                        f"  {idx}. {_esc(c['company'])} — "
                        f"{c['accepted']} accepted, match {c.get('avg_match', 0)}% "
                        f"[{_esc(stage)}]"
                    )
            else:
                lines.append("  <i>No accepted companies yet.</i>")
        except Exception:
            lines.append("  <i>Company data unavailable.</i>")
        lines.append("")
        return lines

    async def _section_sector_signal(self) -> list[str]:
        """Vector-discovered sector clusters from accepted vs market (if embedded)."""
        lines = ["<b>Sector Signal</b>"]
        try:
            signal = await self.store.get_sector_signal(limit=6)
            if signal:
                for idx, s in enumerate(signal, 1):
                    lines.append(
                        f"  {idx}. {_esc(s['label'])} — "
                        f"{s['count']} accepted, {s['pct']}% of accepted"
                    )
            else:
                lines.append("  <i>Embed a few sweeps first; sector signal needs vectors.</i>")
        except Exception:
            lines.append("  <i>Sector data unavailable.</i>")
        lines.append("")
        return lines

    async def _section_radar_stats(self) -> list[str]:
        lines = ["<b>Radar Gate Stats</b>"]
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
            lines.append("  <i>Radar data not yet available.</i>")
        lines.append("")
        return lines

    async def _section_radar_skills(self) -> list[str]:
        lines = ["<b>Most In-Demand Skills</b>"]
        try:
            top = await self.store.get_radar_top_skills(limit=12)
            if top:
                for idx, item in enumerate(top, 1):
                    lines.append(
                        f"  {idx}. {_esc(item['skill'])} ({item['count']} matches)",
                    )
            else:
                lines.append("  <i>No skills data from radar yet.</i>")
        except Exception:
            lines.append("  <i>Radar skills unavailable.</i>")
        lines.append("")
        return lines

    async def _section_radar_arbitrage(self) -> list[str]:
        lines = [
            "<b>Near-Miss Skill Gaps</b>",
            "  <i>(Skills that appear most in near-miss roles)</i>",
        ]
        try:
            arbitrage = await self.store.get_radar_skill_arbitrage()
            if arbitrage:
                for idx, item in enumerate(arbitrage[:8], 1):
                    lines.append(
                        f"  {idx}. {_esc(item['skill'])} (blocked {item['miss_count']} roles)",
                    )
            else:
                lines.append("  <i>No near-miss data yet.</i>")
        except Exception:
            lines.append("  <i>Near-miss data unavailable.</i>")
        lines.append("")
        return lines

    async def _section_radar_rejections(self) -> list[str]:
        lines = ["<b>Rejection Breakdown</b>"]
        try:
            stats = await self.store.get_radar_gate_stats()
            top = stats.get("top_rejection_reasons", [])
            if top:
                for rr in top[:6]:
                    reason = rr.get("reason", "?").replace("_", " ")
                    lines.append(f"  • {reason}: {rr.get('count', 0)}")
            else:
                lines.append("  <i>No rejections recorded yet.</i>")
        except Exception:
            lines.append("  <i>Rejection data unavailable.</i>")
        lines.append("")
        return lines

    async def _section_radar_salaries(self) -> list[str]:
        lines = ["<b>Salary Statistics</b>"]
        try:
            s = await self.store.get_salary_stats()
            if s.get("count", 0) > 0:
                lines.append(f"  Median: {s.get('median', 0):,}")
                lines.append(f"  Average: {s.get('avg', 0):,}")
                lines.append(f"  Roles with salary: {s.get('count', 0)}")
            else:
                lines.append("  <i>Not enough salary data yet.</i>")
        except Exception:
            lines.append("  <i>Salary data unavailable.</i>")
        lines.append("")
        return lines

    async def _section_radar_freshness(self) -> list[str]:
        lines = ["<b>Posting Freshness Lanes</b>"]
        try:
            stats = await self.store.get_radar_gate_stats()
            total = stats.get("total", 0)
            if total > 0:
                pct_urgent = stats.get("urgent", 0) / total * 100
                pct_review = stats.get("review", 0) / total * 100
                lines.append(f"  Urgent: {stats.get('urgent', 0)} ({pct_urgent:.0f}%)")
                lines.append(f"  Review: {stats.get('review', 0)} ({pct_review:.0f}%)")
            else:
                lines.append("  <i>No freshness data yet.</i>")
        except Exception:
            lines.append("  <i>Freshness data unavailable.</i>")
        lines.append("")
        return lines


def _esc(text: str) -> str:
    import html

    return html.escape(str(text))
