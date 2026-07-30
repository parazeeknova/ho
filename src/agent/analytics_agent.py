"""AnalyticsAgent: Generates market intelligence, skill arbitrage reports,
and dynamic company hiring DNA profiles from PostgreSQL and Neo4j.
"""  # noqa: E501

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.logging import get_logger

if TYPE_CHECKING:
    from src.graph.graph_store import GraphStore
    from src.llm.context import ContextManager
    from src.memory.pgvector_store import MemoryStore

logger = get_logger("analytics_agent")

_HIRING_DNA_PROMPT = (
    "Analyze this aggregate hiring data for {company}. "
    "Write a punchy, 2-sentence 'Hiring DNA' profile summarizing their "
    "stack preferences, seniority bias, and remote flexibility. "
    "Do not use filler words.\n\n"
    "{data}"
)


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

    async def generate_market_report(self) -> str:
        lines: list[str] = ["<b>📊 Market Intelligence Report</b>", ""]

        lines.extend(await self._section_top_skills())
        lines.extend(await self._section_arbitrage())
        lines.extend(await self._section_vc_tier_list())
        lines.extend(await self._section_tech_momentum())
        lines.extend(await self._section_ats_blackhole())
        lines.extend(await self._section_marginal_valuation())
        lines.extend(await self._section_stealth_signals())

        count = await self.store.get_job_ledger_count()
        lines.append(f"<i>Total jobs tracked: {count}</i>")

        return "\n".join(lines)

    async def generate_resilient_report(self) -> list[str]:
        """Generate analytics with each section executing independently.

        A single failing section does not crash the entire report.
        Salary rows with malformed data are silently ignored.
        Returns list of message chunks so TelegramAgent can send
        them without hitting the 4000-char limit.
        """
        sections: list[list[str]] = []
        section_funcs = [
            ("🔥 Most In-Demand Skills", self._section_top_skills),
            ("📈 High-ROI Missing Skills", self._section_arbitrage),
            ("💸 VC Tier List", self._section_vc_tier_list),
            ("🚀 Pre-Trend Radar", self._section_tech_momentum),
            ("🕳️ ATS Black Hole Index", self._section_ats_blackhole),
            ("💰 Marginal Skill Valuation", self._section_marginal_valuation),
            ("🕵️ Stealth Hiring Signals", self._section_stealth_signals),
            ("🧮 Radar Gate Stats", self._section_radar_gate_stats),
        ]

        for name, func in section_funcs:
            try:
                result = await func()
                if result:
                    sections.append(result)
            except Exception as e:
                logger.warning("Analytics section failed", section=name, exception=str(e))
                sections.append(
                    [f"<b>{name}</b>", "  <i>Data unavailable for this section.</i>", ""]
                )

        try:
            count = await self.store.get_job_ledger_count()
            sections.append([f"<i>Total jobs tracked: {count}</i>"])
        except Exception:
            sections.append(["<i>Job count unavailable.</i>"])

        try:
            salary_stats = await self.store.get_salary_stats()
            if salary_stats.get("count", 0) > 0:
                sections.append(
                    [
                        "<b>💵 Salary Stats</b>",
                        f"  Median: {salary_stats.get('median', 0):,}",
                        f"  Average: {salary_stats.get('avg', 0):,}",
                        f"  Roles with salary data: {salary_stats.get('count', 0)}",
                        "",
                    ]
                )
        except Exception:
            pass

        return ["\n".join(s) for s in sections]

    async def _section_top_skills(self) -> list[str]:
        top = await self.store.get_top_skills(days=30, limit=12)
        lines = ["<b>🔥 Most In-Demand Skills</b>"]
        if top:
            for idx, item in enumerate(top, 1):
                lines.append(f"  {idx}. {_esc(item['skill'])} ({item['job_count']} jobs)")
        else:
            lines.append("  <i>No data yet — keep scraping.</i>")
        lines.append("")
        return lines

    async def _section_arbitrage(self) -> list[str]:
        arbitrage = await self.store.get_skill_arbitrage(min_match=50, max_match=69)
        lines = [
            "<b>📈 High-ROI Missing Skills</b>",
            "  <i>(These blocked you from near-miss roles)</i>",
        ]
        if arbitrage:
            for idx, item in enumerate(arbitrage[:8], 1):
                line = f"  {idx}. {_esc(item['skill'])} (blocked {item['miss_count']} jobs)"
                if item.get("avg_salary"):
                    line += f" — avg {item['avg_salary']:,}"
                lines.append(line)
        else:
            lines.append("  <i>No near-miss data yet.</i>")
        lines.append("")
        return lines

    async def _section_vc_tier_list(self) -> list[str]:
        vcs = await self.graph.get_vc_tier_list(limit=10)
        lines = ["<b>💸 VC Tier List — Who Funds Junior Hires</b>"]
        if vcs:
            for idx, vc in enumerate(vcs, 1):
                lines.append(
                    f"  {idx}. {_esc(vc['vc_firm'])} — "
                    f"{vc['junior_friendly_jobs']} junior jobs "
                    f"across {vc['portfolio_companies']} companies"
                )
        else:
            lines.append("  <i>Graph data too sparse — invest more sweeps.</i>")
        lines.append("")
        return lines

    async def _section_tech_momentum(self) -> list[str]:
        momentum = await self.store.get_tech_stack_momentum(limit=10)
        lines = ["<b>🚀 Pre-Trend Radar — MoM Skill Growth</b>"]
        if momentum:
            for idx, item in enumerate(momentum, 1):
                direction = "↑" if item["pct_growth"] >= 0 else "↓"
                lines.append(
                    f"  {idx}. {_esc(item['skill'])} "
                    f"{direction}{abs(item['pct_growth'])}% "
                    f"({item['current_count']} vs {item['prev_count']})"
                )
        else:
            lines.append("  <i>Need 60+ days of data for momentum calc.</i>")
        lines.append("")
        return lines

    async def _section_ats_blackhole(self) -> list[str]:
        ats = await self.store.get_ats_blackhole_index()
        lines = ["<b>🕳️ ATS Black Hole Index</b>"]
        if ats:
            for item in ats:
                risk = (
                    "🟢" if item["avg_match"] >= 60 else ("🟡" if item["avg_match"] >= 40 else "🔴")
                )
                lines.append(
                    f"  {risk} {_esc(item['ats_domain'])}: "
                    f"{item['avg_match']}% avg match, "
                    f"{item['avg_days_open']}d open, "
                    f"{item['job_count']} jobs"
                )
        else:
            lines.append("  <i>No ATS data aggregated yet.</i>")
        lines.append("")
        return lines

    async def _section_marginal_valuation(self) -> list[str]:
        val = await self.store.get_marginal_skill_valuation()
        lines = ["<b>💰 Marginal Skill Valuation (Salary Premium)</b>"]
        if val:
            for idx, item in enumerate(val[:8], 1):
                lines.append(
                    f"  {idx}. {_esc(item['skill'])} — "
                    f"median {item['median_salary']:,}, "
                    f"avg {item['avg_salary']:,} "
                    f"({item['job_count']} jobs)"
                )
        else:
            lines.append("  <i>Not enough salary data yet.</i>")
        lines.append("")
        return lines

    async def _section_stealth_signals(self) -> list[str]:
        stealth = await self.graph.detect_stealth_hiring_signals(limit=8)
        lines = ["<b>🕵️ Stealth Hiring Signals</b>"]
        lines.append("  <i>Funded companies with zero job postings → DM now</i>")
        if stealth:
            for idx, s in enumerate(stealth, 1):
                lines.append(
                    f"  {idx}. {_esc(s['company_name'])} "
                    f"(funding: {_esc(s['funding_stage'])}) "
                    f"[PR {s['pagerank']}]"
                )
        else:
            lines.append("  <i>No stealth signals detected.</i>")
        lines.append("")
        return lines

    async def _section_radar_gate_stats(self) -> list[str]:
        """Radar v2 rejection and eligibility statistics."""
        lines = ["<b>🧮 Radar Gate Stats</b>"]
        try:
            stats = await self.store.get_radar_gate_stats()
            lines.append(f"  Total candidates: {stats.get('total', 0)}")
            lines.append(f"  Accepted: {stats.get('accepted', 0)}")
            lines.append(f"  Near-miss: {stats.get('near_miss', 0)}")
            lines.append(f"  Rejected: {stats.get('rejected', 0)}")
            lines.append(f"  Urgent lane: {stats.get('urgent', 0)}")
            lines.append(f"  Review lane: {stats.get('review', 0)}")

            top_rejections = stats.get("top_rejection_reasons", [])
            if top_rejections:
                lines.append("  <b>Top Rejection Reasons:</b>")
                for rr in top_rejections[:5]:
                    reason = rr.get("reason", "?").replace("_", " ")
                    lines.append(f"    • {reason}: {rr.get('count', 0)}")
        except Exception as e:
            logger.warning("Radar gate stats failed", exception=str(e))
            lines.append("  <i>Radar data not yet available.</i>")
        lines.append("")
        return lines

    async def compute_hiring_dna(self, company_name: str) -> str:
        data = await self.store.get_company_aggregate_data(company_name)
        if not data or data.get("total_postings", 0) == 0:
            return f"<i>No hiring data yet for {_esc(company_name)}.</i>"

        prompt = _HIRING_DNA_PROMPT.replace("{company}", company_name)
        prompt = prompt.replace("{data}", json.dumps(data, indent=2))

        try:
            dna = await self.ctx.chat(prompt[:6000])
            dna = dna.strip().strip('"')
        except Exception as e:
            logger.error(
                "LLM hiring DNA failed",
                entity=company_name,
                exception=str(e),
            )
            dna = (
                f"<b>{_esc(company_name)}</b> has {data['total_postings']} tracked "
                f"postings with avg match {data['avg_match']}%."
            )

        try:
            from src.graph.entity import make_canonical_company_id

            node = await self.graph.get_node(company_name.lower())
            if node is None:
                cid = make_canonical_company_id(company_name)
                node = await self.graph.get_node(cid)
            if node:
                node.data["hiring_fingerprint"] = dna
                await self.graph.upsert_node(node)
        except Exception as e:
            logger.error(
                "Failed to persist hiring DNA to graph",
                entity=company_name,
                exception=str(e),
            )

        avg = data["avg_match"]
        best = data["best_match"]
        total = data["total_postings"]
        skills = data.get("top_matching_skills", [])

        lines = [
            f"<b>🧬 Hiring DNA: {_esc(company_name)}</b>",
            "",
            f"Tracked: <b>{total}</b> postings  |  Avg: <b>{avg}%</b>  |  Best: <b>{best}%</b>",
        ]
        if skills:
            skill_list = ", ".join(s["skill"] for s in skills[:5])
            lines.append(f"Key skills matched: {_esc(skill_list)}")
        lines.extend(["", f"<blockquote>{_esc(dna)}</blockquote>"])

        return "\n".join(lines)


def _esc(text: str) -> str:
    import html

    return html.escape(str(text))
