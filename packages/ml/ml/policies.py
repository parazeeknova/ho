"""Policy layer — separate contextual bandits per decision.

Separate policies (#16): DiscoveryPolicy (sources), QueryPolicy (search
templates), RecommendationPolicy (job -> recommend), BoardRoutingPolicy
(board -> render/proxy), ApplicationPolicy (job -> apply, gated later).

They share event infrastructure but never each other's reward streams.
"""

from __future__ import annotations

from .bandits import ThompsonPolicy

# Source family -> concrete sources. The review's "don't let the bandit decide
# the whole internet" ask: learn at the family level (ATS family high yield,
# HN low volume high quality, RemoteOK low interview rate, unknown ATS
# promising) AND at the source level (greenhouse vs ashby vs lever), instead of
# eight flat coarse arms.
DISCOVERY_HIERARCHY: dict[str, list[str]] = {
    "ats": [
        "greenhouse",
        "ashby",
        "lever",
        "workable",
        "smartrecruiters",
        "workday",
        "rippling",
        "teamtailor",
        "recruitee",
        "comeet",
        "jobscore",
        "jazzhr",
    ],
    "search": ["searxng_dork", "common_crawl", "web_lane"],
    "community": ["hn"],
    "startup_db": ["yc", "dealroom", "betalist"],
    "vc_portfolio": ["vc"],
    "aggregator": ["remoteok", "weworkremotely"],
    "career_pages": ["company_careers"],
}


class DiscoveryPolicy:
    """Hierarchical source-selection bandit.

    Two-level Thompson: first pick a source FAMILY (ats, search, community,
    startup_db, vc_portfolio, aggregator, career_pages), then a concrete source
    within that family. Each level keeps its own ThompsonPolicy, so the system
    can learn 'ATS family -> high yield, HN -> low volume but high quality,
    RemoteOK -> low interview rate, unknown ATS -> promising' rather than the
    flat 'search = good, YC = bad'.

    choose() returns (source, propensity). propensity is the product of the
    family-level π and the source-level π (a proper behavior-policy probability
    over the two-level decision). update() routes the reward to both levels.
    """

    def __init__(self, hierarchy: dict[str, list[str]] | None = None):
        self.hierarchy = hierarchy or DISCOVERY_HIERARCHY
        self.family_policy = ThompsonPolicy(list(self.hierarchy.keys()))
        self.source_policies: dict[str, ThompsonPolicy] = {
            fam: ThompsonPolicy(sources) for fam, sources in self.hierarchy.items()
        }

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        if not self.hierarchy:
            return "", 0.0
        family, pi_family = self.family_policy.choose(exploration)
        source_policy = self.source_policies.get(family)
        if source_policy is None or not source_policy.arms:
            return family, pi_family
        source, pi_source = source_policy.choose(exploration)
        # Product of the two-level probabilities = behavior propensity μ.
        return source, pi_family * pi_source

    def update(self, arm: str, reward: float) -> None:
        """Route a reward to the family that owns the arm AND the arm itself."""
        for fam, sources in self.hierarchy.items():
            if arm in sources:
                self.family_policy.update(fam, reward)
                self.source_policies[fam].update(arm, reward)
                return
        # Unknown arm: still credit the family_policy arms as a weak signal.
        self.family_policy.update("search", reward)


class QueryPolicy:
    """Query-template bandit. Arms = query templates. Reward = downstream
    quality (jobs that pass gate -> apply -> positive), not result count.
    Mandatory exploration + new-query generation preserved (#13)."""

    def __init__(self, arms: list[str] | None = None):
        # Expanded query arms (the review's "add an explicit site:* exploration
        # layer covering other ATS families and arbitrary career pages"): keep
        # the original behavioral templates and add site:* arms for every ATS
        # family plus a generic career-page arm.
        self.arms = arms or [
            "site_greenhouse",
            "site_ashby",
            "site_lever",
            "site_workable",
            "site_smartrecruiters",
            "site_workday",
            "site_rippling",
            "site_teamtailor",
            "site_recruitee",
            "site_comeet",
            "site_jobscore",
            "site_jazzhr",
            "career_page",
            "skill_role_startup",
            "skill_hiring_remote",
            "role_series_a",
            "role_founding",
        ]
        self.policy = ThompsonPolicy(self.arms)

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        return self.policy.choose(exploration)

    def update(self, arm: str, reward: float) -> None:
        self.policy.update(arm, reward)


class RecommendationPolicy:
    """Job -> recommend? bandit. Uses staged hiring reward. NEVER controls
    autonomous application directly (#15)."""

    def __init__(self, arms: list[str] | None = None):
        self.arms = arms or [
            "top_n",
            "adjacent_family",
            "unknown_source",
            "unknown_company",
            "novel_skill",
        ]
        self.policy = ThompsonPolicy(self.arms)

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        return self.policy.choose(exploration)

    def update(self, arm: str, reward: float) -> None:
        self.policy.update(arm, reward)


class BoardRoutingPolicy:
    """Board -> render/proxy? bandit. Infrastructure optimization, separate from
    recommendation (#16). Generalizes the existing board_routing_stats table."""

    def __init__(self, arms: list[str] | None = None):
        self.arms = arms or ["direct_render", "proxied_render", "static_fetch"]
        self.policy = ThompsonPolicy(self.arms)

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        return self.policy.choose(exploration)

    def update(self, arm: str, reward: float) -> None:
        self.policy.update(arm, reward)
