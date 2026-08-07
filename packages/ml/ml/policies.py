"""Policy layer — separate contextual bandits per decision.

Separate policies (#16): DiscoveryPolicy (sources), QueryPolicy (search
templates), RecommendationPolicy (job -> recommend), BoardRoutingPolicy
(board -> render/proxy), ApplicationPolicy (job -> apply, gated later).

They share event infrastructure but never each other's reward streams.
"""

from __future__ import annotations

from .bandits import ThompsonPolicy


class DiscoveryPolicy:
    """Source-selection bandit. Arms = discovery sources. Reward = hierarchical
    quick tier (+0.01 dedup -> +20 offer, #12). Attribution: primary source
    gets full reward (#11)."""

    def __init__(self, arms: list[str] | None = None):
        self.arms = arms or [
            "dealroom",
            "yc",
            "vc",
            "hn",
            "remoteok",
            "weworkremotely",
            "betalist",
            "search",
        ]
        self.policy = ThompsonPolicy(self.arms)

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        return self.policy.choose(exploration)

    def update(self, arm: str, reward: float) -> None:
        self.policy.update(arm, reward)


class QueryPolicy:
    """Query-template bandit. Arms = query templates. Reward = downstream
    quality (jobs that pass gate -> apply -> positive), not result count.
    Mandatory exploration + new-query generation preserved (#13)."""

    def __init__(self, arms: list[str] | None = None):
        self.arms = arms or [
            "site_greenhouse",
            "site_ashby",
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
