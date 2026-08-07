"""Contextual bandits — Thompson Sampling + LinUCB.

Separate policies: RecommendationPolicy, DiscoveryPolicy, QueryPolicy,
BoardRoutingPolicy. They share the event infrastructure, never each other's
reward streams.

Discovery bandit reward: hierarchical quick tier (+0.01 dedup → +20 offer).
Recommendation bandit reward: staged hiring funnel.
Board-routing bandit: render/block/application outcome (already in board_routing_stats).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class BetaArm:
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        # Bernoulli Thompson: treat reward>0 as success, else failure.
        # For staged rewards, normalize to [0,1] via reward/max_reward.
        r = max(0.0, min(1.0, reward / 100.0))
        self.alpha += r
        self.beta += 1 - r
        self.pulls += 1


class ThompsonPolicy:
    def __init__(self, arms: list[str]):
        self.arms: dict[str, BetaArm] = {a: BetaArm() for a in arms}

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        if random.random() < exploration:
            arm = random.choice(list(self.arms.keys()))
            return arm, 1.0 / len(self.arms)
        best = max(self.arms, key=lambda a: self.arms[a].sample())
        return best, 0.95  # propensity approx

    def update(self, arm: str, reward: float) -> None:
        if arm in self.arms:
            self.arms[arm].update(reward)


@dataclass
class LinUCBArm:
    d: int
    A: Any = None  # set lazily
    b: Any = None

    def __post_init__(self):
        import numpy as np

        self.A = np.eye(self.d)
        self.b = np.zeros(self.d)


class LinUCBPolicy:
    def __init__(self, arms: list[str], d: int, alpha: float = 1.0):
        self.arms: dict[str, LinUCBArm] = {a: LinUCBArm(d=d) for a in arms}
        self.alpha = alpha
        self.d = d

    def choose(self, context: list[float]) -> tuple[str, float]:
        import numpy as np

        ctx = np.array(context)
        best_arm = None
        best_score = -1e9
        for name, arm in self.arms.items():
            A_inv = np.linalg.inv(arm.A)
            theta = A_inv @ arm.b
            p = float(theta @ ctx + self.alpha * np.sqrt(ctx @ A_inv @ ctx))
            if p > best_score:
                best_score = p
                best_arm = name
        return best_arm or next(iter(self.arms)), 0.9

    def update(self, arm: str, context: list[float], reward: float) -> None:
        import numpy as np

        if arm not in self.arms:
            return
        a = self.arms[arm]
        ctx = np.array(context)
        a.A += np.outer(ctx, ctx)
        a.b += reward * ctx
