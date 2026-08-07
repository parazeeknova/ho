"""Contextual bandits — Thompson Sampling + LinUCB with correct propensities.

The critical fix: propensities must be the ACTUAL action-selection probability
π(a|x), not a fake constant. For Thompson Sampling we use an ε-mixture:

    π(a|x) = (1-ε) * P(a sampled posterior argmax) + ε / |A|

and we MONTE-CARLO estimate the Thompson distribution over the posterior.
For LinUCB we use a softmax over UCB scores:

    π(a|x) = exp(score_a / τ) / Σ_j exp(score_j / τ)

so the logged propensity is mathematically defined and IPS/SNIPS are valid.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BetaArm:
    alpha: float = 1.0
    beta: float = 1.0
    pulls: int = 0

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: float) -> None:
        r = max(0.0, min(1.0, reward / 100.0))
        self.alpha += r
        self.beta += 1 - r
        self.pulls += 1


class ThompsonPolicy:
    """Thompson Sampling with ε-mixture and MC-estimated action probabilities.

    choose() returns (arm, propensity). propensity is the true π(a|x):
        π(a|x) = (1-ε) * MC_P(a) + ε / |A|
    where MC_P(a) is the empirical frequency that arm a's posterior draw is
    the argmax over N Monte-Carlo samples.
    """

    def __init__(self, arms: list[str], mc_samples: int = 100):
        self.arms: dict[str, BetaArm] = {a: BetaArm() for a in arms}
        self.mc_samples = mc_samples

    def _mc_thompson_dist(self) -> dict[str, float]:
        """P(a) = freq. arm a's posterior draw is the max over MC samples."""
        counts = {a: 0.0 for a in self.arms}
        names = list(self.arms)
        for _ in range(self.mc_samples):
            draws = {a: self.arms[a].sample() for a in names}
            best = max(draws.keys(), key=lambda a: draws[a])
            counts[best] += 1.0
        total = sum(counts.values()) or 1.0
        return {a: c / total for a, c in counts.items()}

    def choose(self, exploration: float = 0.0) -> tuple[str, float]:
        if not self.arms:
            return "", 0.0
        eps = max(0.0, min(1.0, exploration))
        dist = self._mc_thompson_dist()
        n = len(self.arms)
        # ε-mixture: π(a|x) = (1-ε)*MC_P(a) + ε/n
        pi = {a: (1 - eps) * p + eps / n for a, p in dist.items()}
        if random.random() < eps:
            arm = random.choice(list(self.arms.keys()))
            return arm, pi[arm]
        arm = max(dist.keys(), key=lambda a: dist[a])
        return arm, pi[arm]

    def update(self, arm: str, reward: float) -> None:
        if arm in self.arms:
            self.arms[arm].update(reward)


class LinUCBPolicy:
    """LinUCB with a softmax propensity policy (true π(a|x))."""

    def __init__(self, arms: list[str], d: int, alpha: float = 1.0, tau: float = 1.0):
        self.arms: dict[str, LinUCBArm] = {a: LinUCBArm(d=d) for a in arms}
        self.alpha = alpha
        self.tau = tau
        self.d = d

    def _ucb_scores(self, context: list[float]) -> dict[str, float]:
        ctx = np.array(context, dtype=float)
        scores: dict[str, float] = {}
        for name, arm in self.arms.items():
            A_inv = np.linalg.inv(arm.A)
            theta = A_inv @ arm.b
            p = float(theta @ ctx + self.alpha * np.sqrt(ctx @ A_inv @ ctx))
            scores[name] = p
        return scores

    def _softmax(self, scores: dict[str, float], exploration: float = 0.0) -> dict[str, float]:
        if self.tau <= 0:
            raise ValueError("tau must be > 0 for softmax propensity")
        vals = np.array(list(scores.values()), dtype=float)
        vals = (vals - vals.max()) / self.tau  # numeric stability
        exps = np.exp(vals)
        base = exps / exps.sum()
        n = len(scores)
        eps = max(0.0, min(1.0, exploration))
        # ε-mixture over the softmax.
        pi = {name: (1 - eps) * float(base[i]) + eps / n for i, name in enumerate(scores)}
        return pi

    def choose(self, context: list[float], exploration: float = 0.0) -> tuple[str, float]:
        scores = self._ucb_scores(context)
        pi = self._softmax(scores, exploration)
        if random.random() < exploration:
            arm = random.choice(list(self.arms.keys()))
        else:
            arm = max(pi.keys(), key=lambda a: pi[a])
        return arm, pi[arm]

    def update(self, arm: str, context: list[float], reward: float) -> None:
        if arm not in self.arms:
            return
        a = self.arms[arm]
        ctx = np.array(context, dtype=float)
        a.A += np.outer(ctx, ctx)
        a.b += reward * ctx


@dataclass
class LinUCBArm:
    d: int
    A: Any = None
    b: Any = None

    def __post_init__(self):
        self.A = np.eye(self.d)
        self.b = np.zeros(self.d)
