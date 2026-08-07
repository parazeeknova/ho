"""Structured exploration + counterfactual evaluation.

Asymmetric exploration (#14): recommendation can explore ~10%, autonomous
application only 1-3%. Never random garbage — structured mix (top-N,
adjacent role family, unknown source/company, novel skill). Propensity
logged on every action for IPS/SNIPS/doubly-robust counterfactual eval (#3).
"""

from __future__ import annotations

import random


def annealed_epsilon(step: int, initial: float = 0.30, mature: float = 0.05, anneal_steps: int = 500) -> float:
    if step >= anneal_steps:
        return mature
    t = step / max(anneal_steps, 1)
    return initial + (mature - initial) * t


def choose_explore(
    rng: random.Random,
    is_application: bool = False,
    eps_recommend: float = 0.10,
    eps_application: float = 0.02,
) -> bool:
    eps = eps_application if is_application else eps_recommend
    return rng.random() < eps


def exploration_strategy(
    rng: random.Random,
    has_adjacent: bool = False,
    has_unknown_source: bool = False,
    has_unknown_company: bool = False,
    has_novel_skill: bool = False,
) -> str:
    """Structured exploration mix — pick which exploration bucket to use."""
    buckets = ["top_n"] * 80
    if has_adjacent:
        buckets += ["adjacent_family"] * 10
    if has_unknown_source:
        buckets += ["unknown_source"] * 5
    if has_unknown_company:
        buckets += ["unknown_company"] * 3
    if has_novel_skill:
        buckets += ["novel_skill"] * 2
    return rng.choice(buckets)


def inverse_propensity_weight(propensity: float) -> float:
    """IPS weight = 1/P(action|context). Guards against zero/overlap."""
    p = max(min(float(propensity), 0.999), 0.001)
    return 1.0 / p


def snips_weight(
    propensity: float, action_probability: float, num_actions: int
) -> float:
    return inverse_propensity_weight(propensity) / action_probability * (1.0 / num_actions)
