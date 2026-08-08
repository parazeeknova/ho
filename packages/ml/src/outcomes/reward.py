"""Reward layer — maps semantic outcome events to numeric rewards.

The email/classifier subsystem emits *semantic* events (screening, offer, …);
this module converts them to numbers. Keeping the mapping here (not inside
gmail_push) makes the reward function replaceable without touching ingestion.
"""

from __future__ import annotations

from ml import GAMMA, REWARD_MAP, SOURCE_REWARD_MAP


def reward_for(event_type: str, source_tier: str = "hiring") -> float:
    """Numeric reward for an event type. Tier selects the reward map."""
    m = SOURCE_REWARD_MAP if source_tier == "source" else REWARD_MAP
    return float(m.get(event_type, 0.0))


def discounted_reward(
    event_type: str,
    event_ts: float,
    decision_ts: float,
    gamma: float = GAMMA,
    source_tier: str = "hiring",
) -> float:
    """Reward discounted by time-to-outcome: reward * gamma ** delta_days."""
    base = reward_for(event_type, source_tier=source_tier)
    delta_days = max(0.0, (event_ts - decision_ts) / 86400.0)
    return base * (gamma**delta_days)


def is_terminal_outcome(event_type: str) -> bool:
    return event_type in {
        "offer",
        "rejection_email",
        "withdrawn",
        "interview",
        "screening",
        "screening_email",
    }
