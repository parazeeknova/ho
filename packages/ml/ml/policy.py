"""Policy engine — EV ranking over calibrated funnel probabilities.

Model-vs-policy separation (#26): the model produces P(screen), P(interview),
P(offer); the POLICY decides action. Replacing the policy does not require
retraining the model.

EV(job) = P(screen) * P(interview|screen) * P(offer|interview) * offer_utility
          - application_cost          (#27)

Hard-constraint layer stays ABOVE ML (#28): work auth, location, salary floor,
seniority ceiling, blacklist, user exclusions. This policy only ranks the
surviving valid set.
"""

from __future__ import annotations

from typing import Any


def application_cost(tokens: int = 2000, base: float = 0.5) -> float:
    """Normalized cost of one application (time + LLM tokens + browser + human)."""
    return base + tokens / 100_000.0


def expected_utility(
    p_screen: float,
    p_interview_given_screen: float,
    p_offer_given_interview: float,
    offer_utility: float,
    cost: float | None = None,
) -> float:
    cost = cost if cost is not None else application_cost()
    return (
        p_screen * p_interview_given_screen * p_offer_given_interview * offer_utility - cost
    )


def ev_rank_key(candidate: dict[str, Any]) -> float:
    """Rank key for a candidate given calibrated probabilities on the candidate."""
    p_screen = float(candidate.get("p_screen", 0.5))
    p_int = float(candidate.get("p_interview", 0.25))
    p_offer = float(candidate.get("p_offer", 0.05))
    utility = float(candidate.get("offer_utility", 120000.0))
    cost = float(candidate.get("application_cost", application_cost()))
    return expected_utility(p_screen, p_int, p_offer, utility, cost)


# Hard constraints — deterministic, never learned away (#28)
def satisfies_hard_constraints(candidate: dict[str, Any]) -> bool:
    if candidate.get("blocked_company"):
        return False
    if not candidate.get("work_authorized", True):
        return False
    floor = candidate.get("salary_floor")
    sal = candidate.get("salary_amount")
    if floor is not None and sal is not None and sal < floor:
        return False
    return True


def rank_by_ev(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [c for c in candidates if satisfies_hard_constraints(c)]
    return sorted(valid, key=ev_rank_key, reverse=True)
