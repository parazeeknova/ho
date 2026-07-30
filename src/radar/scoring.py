"""Underdog scoring using public evidence.

Ranks companies outside the high-competition 'major tech' cohort higher
based on signals that reduce application density.
"""

from __future__ import annotations

from src.radar.models import JobCandidate

_MAJOR_TECH = frozenset(
    {
        "google",
        "microsoft",
        "amazon",
        "apple",
        "meta",
        "netflix",
        "nvidia",
        "openai",
        "anthropic",
        "stripe",
        "airbnb",
        "uber",
        "spotify",
        "atlassian",
        "databricks",
        "snowflake",
        "palantir",
        "cloudflare",
        "roblox",
        "snap",
        "pinterest",
        "reddit",
        "doordash",
        "instacart",
        "salesforce",
        "adobe",
        "ibm",
        "oracle",
        "intel",
        "amd",
    }
)


def compute_underdog_score(candidate: JobCandidate) -> float:
    """Return 0–1 score; higher = less competitive applicant pool (better for you)."""
    score = 0.5
    company = candidate.normalized_company.lower().strip()

    if company not in _MAJOR_TECH:
        score += 0.20

    stage = (candidate.funding_stage or "").lower()
    if any(kw in stage for kw in ("seed", "pre-seed", "series a")):
        score += 0.15
    elif "series b" in stage:
        score += 0.10

    if candidate.extra.get("discovered_from_ecosystem"):
        score += 0.10

    if candidate.salary_annual_usd and candidate.salary_annual_usd > 60000:
        score += 0.05

    return min(1.0, score)
