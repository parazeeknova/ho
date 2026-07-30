"""Underdog scoring using public evidence.

Ranks companies by opportunity: lower application density = higher score.
Based on: startup ecosystem origin, funding stage/recency, company age,
hiring momentum signals, and major-tech penalty.
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

_EARLY_STAGE = frozenset({"seed", "pre-seed", "series a"})


def compute_underdog_score(candidate: JobCandidate) -> float:
    """0-1 score; higher = less competitive (better opportunity)."""
    score = 0.5
    company = candidate.normalized_company.lower().strip()

    # Major-tech penalty
    if company in _MAJOR_TECH:
        score -= 0.15

    # Funding stage: earlier = more underdog
    stage = (candidate.funding_stage or "").lower()
    if any(kw in stage for kw in _EARLY_STAGE):
        score += 0.20
    elif "series b" in stage:
        score += 0.12
    elif any(kw in stage for kw in ("series c", "series d")):
        score += 0.05

    # Ecosystem origin (discovered, not seed-listed)
    origin = candidate.extra.get("discovery_origin", "")
    if origin in ("yc_directory", "vc_portfolio", "wellfound"):
        score += 0.12
    elif origin in ("searxng_hiring", "searxng_funding", "searxng_launch"):
        score += 0.15

    # Hiring momentum from osint signals
    signals = candidate.osint_signals or []
    if any("hiring" in str(s).lower() for s in signals):
        score += 0.05
    if any("raised" in str(s).lower() or "funding" in str(s).lower() for s in signals):
        score += 0.04

    # High-salary roles at non-major companies = premium underdog
    if (
        company not in _MAJOR_TECH
        and candidate.salary_annual_usd
        and candidate.salary_annual_usd > 80000
    ):
        score += 0.08

    return min(1.0, score)
