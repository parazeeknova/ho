"""Composite ranking: eligibility × freshness × fit × salary × underdog × trust."""

from __future__ import annotations

from src.radar.models import FreshnessLane, JobCandidate

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
    score = 0.5
    company = candidate.normalized_company.lower().strip()

    if company in _MAJOR_TECH:
        score -= 0.15

    stage = (candidate.funding_stage or "").lower()
    if any(kw in stage for kw in _EARLY_STAGE):
        score += 0.20
    elif "series b" in stage:
        score += 0.12
    elif any(kw in stage for kw in ("series c", "series d")):
        score += 0.05

    origin = candidate.extra.get("discovery_origin", "")
    if origin in ("yc_directory", "vc_portfolio", "wellfound"):
        score += 0.12
    elif origin in (
        "searxng_hiring",
        "searxng_funding",
        "searxng_launch",
        "search_ats",
        "search_startup",
        "search_founder",
    ):
        score += 0.15

    signals = candidate.osint_signals or []
    if any("hiring" in str(s).lower() for s in signals):
        score += 0.05
    if any("raised" in str(s).lower() or "funding" in str(s).lower() for s in signals):
        score += 0.04

    if (
        company not in _MAJOR_TECH
        and candidate.salary_annual_usd
        and candidate.salary_annual_usd > 80000
    ):
        score += 0.08

    return min(1.0, score)


def rank_score(candidate: JobCandidate) -> float:
    """Composite rank: eligibility × freshness × fit × salary × underdog × trust.

    Higher = better. Used for sort ordering.
    """
    eligibility = _eligibility_score(candidate)
    freshness = _freshness_score(candidate)
    fit = candidate.match_percent / 100.0 if candidate.match_percent else 0.3
    salary = _salary_score(candidate)
    underdog = compute_underdog_score(candidate)
    trust = candidate.source_confidence

    return (
        eligibility * 0.30
        + freshness * 0.20
        + fit * 0.15
        + salary * 0.15
        + underdog * 0.10
        + trust * 0.10
    )


def _eligibility_score(candidate: JobCandidate) -> float:
    score = 0.5
    if candidate.sponsors_visa:
        score += 0.25
    if candidate.extra.get("relocation", False):
        score += 0.15
    if candidate.is_remote:
        score += 0.10
    return min(1.0, score)


def _freshness_score(candidate: JobCandidate) -> float:
    if candidate.freshness_lane == FreshnessLane.URGENT:
        return 1.0
    if candidate.freshness_lane == FreshnessLane.REVIEW:
        return 0.5
    return 0.2


def _salary_score(candidate: JobCandidate) -> float:
    sal = candidate.salary_annual_usd
    if sal is None:
        return 0.3
    if sal >= 120000:
        return 1.0
    if sal >= 80000:
        return 0.8
    if sal >= 60000:
        return 0.6
    if sal >= 40000:
        return 0.4
    return 0.2
