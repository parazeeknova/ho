"""Cold-outreach card generation.

Produces a cold-outreach card only when there is a concrete public
hiring, funding, launch, engineering-growth, or open-role signal.
"""

from __future__ import annotations

from typing import Any

from src.logging import get_logger
from src.radar.core.models import ColdOutreachCard, JobCandidate

logger = get_logger("outreach")


def generate_outreach_card(candidate: JobCandidate) -> ColdOutreachCard | None:
    """Generate a cold-outreach card if sufficient signals exist."""

    hiring_signals = candidate.extra.get("hiring_signals", [])

    signal = _determine_primary_signal(candidate, hiring_signals)
    if signal is None:
        return None

    why_now = _build_why_now(candidate, signal)
    founder_profiles = _build_founder_profiles(candidate)
    contact_route = _find_best_contact_route(candidate)
    role_relevance = _build_role_relevance(candidate)
    source_links = _collect_source_links(candidate)

    confidence = _compute_outreach_confidence(candidate, signal, founder_profiles)

    return ColdOutreachCard(
        company=candidate.normalized_company,
        why_now=why_now,
        founder_profiles=founder_profiles,
        official_contact_route=contact_route,
        role_relevance=role_relevance,
        source_links=source_links,
        hiring_signal=signal,
        confidence=confidence,
    )


def _determine_primary_signal(
    candidate: JobCandidate,
    hiring_signals: list[dict[str, Any]],
) -> str | None:
    if candidate.is_urgent and candidate.is_accepted:
        return "open_role"

    if candidate.funding_stage and candidate.funding_stage not in ("N/A", "", "-"):
        return "funding"

    if candidate.extra.get("launch_signals"):
        return "launch"

    if candidate.extra.get("engineering_growth_signals"):
        return "engineering_growth"

    if hiring_signals and len(hiring_signals) > 0:
        return "open_role"

    if candidate.is_accepted and candidate.match_percent >= 60:
        return "open_role"

    return None


def _build_why_now(candidate: JobCandidate, signal: str) -> str:
    company = candidate.normalized_company
    stage = candidate.funding_stage or ""

    templates: dict[str, str] = {
        "open_role": (
            f"{company} has an active opening for {candidate.normalized_role}. "
            "The role aligns with your skills and the posting was verified as recent."
        ),
        "funding": (
            f"{company} recently raised {stage} funding. "
            "Companies post-funding typically expand engineering headcount within 90 days."
        ),
        "launch": (
            f"{company} has recent launch or product momentum. "
            "Growth-phase companies are receptive to proactive engineering outreach."
        ),
        "engineering_growth": (
            f"{company} shows engineering team growth signals. "
            "Expanding teams often have unlisted openings before public job postings."
        ),
    }

    return templates.get(signal, f"{company} appears to be in an active hiring cycle.")


def _build_founder_profiles(candidate: JobCandidate) -> list[dict[str, str]]:
    profiles = []
    for f in candidate.founders:
        profile = {}
        if isinstance(f, dict):
            name = f.get("name", "")
            if name:
                profile["name"] = name
            if f.get("linkedin_url"):
                profile["linkedin"] = f["linkedin_url"]
            if f.get("github_url"):
                profile["github"] = f["github_url"]
            if f.get("title"):
                profile["title"] = f["title"]
        if profile:
            profiles.append(profile)
    return profiles


def _find_best_contact_route(candidate: JobCandidate) -> str:
    for f in candidate.founders:
        if isinstance(f, dict) and f.get("linkedin_url"):
            return f["linkedin_url"]

    if candidate.extra.get("public_contact_email"):
        return candidate.extra["public_contact_email"]

    return candidate.direct_apply_url


def _build_role_relevance(candidate: JobCandidate) -> str:
    parts = []
    skills = candidate.matching_skills[:5]
    if skills:
        parts.append(f"Skills match: {', '.join(skills)}")
    if candidate.match_percent > 0:
        parts.append(f"Match score: {candidate.match_percent}%")
    if candidate.role_family:
        parts.append(f"Role family: {candidate.role_family.value}")
    return ". ".join(parts) if parts else "Relevant technical background"


def _collect_source_links(candidate: JobCandidate) -> list[str]:
    links = []
    if candidate.direct_apply_url:
        links.append(candidate.direct_apply_url)
    for f in candidate.founders:
        if isinstance(f, dict):
            if f.get("linkedin_url"):
                links.append(f["linkedin_url"])
            if f.get("github_url"):
                links.append(f["github_url"])
    src_links = candidate.extra.get("source_links", [])
    if isinstance(src_links, list):
        for link in src_links:
            if isinstance(link, str) and link not in links:
                links.append(link)
    return links[:5]


def _compute_outreach_confidence(
    candidate: JobCandidate,
    signal: str,
    founder_profiles: list[dict[str, str]],
) -> float:
    score = 0.5
    if signal == "open_role":
        score += 0.2
    elif signal == "funding":
        score += 0.15
    if founder_profiles:
        score += 0.1
    if candidate.match_percent >= 60:
        score += 0.1
    if candidate.is_urgent:
        score += 0.05
    return min(1.0, score)
