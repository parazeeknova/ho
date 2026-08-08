"""Source attribution — who gets credit when a job succeeds.

When the same canonical job is discovered by multiple sources, only the
primary (first) discoverer gets the full reward; secondaries are recorded
but not rewarded. This prevents the source bandit learning garbage.

Also handles the “hiring vs discovery” reward split: the source bandit
learns on quick discovery signals, the hiring bandit on long-term outcomes.
"""

from __future__ import annotations

from typing import Any


def resolve_primary_source(existing_sources: list[str], new_source: str) -> tuple[str, list[str]]:
    """Return (primary, secondaries) for a job's source list."""
    if not existing_sources:
        return new_source, []
    # First discoverer stays primary.
    primary = existing_sources[0]
    secondaries = [s for s in existing_sources if s != primary]
    if new_source not in existing_sources:
        secondaries.append(new_source)
    return primary, secondaries


def attribution_for_job(job_row: dict[str, Any]) -> dict[str, Any]:
    """Build attribution jsonb for a decision event from a job row."""
    primary = job_row.get("primary_discovery_source") or job_row.get("source") or "unknown"
    secondaries = job_row.get("secondary_sources") or []
    return {"primary": primary, "secondaries": secondaries}
