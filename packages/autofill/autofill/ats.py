"""ATS platform classification from a job posting URL.

Mirrors the runner's adapter registry (packages/autofill/node/runner.ts) so the
enqueued job is tagged with the same platform the browser adapter will pick.
This is what makes per-platform metrics and the specialized adapters measurable:
without it every job falls through to the generic adapter and classification is
dead effort.
"""

from __future__ import annotations

import re

# Ordered; first regex match wins, matching the runner's adapterRegistry.
_ATS_RULES: list[tuple[str, str]] = [
    ("greenhouse", r"greenhouse\.io"),
    ("ashby", r"jobs\.ashbyhq\.com"),
    ("lever", r"jobs\.lever\.co"),
    ("workday", r"myworkdayjobs\.com"),
]


def classify_ats(url: str) -> str:
    """Return the ATS platform for a posting URL, or ``"generic"``.

    Examples:
        https://boards.greenhouse.io/neo4j/jobs/123  -> "greenhouse"
        https://jobs.ashbyhq.com/replit/abc           -> "ashby"
        https://jobs.lever.co/acme/dev                 -> "lever"
        https://acme.wd12.myworkdayjobs.com/role       -> "workday"
        https://some-company.com/careers               -> "generic"
    """
    if not url:
        return "generic"
    low = (url or "").lower()
    for platform, pattern in _ATS_RULES:
        if re.search(pattern, low):
            return platform
    return "generic"
