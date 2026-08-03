"""Pydantic schemas for the multi-agent job matching pipeline and input
canonicalization utilities.
"""

from __future__ import annotations

import re
from typing import Self
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator


class JobMatch(BaseModel):
    role: str = Field(max_length=120)
    company: str = Field(max_length=120)
    company_description: str = ""
    role_summary: str = ""
    match_percent: int = Field(ge=0, le=100)
    shortlist_probability: int = Field(ge=0, le=100)
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    jd_summary: str = ""
    salary: str | None = None
    posted_date: str | None = None
    apply_link: str | None = None
    location: str = ""
    is_remote: bool = False
    is_undergrad_friendly: bool = False
    verdict: str = Field(default="NO_MATCH")

    @model_validator(mode="after")
    def _clamp_verdict(self) -> Self:
        valid = {"STRONG_MATCH", "GOOD_MATCH", "WEAK_MATCH", "NO_MATCH"}
        if self.verdict not in valid:
            self.verdict = "NO_MATCH"
        return self


class CriticReview(BaseModel):
    passed: bool
    critique_reason: str = ""
    requires_rescore: bool = False


# Input canonicalization

_DIRECTORY_LANDING_DOMAINS = (
    "internshala.com",
    "web3.career",
)

_DIRECTORY_LANDING_PATHS = ("/jobs", "/careers", "/positions")

_HTML_BOILERPLATE_RE = re.compile(
    r"<(script|style|nav|footer|header|noscript)[^>]*>.*?</\1>",
    flags=re.DOTALL | re.IGNORECASE,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")

_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def canonicalize_markdown(raw: str, url: str | None = None) -> str | None:
    """Clean scraped markdown: strip HTML, drop known directory landing pages.

    Returns ``None`` when the page should be discarded entirely.
    """
    if not raw or len(raw.strip()) < 40:
        return None

    if url and _is_directory_landing(url):
        return None

    text = _strip_html(raw)

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)

    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip() or None


def _strip_html(text: str) -> str:
    text = _HTML_BOILERPLATE_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    return text


def _is_directory_landing(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return True

    host = parsed.netloc.lower()
    path = parsed.path.lower().rstrip("/") or "/"

    for domain in _DIRECTORY_LANDING_DOMAINS:
        if domain in host:
            return True

    return path in _DIRECTORY_LANDING_PATHS or path == "/"
