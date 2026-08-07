"""Cross-reference and multi-source verification pipeline.

Features:
  - Multi-source verification boost: when 2+ independent connectors
    confirm the same fact, confidence approaches 1.0.
  - Contradiction detection: conflicting facts are flagged for re-verification
    and confidence is decreased.
  - Confidence decay: periodic decay for stale, unverified nodes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.llm.context import VERIFY_SCHEMA, ContextManager
from src.logging import get_logger

logger = get_logger("verifier")

VERIFY_PROMPT = """Compare two scraped job listings. Are they the same job?
Original: {original}
Alternate: {alternate}
Return valid JSON matching the required schema."""


@dataclass
class FactVerification:
    fact_key: str
    value: Any
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    last_verified: datetime = field(default_factory=lambda: datetime.now(UTC))
    needs_reverification: bool = False

    def boost_confidence(self, new_source: str, value: Any) -> float:
        """Multi-source confidence boost.

        - 2+ independent sources confirming same fact: boost toward 1.0.
        - Contradicting values from different sources: decrease and flag.
        - Each additional confirming source adds diminishing returns.
        """
        if new_source not in self.sources:
            self.sources.append(new_source)
            self.last_verified = datetime.now(UTC)

        if value is not None and self.value is not None and value != self.value:
            self.contradictions.append(
                {
                    "source": new_source,
                    "value": value,
                    "existing_value": self.value,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            self.confidence = max(0.1, self.confidence - 0.15)
            self.needs_reverification = True
            return self.confidence

        source_count = len(self.sources)
        if source_count >= 3:
            self.confidence = min(1.0, 0.95)
            self.needs_reverification = False
        elif source_count == 2:
            self.confidence = min(1.0, self.confidence + 0.25)
            self.needs_reverification = False
        else:
            self.confidence = min(0.75, self.confidence)

        return self.confidence

    def decay(self, max_age_days: int = 30) -> float:
        """Decay confidence for old, unverified facts.

        Confidence drops gradually after max_age_days without re-verification.
        """
        age = (datetime.now(UTC) - self.last_verified).days
        if age > max_age_days:
            decay_factor = max(0.1, 1.0 - (age - max_age_days) / 60.0)
            self.confidence = max(0.1, self.confidence * decay_factor)
            if age > max_age_days * 2:
                self.needs_reverification = True
        return self.confidence


class MultiSourceVerifier:
    """Tracks facts across connectors and applies confidence rules.

    Manages a registry of fact verifications keyed by (entity_id, field).
    When the same fact is discovered by multiple connectors, confidence
    rises; contradictions are flagged and confidence decreases.
    """

    def __init__(self) -> None:
        self._facts: dict[str, FactVerification] = {}

    def _key(self, entity_id: str, field: str) -> str:
        return f"{entity_id}:{field}"

    def register_fact(
        self,
        entity_id: str,
        field: str,
        value: Any,
        source: str,
        initial_confidence: float = 0.5,
    ) -> FactVerification:
        """Register or update a fact from a connector source.

        Returns the updated FactVerification with boosted or penalized confidence.
        """
        key = self._key(entity_id, field)
        if key not in self._facts:
            fv = FactVerification(
                fact_key=field,
                value=value,
                sources=[source],
                confidence=initial_confidence,
            )
            self._facts[key] = fv
            return fv

        fv = self._facts[key]
        fv.boost_confidence(source, value)
        return fv

    def get_confidence(self, entity_id: str, field: str) -> float:
        key = self._key(entity_id, field)
        fv = self._facts.get(key)
        return fv.confidence if fv else 0.0

    def get_fact(self, entity_id: str, field: str) -> FactVerification | None:
        return self._facts.get(self._key(entity_id, field))

    async def decay_all(self, max_age_days: int = 30) -> int:
        """Run confidence decay on all tracked facts.

        Returns count of facts flagged for re-verification.
        """
        flagged = 0
        for fv in self._facts.values():
            fv.decay(max_age_days)
            if fv.needs_reverification:
                flagged += 1
        logger.info(
            "Verifier decay complete",
            extra={"total_facts": len(self._facts), "flagged": flagged},
        )
        return flagged

    def get_pending_reverification(self) -> list[FactVerification]:
        return [fv for fv in self._facts.values() if fv.needs_reverification]

    def source_count(self, entity_id: str, field: str) -> int:
        fv = self._facts.get(self._key(entity_id, field))
        return len(fv.sources) if fv else 0


async def _scrape_alternate(role: str, company: str, original_url: str) -> str:
    """Fetch an alternate source describing the same role.

    Replaces Firecrawl's search+scrape: derive likely alternate URLs from the
    original posting (company site + careers paths) and markdownify the first
    that resolves. Returns '' when none yield usable content.
    """
    from urllib.parse import urlparse

    from src.render import markdownify

    parsed = urlparse(original_url or "")
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else ""
    candidates = [original_url]
    if origin:
        candidates += [
            f"{origin}/careers",
            f"{origin}/jobs",
            f"{origin}/about/careers",
            f"{origin}/careers?q={role}",
        ]
    seen: set[str] = set()
    for alt_url in candidates:
        if not alt_url or alt_url in seen:
            continue
        seen.add(alt_url)
        try:
            text = await markdownify(alt_url, timeout=20.0)
            if text and len(text) >= 100:
                return text
        except Exception:
            continue
    return ""


async def _verify_one(
    role: str,
    company: str,
    original_url: str,
    ctx: ContextManager,
    sem: asyncio.Semaphore,
) -> bool:
    async with sem:
        alt_content = await _scrape_alternate(role, company, original_url)
        if not alt_content:
            return True

        prompt = VERIFY_PROMPT.replace("{original}", f"{role} @ {company} [{original_url}]")
        prompt = prompt.replace("{alternate}", alt_content[:3000])

        result = await ctx.json_chat(prompt, schema=VERIFY_SCHEMA)
        if isinstance(result, dict):
            confidence = result.get("confidence", 50)
            return confidence >= 30
        return True


async def verify_jobs(
    jobs: list[dict],
    ctx: ContextManager,
    concurrency: int = 4,
) -> list[dict]:
    if not jobs:
        return []

    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for j in jobs:
        role = j.get("role", "?")
        company = j.get("company", "?")
        url = j.get("source_url", "")
        tasks.append(_verify_one(role, company, url, ctx, sem))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified = []
    for j, result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException) or result is True:
            verified.append(j)
        else:
            print(f"    [red]✗ FAILED: {j.get('role')} @ {j.get('company')}[/red]")

    return verified
