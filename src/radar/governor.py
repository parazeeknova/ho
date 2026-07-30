"""Process-wide LLM governor.

Every ContextManager.chat/json_chat call goes through one shared
token-bucket + RPM/TPM governor. The radar queue, startup enrichment,
analytics, and Telegram commands all share it.

On 429: all lanes pause, retried with jittered backoff.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

from src.configuration import get_config
from src.logging import get_logger

logger = get_logger("llm_governor")


@dataclass
class GovernorState:
    rpm_limit: int = 70
    tpm_limit: int = 50000
    max_in_flight: int = 2
    in_flight: int = 0
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    window_start: float = 0.0
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_429s: int = 0
    total_failures: int = 0
    # Reserved lane for interactive/commands
    reserved_requests_this_minute: int = 0
    reserved_max_per_minute: int = 5


_state = GovernorState()
_lock = asyncio.Lock()

_AGGREGATOR_DOMAINS = frozenset(
    {
        "techcrunch.com",
        "crunchbase.com",
        "linkedin.com",
        "producthunt.com",
        "wellfound.com",
        "glassdoor.com",
        "indeed.com",
        "ziprecruiter.com",
        "remoteok.com",
        "simplyhired.com",
        "news.ycombinator.com",
        "ycombinator.com",
        "sequoiacap.com",
        "a16z.com",
        "benchmark.com",
        "accel.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "medium.com",
        "blogspot.com",
        "wordpress.com",
    }
)


def init_governor() -> None:
    """Initialize governor limits from config."""
    cfg = get_config().llm_queue
    _state.rpm_limit = cfg.requests_per_minute
    _state.tpm_limit = cfg.estimated_tokens_per_minute
    _state.max_in_flight = cfg.max_in_flight


async def acquire_budget(
    estimated_tokens: int = 600,
    interactive: bool = False,
) -> None:
    """Acquire LLM budget. Block until tokens are available."""
    while True:
        now = time.monotonic()
        wait_secs: float | None = None

        async with _lock:
            if _state.cooldown_until > 0 and now < _state.cooldown_until:
                wait_secs = _state.cooldown_until - now
            elif now - _state.window_start >= 60.0:
                _state.window_start = now
                _state.requests_this_minute = 0
                _state.tokens_this_minute = 0
                _state.reserved_requests_this_minute = 0
            elif (
                (
                    interactive
                    and _state.reserved_requests_this_minute >= _state.reserved_max_per_minute
                )
                or (
                    not interactive
                    and _state.requests_this_minute
                    >= _state.rpm_limit - _state.reserved_max_per_minute
                )
                or _state.tokens_this_minute + estimated_tokens > _state.tpm_limit
            ):
                wait_secs = 60.0 - (now - _state.window_start)
            elif _state.in_flight >= _state.max_in_flight:
                wait_secs = 0.5
            else:
                _state.requests_this_minute += 1
                _state.tokens_this_minute += estimated_tokens
                _state.in_flight += 1
                _state.total_requests += 1
                if interactive:
                    _state.reserved_requests_this_minute += 1
                return

        if wait_secs is not None:
            await asyncio.sleep(wait_secs + 0.1)


def release_budget() -> None:
    """Release in-flight slot."""
    _state.in_flight = max(0, _state.in_flight - 1)


async def handle_429() -> None:
    """Called on a 429 response: pause all lanes."""
    async with _lock:
        _state.total_429s += 1
        cooldown = 30.0 + random.uniform(0, 10.0)
        _state.cooldown_until = max(_state.cooldown_until, time.monotonic() + cooldown)
        logger.warning("LLM governor: 429 received, cooldown", seconds=cooldown)


def get_governor_status() -> dict[str, Any]:
    return {
        "in_flight": _state.in_flight,
        "max_in_flight": _state.max_in_flight,
        "requests_this_minute": _state.requests_this_minute,
        "rpm_limit": _state.rpm_limit,
        "tokens_this_minute": _state.tokens_this_minute,
        "tpm_limit": _state.tpm_limit,
        "cooldown_active": _state.cooldown_until > time.monotonic(),
        "total_requests": _state.total_requests,
        "total_429s": _state.total_429s,
        "total_failures": _state.total_failures,
    }


def _is_429(err_msg: str) -> bool:
    return (
        "429" in err_msg
        or "rate limit" in err_msg.lower()
        or "too many requests" in err_msg.lower()
    )


def is_aggregator_domain(domain: str) -> bool:
    d = domain.lower().rstrip(".")
    if d in _AGGREGATOR_DOMAINS:
        return True
    for suffix in (".linkedin.com", ".blogspot.com", ".wordpress.com"):
        if d.endswith(suffix):
            return True
    return False
