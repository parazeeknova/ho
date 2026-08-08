"""Process-wide LLM governor.

Every ContextManager.chat/json_chat call goes through one shared
token-bucket + RPM/TPM governor. The radar queue, startup enrichment,
analytics, and Discord commands all share it.

The provider's per-minute quota is shared across the radar processes, so
radar reserves a configurable fraction of it (LLM_BUDGET_RADAR_RPM/TPM)
and tracks usage atomically in Redis across all radar processes. When
Redis is unavailable the governor degrades to process-local accounting.

On 429: all lanes pause, retried with jittered backoff.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass
from typing import Any

from src.configuration import get_config
from src.logging import get_logger

logger = get_logger("llm_governor")


@dataclass
class GovernorState:
    rpm_limit: int = 90
    tpm_limit: int = 180000
    max_in_flight: int = 15
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
    # Shared (cross-process) budget state
    shared_enabled: bool = True
    shared_rpm_limit: int = 70
    shared_tpm_limit: int = 140000


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


class _RedisBudget:
    """Cross-process shared per-minute LLM budget tracked in Redis.

    Reserve is atomic (Lua): no two radar processes can overspend the
    radar share of the provider quota. Degrades to process-local
    accounting when Redis is unreachable, so radar never blocks on
    infrastructure it does not strictly need.
    """

    _RESERVE_SCRIPT = """
local req = tonumber(redis.call('GET', KEYS[1]) or '0')
local tok = tonumber(redis.call('GET', KEYS[2]) or '0')
if req >= tonumber(ARGV[1]) then return 0 end
if (tok + tonumber(ARGV[2])) > tonumber(ARGV[3]) then return 0 end
redis.call('INCR', KEYS[1])
redis.call('INCRBY', KEYS[2], ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[4])
return 1
"""

    def __init__(self) -> None:
        cfg = get_config().llm_queue
        self._url = cfg.budget_redis_url
        self._enabled = cfg.budget_redis_enabled
        self._client: Any | None = None
        self._warned = False
        self._rpm = cfg.budget_radar_rpm
        self._tpm = cfg.budget_radar_tpm

    def _connect(self) -> Any | None:
        if not self._enabled:
            return None
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self._url,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.5,
                    decode_responses=True,
                )
            except Exception:
                self._client = None
        return self._client

    @staticmethod
    def _minute_key() -> str:
        return time.strftime("%Y%m%d%H%M", time.gmtime())

    async def reserve(self, estimated_tokens: int) -> bool:
        """Atomically reserve budget. True = granted, False = try later.

        Unreachable Redis grants (fail-open) so the pipeline never stalls.
        """
        client = self._connect()
        if client is None:
            return True
        try:
            prefix = "llm_budget:" + self._minute_key()
            granted = await client.eval(
                self._RESERVE_SCRIPT,
                2,
                f"{prefix}:req",
                f"{prefix}:tok",
                self._rpm,
                estimated_tokens,
                self._tpm,
                120,
            )
            return bool(granted)
        except Exception as exc:
            if not self._warned:
                self._warned = True
                logger.warning("LLM governor: Redis budget unavailable, local-only", err=str(exc))
            return True

    async def set_cooldown(self, seconds: float) -> None:
        client = self._connect()
        if client is None:
            return
        with contextlib.suppress(Exception):
            await client.set(
                "llm_budget:cooldown", str(time.monotonic() + seconds), ex=seconds + 10
            )

    async def cooldown_until(self) -> float:
        client = self._connect()
        if client is None:
            return 0.0
        try:
            raw = await client.get("llm_budget:cooldown")
            return float(raw) if raw else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def seconds_to_minute_rollover() -> float:
        now = time.time()
        return 60.0 - (now % 60.0) + 0.5


_redis_budget = _RedisBudget()


def init_governor() -> None:
    """Initialize governor limits from config."""
    cfg = get_config().llm_queue
    _state.rpm_limit = cfg.requests_per_minute
    _state.tpm_limit = cfg.estimated_tokens_per_minute
    _state.max_in_flight = cfg.max_in_flight
    _state.shared_enabled = cfg.budget_redis_enabled
    _state.shared_rpm_limit = cfg.budget_radar_rpm
    _state.shared_tpm_limit = cfg.budget_radar_tpm


async def acquire_budget(
    estimated_tokens: int = 600,
    interactive: bool = False,
) -> None:
    """Acquire LLM budget. Block until tokens are available."""
    while True:
        now = time.monotonic()
        wait_secs: float | None = None
        granted = False

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
                granted = True

        if wait_secs is not None:
            await asyncio.sleep(wait_secs + 0.1)
            continue
        if not granted:
            # Window-reset iteration; loop around and acquire on the next pass.
            continue

        # Local slot granted. Reserve against the shared Redis budget so the
        # whole radar fleet (master + workers) stays within its provider share.
        if _state.shared_enabled:
            try:
                shared_cooldown = await _redis_budget.cooldown_until()
                if shared_cooldown > time.monotonic():
                    await asyncio.sleep(min(shared_cooldown - time.monotonic(), 10.0))
                    continue
                if not await _redis_budget.reserve(estimated_tokens):
                    await asyncio.sleep(min(_RedisBudget.seconds_to_minute_rollover(), 5.0))
                    continue
            except Exception:
                pass
        return


def release_budget() -> None:
    """Release in-flight slot."""
    _state.in_flight = max(0, _state.in_flight - 1)


async def handle_429() -> None:
    """Called on a 429 response: pause all lanes (local + shared)."""
    async with _lock:
        _state.total_429s += 1
        cooldown = 30.0 + random.uniform(0, 10.0)
        _state.cooldown_until = max(_state.cooldown_until, time.monotonic() + cooldown)
        logger.warning("LLM governor: 429 received, cooldown", seconds=cooldown)
    if _state.shared_enabled:
        await _redis_budget.set_cooldown(cooldown)


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
        "shared_budget_enabled": _state.shared_enabled,
        "shared_rpm_limit": _state.shared_rpm_limit,
        "shared_tpm_limit": _state.shared_tpm_limit,
    }


def _is_429(err_msg: str) -> bool:
    return (
        "429" in err_msg
        or "rate limit" in err_msg.lower()
        or "too many requests" in err_msg.lower()
    )


def is_aggregator_domain(domain: str) -> bool:
    """Check if a domain is a known aggregator/news/social/VC site."""
    d = domain.lower().rstrip(".")
    if d.startswith("www."):
        d = d[4:]
    # Exact match
    if d in _AGGREGATOR_DOMAINS:
        return True
    # Parent-domain match (e.g. sub.techcrunch.com → techcrunch.com)
    parts = d.split(".")
    if len(parts) >= 2:
        for i in range(1, len(parts)):
            parent = ".".join(parts[i:])
            if parent in _AGGREGATOR_DOMAINS:
                return True
    # Suffix-based checks
    for suffix in (".linkedin.com", ".blogspot.com", ".wordpress.com"):
        if d.endswith(suffix):
            return True
    return False
