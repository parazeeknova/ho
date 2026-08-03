"""Central retry utility with exponential backoff, jitter, and metrics.

Replaces scattered retry loops in connectors, LLM calls, and HTTP requests.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.configuration import RetryConfig, get_config

TRANSIENT_EXCEPTIONS = (
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
)


@dataclass
class RetryMetrics:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_retries: int = 0
    total_wait_time: float = 0.0
    failure_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def success_rate(self) -> float:
        total = self.attempts
        if total == 0:
            return 1.0
        return self.successes / total


def _is_transient(exc: BaseException) -> bool:
    """Return True if the exception represents a transient (retryable) failure."""
    if isinstance(exc, TRANSIENT_EXCEPTIONS):
        return True
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("429", "rate limit", "too many requests", "503", "502", "timeout", "connection")
    )


def _jitter(delay: float) -> float:
    return delay * (0.5 + random.random())


async def retry[T](
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    jitter: bool | None = None,
    metrics: RetryMetrics | None = None,
    should_retry: Callable[[BaseException], bool] = _is_transient,
) -> T:
    """Execute *fn* with exponential backoff and optional jitter.

    Only retries on transient failures (timeouts, connection errors, 429, 5xx).
    Non-transient exceptions are re-raised immediately.
    Cancellation via asyncio.CancelledError is always propagated.
    """
    cfg = config or get_config().retry
    _max_retries = max_retries if max_retries is not None else cfg.max_retries
    _base_delay = base_delay if base_delay is not None else cfg.base_delay
    _max_delay = max_delay if max_delay is not None else cfg.max_delay
    _use_jitter = jitter if jitter is not None else cfg.jitter

    if metrics is not None:
        metrics.attempts += 1

    last_error: BaseException | None = None

    for attempt in range(_max_retries + 1):
        try:
            result = await fn()
            if metrics is not None:
                metrics.successes += 1
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc

            if not should_retry(exc):
                if metrics is not None:
                    metrics.failures += 1
                    metrics.failure_reasons[type(exc).__name__] += 1
                raise

            if attempt >= _max_retries:
                if metrics is not None:
                    metrics.failures += 1
                    metrics.failure_reasons[type(exc).__name__] += 1
                raise

            if metrics is not None:
                metrics.total_retries += 1

            delay = min(_base_delay * (2**attempt), _max_delay)
            if _use_jitter:
                delay = _jitter(delay)

            if metrics is not None:
                metrics.total_wait_time += delay

            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


async def retry_http[T](
    fn: Callable[[], Awaitable[T]],
    **kwargs: Any,
) -> T:
    """Convenience: retry with default HTTP-appropriate settings."""
    return await retry(fn, **kwargs)


class RateLimiter:
    """Async rate limiter with token-bucket semantics."""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._last_call: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            self._last_call = time.time()
