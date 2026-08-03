"""Structured logging for the ho pipeline.

Replaces all print() calls with machine-readable structured logs.
Supports contextual fields (connector, source, entity, worker, event_id,
latency, retry_count, exception).

Levels: DEBUG < INFO < WARNING < ERROR
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    timestamp: str = ""
    level: str = "INFO"
    message: str = ""
    logger: str = "ho"
    # Optional contextual fields
    connector: str = ""
    source: str = ""
    entity: str = ""
    worker: str = ""
    event_id: str = ""
    latency: float | None = None
    retry_count: int | None = None
    exception: str = ""
    # Arbitrary extras
    extra: dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    def __init__(self, name: str = "ho", min_level: LogLevel = LogLevel.INFO) -> None:
        self.name = name
        self.min_level = min_level
        self._context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> StructuredLogger:
        """Return a child logger with extra fields baked in."""
        child = StructuredLogger(self.name, self.min_level)
        child._context = {**self._context, **kwargs}
        return child

    def _emit(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        if level.value < self.min_level.value:
            return

        # Separate known LogEntry fields from arbitrary extras
        known = {
            "timestamp",
            "level",
            "message",
            "logger",
            "connector",
            "source",
            "entity",
            "worker",
            "event_id",
            "latency",
            "retry_count",
            "exception",
            "extra",
        }
        merged = {**self._context, **kwargs}
        entry_kwargs = {k: v for k, v in merged.items() if k in known}
        extra_kwargs = {k: v for k, v in merged.items() if k not in known}
        if extra_kwargs:
            entry_kwargs.setdefault("extra", {}).update(extra_kwargs)

        entry = LogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            level=level.value,
            message=message,
            logger=self.name,
            **entry_kwargs,
        )

        # Strip empty fields for cleaner output
        payload = {k: v for k, v in asdict(entry).items() if v not in ("", None, {}, [])}

        # JSON to stderr so stdout is free for data piping
        print(json.dumps(payload, default=str), file=sys.stderr)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._emit(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._emit(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._emit(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._emit(LogLevel.ERROR, message, **kwargs)

    def exception(self, message: str, exc: BaseException | None = None, **kwargs: Any) -> None:
        tb = ""
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self._emit(LogLevel.ERROR, message, exception=tb, **kwargs)


# Module-level convenience
_root = StructuredLogger(name="ho")


def get_logger(name: str) -> StructuredLogger:
    min_level = LogLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return StructuredLogger(name=name, min_level=min_level)
