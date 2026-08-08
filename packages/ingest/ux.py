"""Shared plain-text UI for the memory-setup CLI scripts.

Clean ``[ho]``-prefixed lines, matching the aesthetic of ``bun run run``.
No rich panels, tables, rules, arrows, or ANSI decoration — just the same
plain log lines the pipeline uses.
"""

from __future__ import annotations

PREFIX = "[ho] "


class _Status:
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __enter__(self) -> _Status:
        print(f"{PREFIX}{self.msg}...", flush=True)
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def update(self, msg: str = "") -> None:
        if msg:
            print(f"{PREFIX}{msg}", flush=True)


class _Console:
    """Plain-text stand-in for rich.Console: prints [ho]-prefixed lines."""

    def status(self, msg: str = "", spinner: str = "") -> _Status:
        return _Status(msg)


console = _Console()


def banner(title: str, subtitle: str = "") -> None:
    if subtitle:
        print(f"{PREFIX}{title} — {subtitle}", flush=True)
    else:
        print(f"{PREFIX}{title}", flush=True)


def section(index: int, total: int, label: str, detail: str = "") -> None:
    head = f"[{index}/{total}] {label.upper()}"
    if detail:
        head += f" · {detail}"
    print(f"{PREFIX}{head}", flush=True)


def divider(text: str = "") -> None:
    if text:
        print(f"{PREFIX}{text}", flush=True)


def chip(kind: str, msg: str) -> None:
    """Plain status line:  [ho] OK message / WARN / FAIL / INFO"""
    tags = {"ok": "OK", "warn": "WARN", "err": "FAIL", "info": "INFO"}
    print(f"{PREFIX}{tags.get(kind, 'INFO')} {msg}", flush=True)


def bullet(msg: str, style: str = "dim") -> None:
    print(f"{PREFIX}{msg}", flush=True)


def summary_table(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"{PREFIX}{title}", flush=True)
    for key, value in rows:
        print(f"{PREFIX}  {key}: {value}", flush=True)


def next_steps(steps: list[str]) -> None:
    for step in steps:
        print(f"{PREFIX}  {step}", flush=True)
