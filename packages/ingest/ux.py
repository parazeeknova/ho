"""Shared rich UI for the memory-setup CLI scripts.

Deliberately "AI-slop" aesthetic: cyan/magenta accents, rounded panels,
section headers, status chips, and summary tables. No emojis - only
typographic symbols.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

CYAN = "cyan"
MAGENTA = "bold magenta"
GREEN = "bold green"
YELLOW = "bold yellow"
RED = "bold red"
DIM = "dim"


def banner(title: str, subtitle: str = "") -> None:
    """Big rounded header panel, the signature 'AI slop' look."""
    text = Text()
    text.append(title, style=MAGENTA)
    if subtitle:
        text.append(f"\n{subtitle}", style=f"{CYAN} italic")
    console.print(Panel(text, border_style=CYAN, box=box.ROUNDED, padding=(1, 2)))


def section(index: int, total: int, label: str, detail: str = "") -> None:
    """Section header like:  [2/4]  EMBEDDING SERVER  ·  detail"""
    head = Text(f"[{index}/{total}]", style=CYAN)
    head.append(f"  {label.upper()}", style="bold white")
    if detail:
        head.append(f"  ·  {detail}", style=DIM)
    console.print()
    console.print(head)


def divider(text: str = "") -> None:
    console.print(Rule(text, style=DIM) if text else Rule(style=DIM))


def chip(kind: str, msg: str) -> None:
    """Indented status line with a colored tag:  [OK] message"""
    colors = {"ok": GREEN, "warn": YELLOW, "err": RED, "info": CYAN}
    tags = {"ok": "OK", "warn": "WARN", "err": "FAIL", "info": "INFO"}
    style = colors.get(kind, CYAN)
    text = Text(f"  {tags.get(kind, 'INFO')} ", style=style)
    text.append(f"  {msg}")
    console.print(text)


def bullet(msg: str, style: str = "dim") -> None:
    console.print(Text(f"  ▸ {msg}", style=style))


def summary_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(
        title=title,
        title_style=MAGENTA,
        border_style=CYAN,
        box=box.ROUNDED,
        padding=(0, 2),
        show_header=False,
    )
    table.add_column(style="bold white", justify="left")
    table.add_column(style=CYAN, justify="left")
    for key, value in rows:
        table.add_row(key, value)
    console.print()
    console.print(table)


def next_steps(steps: list[str]) -> None:
    body = Text()
    for i, step in enumerate(steps):
        if i:
            body.append("\n")
        body.append("  ▸ ", style=CYAN)
        body.append(step)
    console.print()
    console.print(
        Panel(
            body,
            title="NEXT STEPS",
            title_align="left",
            border_style=MAGENTA,
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
