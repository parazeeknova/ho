"""Per-question resolution for the per-field form walk.

For each screener field the Node adapter asks Python a single question via the
``answer_question`` RPC. This module decides, in order:

1. Knowledge base (customAnswers, exact learned match, persona embeddings,
   deterministic rules) — via ``ScreenerRAG.kb_answer``.
2. Dropdown questions: map the KB answer onto the real option texts read from
   the page. An unmappable answer is treated as unknown — the option list is
   ground truth, so we ask instead of guessing.
3. Open-ended text questions: LLM generation grounded in persona + resume.
4. Otherwise: Telegram ask — dropdown options included in the same message.
   The answer is persisted into the persona knowledge base (``rag.learn``).
   A Skip/decline leaves the field blank.

Overnight (no human present): any question needing input raises
``DeferredError`` so the run records the question (with its options) and aborts
for the morning digest instead of blocking.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from autofill.rag import ASK_USER
from autofill.telegram import TelegramNotConfiguredError

# Sent RPC error marker that aborts a fill: the job was deferred overnight.
DEFER_MARKER = "AUTOFILL_DEFER"

# Matches the Node adapter's isDeclineOption: user-decline survey choices are
# never valid targets for a definite answer.
_DECLINE_OPTION_RE = re.compile(
    r"(don'?t wish|do not wish|prefer not|choose not|rather not|not wish)", re.I
)


class DeferredError(RuntimeError):
    """Raised overnight when a question needs input the run cannot wait for."""

    def __init__(
        self, question: str, kind: str = "text", options: list[str] | None = None
    ) -> None:
        super().__init__(f"Question deferred: {question}")
        self.question = question
        self.kind = kind
        self.options = list(options or [])


def is_decline_option(text: str) -> bool:
    return bool(_DECLINE_OPTION_RE.search(text or ""))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _candidates(answer: str) -> list[str]:
    """Ordered candidate values: raw answer, first clause, leading Yes/No token."""
    out: list[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        t = value.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)

    raw = (answer or "").strip()
    if not raw:
        return out
    push(raw)
    clause = re.split(r"[.,;]\s*", raw, maxsplit=1)[0].strip()
    if clause and len(clause) < len(raw):
        push(clause)
    tokens = raw.split()
    if tokens and re.fullmatch(r"yes|no", tokens[0], re.I):
        push(tokens[0])
    return out


def match_option(answer: str, options: list[str]) -> str | None:
    """Mirror the Node ``chooseOption``: exact first, then unambiguous
    substring, never against a decline option. Returns None when nothing
    matches confidently — callers must ask rather than guess."""
    eligible = [o for o in (options or []) if not is_decline_option(o)]
    for cand in _candidates(answer):
        nc = _norm(cand)
        if not nc:
            continue
        exact = [o for o in eligible if _norm(o) == nc]
        if len(exact) == 1:
            return exact[0]
        if not exact:
            subs = [o for o in eligible if nc in _norm(o)]
            if len(subs) == 1:
                return subs[0]
    return None


async def resolve_question(
    rag: Any,
    bridge: Any,
    question: str,
    kind: str = "text",
    options: list[str] | None = None,
    overnight: bool = False,
    timeout: float = 300.0,
) -> tuple[str, str]:
    """Resolve one screener question. Returns ``(answer, source)``.

    ``answer`` is the value to fill (blank when the user declined), ``source``
    is one of ``"kb"`` (learned/rules/LLM), ``"telegram"`` (user answered),
    ``"decline"`` (user skipped; leave blank). Raises ``DeferredError``
    (overnight) or ``TelegramNotConfiguredError`` (day, prompting unavailable).
    """
    q = (question or "").strip()
    if not q:
        return ("", "decline")

    if kind == "select":
        # Dropdowns are resolved without the LLM: the option list is ground
        # truth, so the KB answer is only usable when it maps onto a real option.
        kb = await rag.kb_answer(q) if rag is not None else None
        if kb and (kb or "").strip():
            picked = match_option(kb, list(options or []))
            if picked:
                return (picked, "kb")
    else:
        answer = (await rag.answer_questions([q])).get(q, ASK_USER) if rag is not None else ASK_USER
        if answer != ASK_USER and (answer or "").strip():
            return (answer.strip(), "kb")

    if overnight:
        raise DeferredError(q, kind=kind, options=options)

    if bridge is None or not bridge.is_configured:
        raise TelegramNotConfiguredError(
            "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
            f"TELEGRAM_CHAT_ID. Unanswered question: {q}"
        )

    if kind in ("select", "multi") and options:
        picked = await bridge.ask_options(q, list(options), timeout=timeout)
    else:
        picked = await bridge.ask(q, timeout=timeout)

    if picked is None or not (picked or "").strip():
        return ("", "decline")

    picked = picked.strip()
    with contextlib.suppress(Exception):
        await rag.learn(q, picked)  # Learning is best-effort; answer still used.
    return (picked, "telegram")
