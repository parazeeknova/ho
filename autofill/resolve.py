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

from autofill.rag import ASK_USER, is_scoped_question, qualify_question
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
    job_context: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Resolve one screener question. Returns ``(answer, source)``.

    ``answer`` is the value to fill (blank when the user declined), ``source``
    is one of ``"kb"`` (learned/rules/LLM), ``"telegram"`` (user answered),
    ``"decline"`` (user skipped; leave blank). Raises ``DeferredError``
    (overnight) or ``TelegramNotConfiguredError`` (day, prompting unavailable).

    ``job_context`` carries the extracted job description (title, company,
    location, description) so open-ended answers personalise to the role and
    country-scoped questions (work authorization, visa) resolve against the
    job's country instead of leaking a global answer.
    """
    q = (question or "").strip()
    if not q:
        return ("", "decline")

    if kind == "select":
        # Dropdowns are resolved without the LLM: the option list is ground
        # truth, so the KB answer is only usable when it maps onto a real option.
        kb = await rag.kb_answer(q, job_context=job_context) if rag is not None else None
        if kb and (kb or "").strip():
            picked = match_option(kb, list(options or []))
            if picked:
                return (picked, "kb")
    else:
        answer = (
            (await rag.answer_questions([q], job_context=job_context)).get(q, ASK_USER)
            if rag is not None
            else ASK_USER
        )
        if answer != ASK_USER and (answer or "").strip():
            return (answer.strip(), "kb")

    if overnight:
        raise DeferredError(q, kind=kind, options=options)

    if bridge is None or not bridge.is_configured:
        raise TelegramNotConfiguredError(
            "Telegram prompting unavailable: set TELEGRAM_BOT_TOKEN and "
            f"TELEGRAM_CHAT_ID. Unanswered question: {q}"
        )

    # Country-scoped questions are asked with the country named so the answer
    # (and the learned entry) is qualified, never global. If the country
    # cannot be detected from the question or the JD, the user is told so and
    # asked to name it — a scoped answer is never stored without a country.
    display_q = q
    scope_country = None
    if is_scoped_question(q):
        scope_country = rag.target_country(q, job_context) if rag is not None else None
        if scope_country:
            display_q = qualify_question(q, scope_country)
        else:
            display_q = (
                f"{q}\n\n(Job country could not be detected from the posting. "
                "Please include the country in your reply, e.g. \"No (India)\".)"
            )

    if kind in ("select", "multi") and options:
        picked = await bridge.ask_options(display_q, list(options), timeout=timeout)
    else:
        picked = await bridge.ask(display_q, timeout=timeout)

    if picked is None or not (picked or "").strip():
        return ("", "decline")

    picked = picked.strip()
    with contextlib.suppress(Exception):
        # Learning is best-effort; answer still used. The scope country is only
        # passed when derived from the job description — when the question
        # itself names a country, learn() re-derives it. When neither did, the
        # message asked the user to name the country in their reply.
        learn_kwargs: dict[str, Any] = {}
        if is_scoped_question(q):
            if not scope_country and rag is not None:
                scope_country = rag.target_country(picked)
            if scope_country:
                learn_kwargs["country"] = scope_country
        await rag.learn(q, picked, **learn_kwargs)
    return (picked, "telegram")


async def resolve_cover_letter(
    rag: Any, job_context: dict[str, Any] | None = None
) -> tuple[str, str]:
    """Generate a job-personalized cover letter from persona + job context.

    Returns ``(text, "llm")`` when generated, ``("", "decline")`` when the LLM
    has nothing to ground it on. Never prompts the user — a blank cover letter
    is a valid outcome.
    """
    if rag is None:
        return ("", "decline")
    answer = (
        await rag.answer_questions(
            ["Write a personalized cover letter for this application."],
            job_context=job_context,
        )
    ).get("Write a personalized cover letter for this application.", ASK_USER)
    text = (answer or "").strip()
    if not text or text == ASK_USER:
        return ("", "decline")
    return (text, "llm")
