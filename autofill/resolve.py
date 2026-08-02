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
   A Skip/decline is mapped to the form's own decline option when offered;
   only when no such option exists does a dismissal leave the field blank.

Overnight (no human present): any question needing input raises
``DeferredError`` so the run records the question (with its options) and aborts
for the morning digest instead of blocking.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from autofill.rag import ASK_USER, is_scoped_question, qualify_question
from autofill.telegram import (
    TelegramNotConfiguredError,
    edit_distance,
)

# Sent RPC error marker that aborts a fill: the job was deferred overnight.
DEFER_MARKER = "AUTOFILL_DEFER"

# Matches the Node adapter's isDeclineOption: user-decline survey choices are
# never valid targets for a definite answer. Covers "I do not want to answer"
# (used by Greenhouse disability surveys) as well as the don't-wish variants.
_DECLINE_OPTION_RE = re.compile(
    r"(don'?t wish|do not wish|prefer not|choose not|rather not|not wish|"
    r"do not want to answer|not want to answer)",
    re.I,
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
    substring, never against a decline option, then a conservative edit-distance
    fallback that forgives a small typo (only when exactly one option is within
    tolerance). Returns None when nothing matches confidently — callers must
    ask rather than guess."""
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
        # Edit-distance: a small typo against a single token of exactly one
        # option ("bachlors" -> "Bachelor's Degree", but "degre" matches both
        # degree tokens so it stays ambiguous).
        threshold = max(1, min(2, len(nc) // 4))
        near = []
        for o in eligible:
            tokens = [t for t in _norm(o).split() if t]
            if any(t and edit_distance(nc, t) <= threshold for t in tokens):
                near.append(o)
        if len(near) == 1:
            return near[0]
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

    ``answer`` is the value to fill, ``source`` is one of ``"kb"``
    (deterministic persona/learned/rules), ``"llm"`` (grounded LLM with
    resume/persona/JD), ``"telegram"`` (user answered), ``"decline-option"``
    (a dropped prompt was mapped to the form's own decline option — still a
    committed value, never blank) or ``"decline"`` (user skipped and the form
    offers no decline option). Raises ``DeferredError`` (overnight),
    ``TelegramNotConfiguredError`` (day, prompting unavailable) or
    ``TelegramSendError`` (day, the question could not be delivered).

    ``job_context`` carries the extracted job description (title, company,
    location, description) so open-ended answers personalise to the role and
    country-scoped questions (work authorization, visa) resolve against the
    job's country instead of leaking a global answer.
    """
    q = (question or "").strip()
    if not q:
        return ("", "decline")

    # Tier 2: deterministic KB (persona / learned / rules) — no LLM.
    kb = await rag.kb_answer(q, job_context=job_context) if rag is not None else None
    if isinstance(kb, str) and kb.strip():
        if kind in ("select", "multi") and options:
            picked = match_option(kb, list(options))
            if picked:
                return (picked, "kb")
            # KB value doesn't map to a real option: fall through to the LLM.
        else:
            return (kb.strip(), "kb")

    # Visa-sponsorship deterministic policy: when the persona has no
    # country-scoped answer, decide from the job/home country —
    #   unknown job country        -> Yes / H1-B
    #   job country != home        -> Yes / H1-B
    #   job country == home        -> No
    # This replaces the Telegram prompt on visa questions whenever the policy
    # can decide. (The LLM never answers scoped visa questions.)
    if kind in ("select", "multi") and options and rag is not None:
        visa_pick = rag.resolve_visa_policy(q, list(options), job_context)
        if visa_pick is not None:
            return (visa_pick, "kb")

    # Tier 3: grounded LLM (persona + resume + JD, with options for selects).
    # Select answers are validated against the real options inside
    # answer_questions — a non-option is never filled. Only what survives to
    # __ASK_USER__ reaches Telegram below.
    spec: dict[str, Any] = {
        "question": q,
        "kind": kind,
        "options": list(options or []),
    }
    answer = (
        (await rag.answer_questions([spec], job_context=job_context)).get(q, ASK_USER)
        if rag is not None
        else ASK_USER
    )
    if answer != ASK_USER and (answer or "").strip():
        return (answer.strip(), "llm")

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
    elif kind in ("select", "multi"):
        # Dropdown whose options could not be read: still ask as a dropdown so
        # the user replies with an option label, never a free-form guess.
        picked = await bridge.ask_dropdown(display_q, timeout=timeout)
    else:
        picked = await bridge.ask(display_q, timeout=timeout)

    if picked is None or not (picked or "").strip():
        # Zero-blank policy: when the user dismisses a dropdown prompt, fill the
        # form's own decline option (e.g. "I don't wish to answer") if one
        # exists so the field is never silently left empty. Only when no decline
        # option is offered does the walker treat it as an explicit user skip.
        if kind in ("select", "multi") and options:
            decline = next((o for o in options if is_decline_option(o)), None)
            if decline:
                return (decline, "decline-option")
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
    """Generate a structured, fact-grounded cover letter from persona + resume
    + job context via ``ScreenerRAG.generate_cover_letter``.

    Returns ``(text, "llm")`` when generated, ``("", "decline")`` when the LLM
    has nothing to ground it on. Never prompts the user — a blank cover letter
    is a valid outcome.
    """
    if rag is None:
        return ("", "decline")
    gen = getattr(rag, "generate_cover_letter", None)
    if gen is not None:
        text = await gen(job_context)
    else:
        # Fallback for mocks/legacy: old single-question path.
        answer = (
            await rag.answer_questions(
                ["Write a personalized cover letter for this application."],
                job_context=job_context,
            )
        ).get("Write a personalized cover letter for this application.", ASK_USER)
        text = (answer or "").strip()
        if text == ASK_USER:
            text = ""
    if not text or not text.strip():
        return ("", "decline")
    return (text.strip(), "llm")
