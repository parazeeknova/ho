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

from autofill.discord import (
    _SKIP_SENTINEL,
    DiscordNotConfiguredError,
    edit_distance,
)
from autofill.rag import ASK_USER, is_scoped_question, qualify_question

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

    def __init__(self, question: str, kind: str = "text", options: list[str] | None = None) -> None:
        super().__init__(f"Question deferred: {question}")
        self.question = question
        self.kind = kind
        self.options = list(options or [])


def is_decline_option(text: str) -> bool:
    return bool(_DECLINE_OPTION_RE.search(text or ""))


async def _fallback_open_ended(rag: Any, q: str, job_context: dict[str, Any] | None) -> str:
    """Retry an open-ended text question the batch LLM couldn't ground.

    The first LLM pass returns __ASK_USER__ when it isn't confident; ATS
    boards reject submissions with blank required free-text fields. This
    retries with a directive to answer from the resume, and returns a
    concise grounded fallback so the field is never empty. Returns "" when
    even the fallback can't produce anything (caller then defers).
    """
    try:
        jt = (job_context or {}).get("title", "the role")
        jc = (job_context or {}).get("company", "the company")
        prompt = (
            "Answer this application question briefly (2-3 sentences) based ONLY on the "
            "candidate's resume and background. Do NOT refuse; a short honest answer "
            "is required.\n"
            f"Question: {q}\n"
            f"Job: {jt} at {jc}\n"
            "Write in the candidate's first-person voice. Never mention that you are an AI."
        )
        result = await rag.cm.chat(prompt, max_tokens=200, interactive=True)
        text = (result or "").strip()
        # Reject obvious non-answers / refusals.
        if not text or len(text) < 15 or "i'm an ai" in text.lower():
            return ""
        return text[:500]
    except Exception:
        return ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


# Gender-identity synonym map: persona answers ("Male", "Female", "Non-binary")
# vs the option labels Greenhouse/Lever/Workday etc. actually use ("Man",
# "Woman", "Non-binary"). Without this, a gender question whose options are
# "Man/Woman/Non-binary" gets declined/blanked even though the persona has an
# answer, and a required DEI question then blocks submission.
_GENDER_SYNONYMS: dict[str, str] = {
    "male": "man",
    "man": "male",
    "female": "woman",
    "woman": "female",
    "nonbinary": "non-binary",
    "non-binary": "nonbinary",
    "non binary": "non-binary",
}


def _gender_alias(answer: str) -> list[str]:
    """Extra candidate spellings for a gender/DEI answer so option matching
    can bridge persona-vs-form label differences."""
    a = (answer or "").strip().lower()
    out: list[str] = []
    if a in _GENDER_SYNONYMS:
        out.append(_GENDER_SYNONYMS[a])
    # "I prefer to self-describe" style answers
    if "self-describe" in a or "self describe" in a:
        out.append("self-describe")
    return out


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
    candidates = list(_candidates(answer)) + _gender_alias(answer)
    for cand in candidates:
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
    required: bool = True,
) -> tuple[str, str]:
    """Resolve one screener question. Returns ``(answer, source)``.

    ``answer`` is the value to fill, ``source`` is one of ``"kb"``
    (deterministic persona/learned/rules), ``"llm"`` (grounded LLM with
    resume/persona/JD), ``"telegram"`` (user answered), ``"decline-option"``
    (a dropped prompt was mapped to the form's own decline option — still a
    committed value, never blank) or ``"decline"`` (user skipped and the form
    offers no decline option). Raises ``DeferredError`` (overnight),
    ``DiscordNotConfiguredError`` (day, prompting unavailable) or
    ``DiscordSendError`` (day, the question could not be delivered).

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

    # Work-authorization deterministic policy, mirroring visa: a question like
    # "Are you legally authorized to work in Germany?" is decided from the
    # job/home country (job != home -> not authorized -> "No"/sponsorship).
    # When the policy cannot decide it returns None so the LLM tier below can
    # still attempt the question instead of deferring an answerable one.
    if kind in ("select", "multi") and options and rag is not None:
        auth_pick = rag.resolve_authorization_policy(q, list(options), job_context)
        if auth_pick is not None:
            return (auth_pick, "kb")

    # Current-residence, relocation-willingness, and office-commute geography
    # policies: a question like "are you based in Europe?", "willing to
    # relocate to Bangkok?", or "able to work from our SF office?" is a
    # deterministic fact (the candidate is based in India, relocates to
    # first-world but not third-world countries), never a guess.
    if rag is not None:
        for policy in (
            rag.resolve_residence_policy,
            rag.resolve_relocation_policy,
            rag.resolve_work_location_policy,
        ):
            geo_pick = policy(q, list(options or []), job_context)
            if geo_pick is not None:
                if kind in ("select", "multi") and options:
                    picked = match_option(geo_pick, list(options))
                    if picked:
                        return (picked, "kb")
                    continue
                return (geo_pick.strip(), "kb")

    # Affiliation / employment / relationship questions: the candidate has no
    # such affiliations, so the answer is the form's negative option — never a
    # company the LLM picks from the options (it has fabricated prior
    # employment). When the form offers no negative stance, do NOT guess: a
    # required affiliation question falls through to defer/ask below, and only
    # an optional one is left blank.
    if rag is not None:
        aff_pick = rag.resolve_affiliation_policy(q, list(options or []), job_context)
        if aff_pick is not None:
            if kind in ("select", "multi") and options and aff_pick:
                picked = match_option(aff_pick, list(options))
                if picked:
                    return (picked, "kb")
                # The negative stance is a decline option ("I don't wish to
                # answer") that match_option excludes: fill it directly so the
                # field is never silently blank (mirrors the zero-blank path).
                if is_decline_option(aff_pick):
                    return (aff_pick, "decline-option")
                # A non-option negative stance: never fall through to the LLM
                # (it fabricates a company).
                return ("", "decline")
            if aff_pick:
                return (aff_pick.strip(), "kb")
            # No negative stance offered. An optional field is blanked; a
            # required one must be deferred (overnight) or asked (day) — the
            # tiers below already handle that, and answer_questions returns
            # __ASK_USER__ for an unanswered affiliation question (never the
            # LLM, which would fabricate a company).
            if not required:
                return ("", "decline")
            fall_through_affiliation = True
        else:
            fall_through_affiliation = False
    else:
        fall_through_affiliation = False

    # Tier 3: grounded LLM (persona + resume + JD, with options for selects).
    # Select answers are validated against the real options inside
    # answer_questions — a non-option is never filled. Only what survives to
    # __ASK_USER__ reaches Telegram below. Unresolved AFFILIATION questions
    # never reach the LLM — answer_questions guards them (returns __ASK_USER__)
    # and the LLM has fabricated prior employment.
    if not fall_through_affiliation:
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
        # A free-text open-ended question (about the candidate's work) that the
        # LLM couldn't confidently ground must NOT be left blank — ATS boards
        # reject the submission. Retry once with a "must answer from resume"
        # prompt; if that still fails, fill a concise fallback so the field is
        # never empty. Never fabricate identity/contact facts — only generative
        # experience questions get this fallback.
        if kind == "text" and rag is not None:
            fallback = await _fallback_open_ended(rag, q, job_context)
            if fallback:
                return (fallback, "llm")

    if overnight:
        # Overnight (no human present): an unresolved question either defers
        # the job for the morning digest or — when the form does NOT mark it
        # required (no asterisk) — is skipped. Leaving a blank optional field
        # is strictly better than stalling the run on a question the form
        # itself does not require (e.g. "If other, please specify").
        if not required:
            return ("", "decline")
        raise DeferredError(q, kind=kind, options=options)

    if bridge is None or not bridge.is_configured:
        raise DiscordNotConfiguredError(
            "Discord prompting unavailable: set DISCORD_BOT_TOKEN and "
            f"DISCORD_CHANNEL_ID. Unanswered question: {q}"
        )

    # Country-scoped questions are asked with the country named so the answer
    # (and the learned entry) is qualified, never global. If the country
    # cannot be detected from the question or the JD, the user is told so and
    # asked to name it — a scoped answer is never stored without a country.
    from autofill.worker import AutofillWorker

    display_q = AutofillWorker._clean_question(q)
    scope_country = None
    if is_scoped_question(q):
        scope_country = rag.target_country(q, job_context) if rag is not None else None
        if scope_country:
            display_q = qualify_question(display_q, scope_country)
        else:
            display_q = (
                f"{display_q}\n\n(Job country could not be detected from the posting. "
                'Please include the country in your reply, e.g. "No (India)".)'
            )

    if kind in ("select", "multi") and options:
        picked = await bridge.ask_options(display_q, list(options), timeout=timeout)
    elif kind in ("select", "multi"):
        # Dropdown whose options could not be read: still ask as a dropdown so
        # the user replies with an option label, never a free-form guess.
        picked = await bridge.ask_dropdown(display_q, timeout=timeout)
    else:
        picked = await bridge.ask(display_q, timeout=timeout)

    # A TIMEOUT (no answer within the ask window) is NOT a dismissal — it
    # means the user wasn't there. Defer the job so it's re-asked later
    # instead of silently blanking a field the user never saw.
    if picked is _SKIP_SENTINEL:
        # Explicit user skip: fill the form's own decline option if one
        # exists so the field is never silently empty; otherwise blank it.
        if kind in ("select", "multi") and options:
            decline = next((o for o in options if is_decline_option(o)), None)
            if decline:
                return (decline, "decline-option")
        return ("", "decline")
    if picked is None or not (picked or "").strip():
        raise DeferredError(q, kind=kind, options=options)

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
    return (picked, "discord")


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
