"""Unit tests for per-question resolution, overnight defer, and the digest."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from autofill.rag import ASK_USER, default_visa_option
from autofill.resolve import (
    DEFER_MARKER,
    DeferredError,
    match_option,
    resolve_cover_letter,
    resolve_question,
)
from autofill.telegram import TelegramNotConfiguredError, TelegramQuestionBridge
from autofill.worker import (
    _next_digest_time,
    is_overnight,
)


def _bridge(bot_token: str = "token", chat_id: str = "123") -> TelegramQuestionBridge:
    return TelegramQuestionBridge(bot_token=bot_token, chat_id=chat_id)


def _rag(**stubs) -> MagicMock:
    """A resolve_question-ready mock: every deterministic policy is stubbed to
    None so only the behaviors a test explicitly overrides take effect."""
    rag = MagicMock()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={})
    rag.resolve_visa_policy = MagicMock(return_value=None)
    rag.resolve_authorization_policy = MagicMock(return_value=None)
    rag.resolve_residence_policy = MagicMock(return_value=None)
    rag.resolve_relocation_policy = MagicMock(return_value=None)
    rag.resolve_work_location_policy = MagicMock(return_value=None)
    rag.resolve_affiliation_policy = MagicMock(return_value=None)
    rag.learn = AsyncMock(return_value=True)
    rag.target_country = MagicMock(return_value=None)
    for name, value in stubs.items():
        setattr(rag, name, value)
    return rag


# ── resolve_question ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_text_known_kb_no_prompt() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={"Where are you based?": "Hyderabad"})
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value=None)

    answer, source = await resolve_question(
        rag, bridge, "Where are you based?", kind="text", overnight=False
    )

    assert (answer, source) == ("Hyderabad", "llm")
    bridge.ask.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_select_maps_kb_answer_to_option() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value="No")
    bridge = _bridge()
    bridge.ask_options = AsyncMock()

    answer, source = await resolve_question(
        rag,
        bridge,
        "Do you require visa sponsorship?",
        kind="select",
        options=["Yes", "No"],
        overnight=False,
    )

    assert (answer, source) == ("No", "kb")
    bridge.ask_options.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_select_unmappable_kb_asks_with_options() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value="Flexible")
    rag.answer_questions = AsyncMock(return_value={})  # LLM can't map it either
    rag.resolve_visa_policy = MagicMock(return_value=None)
    rag.resolve_authorization_policy = MagicMock(return_value=None)
    rag.learn = AsyncMock(return_value=True)
    bridge = _bridge()
    bridge.ask_options = AsyncMock(return_value="Remote")

    answer, source = await resolve_question(
        rag,
        bridge,
        "Work model?",
        kind="select",
        options=["Remote", "Onsite"],
        overnight=False,
    )

    assert (answer, source) == ("Remote", "telegram")
    bridge.ask_options.assert_awaited_once_with(
        "Work model?", ["Remote", "Onsite"], timeout=300.0
    )
    rag.learn.assert_awaited_once_with("Work model?", "Remote")


@pytest.mark.asyncio
async def test_resolve_text_unknown_asks_and_learns() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={"Are you a current employee?": ASK_USER})
    rag.learn = AsyncMock(return_value=True)
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value="Yes")

    answer, source = await resolve_question(
        rag, bridge, "Are you a current employee?", kind="text", overnight=False
    )

    assert (answer, source) == ("Yes", "telegram")
    rag.learn.assert_awaited_once_with("Are you a current employee?", "Yes")


@pytest.mark.asyncio
async def test_resolve_decline_leaves_blank() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={"Q1": ASK_USER})
    rag.learn = AsyncMock()
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value=None)
    answer, source = await resolve_question(
        rag, bridge, "Q1", kind="text", overnight=False
    )

    assert (answer, source) == ("", "decline")
    rag.learn.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_dismissed_select_maps_to_decline_option() -> None:
    """Zero-blank: dismissing a dropdown prompt fills the form's own decline
    option (e.g. "I don't wish to answer") instead of leaving the field empty."""
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={})  # LLM can't answer
    rag.resolve_visa_policy = MagicMock(return_value=None)
    rag.resolve_authorization_policy = MagicMock(return_value=None)
    rag.learn = AsyncMock()
    bridge = _bridge()
    bridge.ask_options = AsyncMock(return_value=None)

    answer, source = await resolve_question(
        rag,
        bridge,
        "Veteran Status?",
        kind="select",
        options=["I am not a protected veteran", "I don't wish to answer"],
        overnight=False,
    )

    assert (answer, source) == ("I don't wish to answer", "decline-option")
    rag.learn.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_dismissed_select_without_decline_option_stays_blank() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={})  # LLM can't answer
    rag.resolve_visa_policy = MagicMock(return_value=None)
    rag.resolve_authorization_policy = MagicMock(return_value=None)
    bridge = _bridge()
    bridge.ask_options = AsyncMock(return_value=None)

    answer, source = await resolve_question(
        rag,
        bridge,
        "Work model?",
        kind="select",
        options=["Remote", "Onsite"],
        overnight=False,
    )

    assert (answer, source) == ("", "decline")


@pytest.mark.asyncio
async def test_resolve_overnight_defers_without_prompting() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={"Q1": ASK_USER})
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value=None)

    with pytest.raises(DeferredError) as exc_info:
        await resolve_question(rag, bridge, "Q1", kind="text", overnight=True)

    assert exc_info.value.question == "Q1"
    bridge.ask.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_overnight_defers_select_with_options() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={})  # LLM can't answer
    rag.resolve_visa_policy = MagicMock(return_value=None)
    rag.resolve_authorization_policy = MagicMock(return_value=None)
    bridge = _bridge()

    with pytest.raises(DeferredError) as exc_info:
        await resolve_question(
            rag,
            bridge,
            "Gender?",
            kind="select",
            options=["Male", "Female"],
            overnight=True,
        )

    assert exc_info.value.kind == "select"
    assert exc_info.value.options == ["Male", "Female"]


@pytest.mark.asyncio
async def test_resolve_overnight_skips_optional_question() -> None:
    """A question the form does NOT mark required (no asterisk) is skipped
    overnight instead of deferring the whole job for the digest."""
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={"If other, please specify": ASK_USER})
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value=None)

    answer, source = await resolve_question(
        rag,
        bridge,
        "If other, please specify",
        kind="text",
        overnight=True,
        required=False,
    )

    assert (answer, source) == ("", "decline")
    bridge.ask.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_overnight_required_affiliation_no_negative_deferrals() -> None:
    """A REQUIRED affiliation question whose form offers no negative stance is
    deferred overnight (never silently blanked) and asked by day (never
    guessed by the LLM, which fabricates a company)."""
    rag = _rag()
    rag.resolve_affiliation_policy = MagicMock(return_value="")  # no negative option
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(
        return_value={"Are you related to a current Acme employee?": ASK_USER}
    )
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value="No")
    bridge.ask_options = AsyncMock(return_value="No")
    bridge.ask_dropdown = AsyncMock(return_value="No")

    # Overnight + required -> deferred, not blanked.
    with pytest.raises(DeferredError):
        await resolve_question(
            rag,
            bridge,
            "Are you related to a current Acme employee?",
            kind="select",
            options=["Yes"],
            overnight=True,
            required=True,
        )

    # Day + required -> asked via Telegram, and the LLM never ran.
    answer, source = await resolve_question(
        rag,
        bridge,
        "Are you related to a current Acme employee?",
        kind="select",
        options=["Yes"],
        overnight=False,
        required=True,
    )
    assert (answer, source) == ("No", "telegram")
    rag.answer_questions.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_optional_affiliation_no_negative_blanked() -> None:
    """An OPTIONAL affiliation question with no negative stance is blanked."""
    rag = _rag()
    rag.resolve_affiliation_policy = MagicMock(return_value="")
    rag.kb_answer = AsyncMock(return_value=None)
    bridge = _bridge()

    answer, source = await resolve_question(
        rag,
        bridge,
        "Are you related to a current Acme employee?",
        kind="text",
        overnight=True,
        required=False,
    )
    assert (answer, source) == ("", "decline")


@pytest.mark.asyncio
async def test_resolve_scoped_question_display_qualifies_with_jd_country() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(
        return_value={"Are you authorized to work in the country?": ASK_USER}
    )
    rag.target_country = MagicMock(return_value="india")
    rag.learn = AsyncMock(return_value=True)
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value="No")

    answer, source = await resolve_question(
        rag,
        bridge,
        "Are you authorized to work in the country?",
        kind="text",
        overnight=False,
        job_context={"location": "Bengaluru, India"},
    )

    assert (answer, source) == ("No", "telegram")
    displayed = bridge.ask.await_args.args[0]
    assert displayed == "Are you authorized to work in the country? (India)"
    rag.learn.assert_awaited_once_with(
        "Are you authorized to work in the country?", "No", country="india"
    )


@pytest.mark.asyncio
async def test_resolve_scoped_question_asks_for_country_and_learns_it() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(
        return_value={"Are you authorized to work in the country?": ASK_USER}
    )
    rag.target_country = MagicMock(
        side_effect=lambda text, *a, **k: "india" if "No" in text else None
    )
    rag.learn = AsyncMock(return_value=True)
    bridge = _bridge()
    bridge.ask = AsyncMock(return_value="No (India)")

    answer, source = await resolve_question(
        rag,
        bridge,
        "Are you authorized to work in the country?",
        kind="text",
        overnight=False,
    )

    assert (answer, source) == ("No (India)", "telegram")
    displayed = bridge.ask.await_args.args[0]
    assert "country could not be detected" in displayed
    rag.learn.assert_awaited_once_with(
        "Are you authorized to work in the country?", "No (India)", country="india"
    )


@pytest.mark.asyncio
async def test_resolve_unconfigured_raises() -> None:
    rag = _rag()
    rag.kb_answer = AsyncMock(return_value=None)
    rag.answer_questions = AsyncMock(return_value={"Q1": ASK_USER})
    bridge = TelegramQuestionBridge(bot_token="", chat_id="")

    with pytest.raises(TelegramNotConfiguredError):
        await resolve_question(rag, bridge, "Q1", kind="text", overnight=False)

# ── match_option ───────────────────────────────────────────────────


def test_match_option_exact() -> None:
    assert match_option("No", ["Yes", "No"]) == "No"


def test_match_option_disability_decline_excluded() -> None:
    """The "I do not want to answer" decline option must be excluded so the
    "no" substring (also inside "not") is unambiguous against the negation."""
    options = [
        "Yes, I have a disability, or have had one in the past",
        "No, I do not have a disability and have not had one in the past",
        "I do not want to answer",
    ]
    assert match_option("No", options) == options[1]
    assert match_option("Yes", options) == options[0]


def test_match_option_forgives_small_typo() -> None:
    options = ["Bachelor's Degree", "Master's Degree", "PhD"]
    assert match_option("bachlors", options) == "Bachelor's Degree"
    # A typo landing near two options must not pick either.
    assert match_option("degre", ["Bachelor's Degree", "Master's Degree"]) is None


# ── resolve_cover_letter ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_cover_letter_uses_generate_cover_letter() -> None:
    rag = _rag()
    rag.generate_cover_letter = AsyncMock(return_value="A structured cover letter.")

    text, source = await resolve_cover_letter(rag, job_context={"title": "Role"})

    assert (text, source) == ("A structured cover letter.", "llm")
    rag.generate_cover_letter.assert_awaited_once_with({"title": "Role"})


@pytest.mark.asyncio
async def test_resolve_cover_letter_falls_back_when_ungrounded() -> None:
    rag = _rag()
    rag.generate_cover_letter = AsyncMock(return_value="   ")

    text, source = await resolve_cover_letter(rag, job_context={})

    assert (text, source) == ("", "decline")


# ── visa-sponsorship deterministic policy ──────────────────────────


def test_default_visa_option_prefers_h1b_then_yes() -> None:
    assert default_visa_option(["No", "EB-1", "EB-2", "TN-1", "H1-B", "H2-B"]) == "H1-B"
    assert default_visa_option(["No", "Yes"]) == "Yes"
    assert default_visa_option(["No", "H-1B"]) == "H-1B"
    assert default_visa_option(["No"]) == "No"
    assert default_visa_option([]) is None


def test_resolve_visa_policy_unknown_country_defaults_to_h1b() -> None:
    from autofill.rag import ScreenerRAG

    rag = ScreenerRAG(
        exact_answers={},
        profile=MagicMock(location="Bhopal, India", customAnswers={}),
    )
    opts = ["No", "EB-1", "EB-2", "TN-1", "H1-B", "H2-B"]
    q = "Will you require visa sponsorship either now or in the near future?"
    # Job location has no country ("Remote / Boston / ..."): default to H1-B.
    assert (
        rag.resolve_visa_policy(q, opts, {"location": "Remote / Boston", "description": ""})
        == "H1-B"
    )


def test_resolve_visa_policy_job_country_differs_from_home_is_yes() -> None:
    from autofill.rag import ScreenerRAG

    rag = ScreenerRAG(
        exact_answers={},
        profile=MagicMock(location="Bhopal, India", customAnswers={}),
    )
    opts = ["No", "Yes"]
    q = "Will you require visa sponsorship either now or in the near future?"
    # Job country is US, home is India -> default to Yes.
    assert (
        rag.resolve_visa_policy(
            q, opts, {"location": "San Francisco, USA", "description": ""}
        )
        == "Yes"
    )


def test_resolve_visa_policy_job_country_equals_home_is_no() -> None:
    from autofill.rag import ScreenerRAG

    rag = ScreenerRAG(
        exact_answers={},
        profile=MagicMock(location="Bhopal, India", customAnswers={}),
    )
    opts = ["No", "Yes"]
    q = "Will you require visa sponsorship either now or in the near future?"
    # Job is in India (home) -> no sponsorship needed.
    assert (
        rag.resolve_visa_policy(q, opts, {"location": "Bengaluru, India", "description": ""})
        == "No"
    )


def test_resolve_visa_policy_non_visa_question_returns_none() -> None:
    from autofill.rag import ScreenerRAG

    rag = ScreenerRAG(
        exact_answers={},
        profile=MagicMock(location="Bhopal, India", customAnswers={}),
    )
    assert rag.resolve_visa_policy("Work model?", ["Remote", "Onsite"], {}) is None


def test_match_option_unambiguous_substring() -> None:
    assert match_option("Remote", ["Remote-first", "In-person"]) == "Remote-first"


def test_match_option_ambiguous_substring_returns_none() -> None:
    assert match_option("Yes", ["Yes, please", "Yes, but later"]) is None


def test_match_option_decline_never_matches() -> None:
    assert match_option("Prefer not to say", ["Yes", "Prefer not to say"]) is None


def test_match_option_first_clause_candidate() -> None:
    assert match_option("Yes, I am eligible", ["Yes", "No"]) == "Yes"


def test_match_option_no_match() -> None:
    assert match_option("Maybe", ["Yes", "No"]) is None


# ── mode helpers ───────────────────────────────────────────────────


def test_is_overnight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERNIGHT_LOOP", "true")
    assert is_overnight() is True
    monkeypatch.setenv("OVERNIGHT_LOOP", "TRUE")
    assert is_overnight() is True
    monkeypatch.setenv("OVERNIGHT_LOOP", "false")
    assert is_overnight() is False
    monkeypatch.delenv("OVERNIGHT_LOOP")
    assert is_overnight() is False


def test_next_digest_time() -> None:
    import datetime as dt

    target = _next_digest_time("08:00")
    assert target.hour == 8 and target.minute == 0
    assert target > dt.datetime.now()

    fallback = _next_digest_time("not-a-time")
    assert fallback.hour == 8


def test_defer_marker_constant() -> None:
    assert DEFER_MARKER == "AUTOFILL_DEFER"
