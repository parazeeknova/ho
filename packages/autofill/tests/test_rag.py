"""Unit tests for Phase 3 ScreenerRAG integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autofill.rag import ASK_USER, ScreenerRAG, _pick_authorization_answer


def _cfg(min_salary: str = "80K INR/month"):
    cfg = MagicMock()
    candidate = MagicMock()
    candidate.persona = ""
    candidate.min_salary = min_salary
    cfg.candidate = candidate
    return cfg


@pytest.mark.asyncio
async def test_deterministic_question_answering():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(
        return_value='{"Why Twilio?": "I admire Twilio\'s developer-first API culture."}'
    )

    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
    questions = [
        "Do you require visa sponsorship?",
        "Are you legally authorized to work in the United States?",
        "What is your expected salary?",
        "Why Twilio?",
    ]

    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(questions)

    # Scoped questions (visa/authorization) without a same-country answer are
    # never guessed or LLM-answered — they become a user prompt.
    assert answers["Do you require visa sponsorship?"] == ASK_USER
    assert answers["Are you legally authorized to work in the United States?"] == ASK_USER
    assert answers["What is your expected salary?"] == "80K INR/month"

    # Verify LLM call was only made for the open-ended question ("Why Twilio?")
    assert mock_cm.chat.called
    assert answers["Why Twilio?"] == "I admire Twilio's developer-first API culture."


@pytest.mark.asyncio
async def test_expected_comp_answers_in_question_currency():
    """A question naming a foreign currency gets the per-currency target, not
    the INR figure."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})

    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert await rag.kb_answer("What is your expected salary in USD?") == "$8,300"
        assert await rag.kb_answer("What is your expected annual salary in EUR?") == "€65,000"
        # Unspecified granularity defaults to monthly (matches the INR persona).
        assert await rag.kb_answer("What are your salary expectations in GBP?") == "£4,600"
        # No currency named and no job country: unchanged INR behavior.
        assert await rag.kb_answer("What is your expected salary?") == "80K INR/month"


@pytest.mark.asyncio
async def test_expected_comp_uses_job_country_currency():
    """A salary question without an explicit currency follows the job's
    country currency (US job -> USD figure, India job -> INR figure)."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})

    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert (
            await rag.kb_answer(
                "What is your expected salary?",
                {"location": "San Francisco, CA", "description": ""},
            )
            == "$8,300"
        )
        assert (
            await rag.kb_answer(
                "What is your expected salary?",
                {"location": "Bengaluru, India", "description": ""},
            )
            == "80K INR/month"
        )


@pytest.mark.asyncio
async def test_expected_comp_currency_overrides_inr_custom_answer():
    """Even when the persona's INR answer would normally match, an explicit
    foreign currency in the question wins — the bug this fix targets."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(
        context_manager=mock_cm,
        exact_answers={
            _normalise_question_import("what is your expected compensation?"): "80000 INR/ month"
        },
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert await rag.kb_answer("What is your expected compensation in USD?") == "$8,300"
        # The exact INR question still answers INR.
        assert await rag.kb_answer("What is your expected compensation?") == "80000 INR/ month"


def _normalise_question_import(s):
    from autofill.rag import _normalise_question

    return _normalise_question(s)


@pytest.mark.asyncio
async def test_proficiency_gate_uses_resume_skill_whitelist():
    """Technology-proficiency questions resolve from the resume skills only:
    a real resume skill -> Yes, anything else -> No (never guessed by the
    LLM)."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})

    with patch("autofill.rag.get_config", return_value=_cfg()):
        # Resume skills -> Yes.
        assert await rag.kb_answer("Do you have experience with React?") == "Yes"
        assert await rag.kb_answer("Are you comfortable with TypeScript?") == "Yes"
        assert (
            await rag.kb_answer("Do you have strong proficiency in both Python and Rust?") == "Yes"
        )
        # "one of (Go, React, TypeScript)" -> Yes because React/TS are on the resume.
        assert (
            await rag.kb_answer(
                "Do you have proficiency in one of our primary languages (Go, React, TypeScript)?"
            )
            == "Yes"
        )
        # Not on the resume -> No, even when a persona answer claims Yes.
        assert await rag.kb_answer("Do you have experience with Kubernetes?") == "No"
        assert await rag.kb_answer("Do you have experience with Docker?") == "No"
        assert await rag.kb_answer("Do you have proficiency in Go?") == "No"
        # LLM never sees a proficiency question.
        assert mock_cm.chat.called is False


@pytest.mark.asyncio
async def test_proficiency_gate_ignores_non_skill_phrasing():
    """Open-ended 'experience building X' questions are NOT proficiency gates
    and are left for the LLM."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(
        return_value='{"Do you have experience building real-time systems?": "Yes, I built Larity"}'
    )
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})

    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(["Do you have experience building real-time systems?"])
    assert answers["Do you have experience building real-time systems?"] == ("Yes, I built Larity")


@pytest.mark.asyncio
async def test_proficiency_gate_beats_stale_persona_answer():
    """A stale persona 'experience with Kubernetes: Yes' must not override the
    resume-skill gate."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(
        context_manager=mock_cm,
        exact_answers={
            _normalise_question_import("do you have experience with kubernetes?"): "Yes"
        },
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert await rag.kb_answer("Do you have experience with Kubernetes?") == "No"


@pytest.mark.asyncio
async def test_proficiency_gate_non_tech_and_numeric_questions_left_to_llm():
    """Soft-skill / domain / numeric-amount questions are NOT proficiency gates:
    they must return None so the grounded LLM answers them."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert await rag.kb_answer("Do you have experience with leadership?") is None
        assert await rag.kb_answer("Do you have experience with team management?") is None
        assert await rag.kb_answer("Are you familiar with our company culture?") is None
        assert (
            await rag.kb_answer("Do you have any experience with the insurance industry?") is None
        )
        assert await rag.kb_answer("How many years of experience with React do you have?") is None
        assert await rag.kb_answer("What are your years of experience in React?") is None


@pytest.mark.asyncio
async def test_proficiency_gate_or_and_multiword_and_such_as():
    """ "X or Y" is order-independent, multi-word resume skills match, "such
    as" enumerations stay in the token stream, and parenthetical noise in a
    "both" compound is ignored."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert await rag.kb_answer("Do you have experience with Vue or React?") == "Yes"
        assert await rag.kb_answer("Do you have experience with React or Vue?") == "Yes"
        assert await rag.kb_answer("Do you have experience with REST APIs?") == "Yes"
        assert await rag.kb_answer("Do you have experience with Vercel AI SDK?") == "Yes"
        assert (
            await rag.kb_answer(
                "Do you have experience with databases such as PostgreSQL and MongoDB?"
            )
            == "Yes"
        )
        assert (
            await rag.kb_answer("Do you have experience with both React and Node (in production)?")
            == "Yes"
        )
        # Both named, but Kubernetes is not on the resume -> No.
        assert (
            await rag.kb_answer("Do you have experience with both Kubernetes and Docker?") == "No"
        )


@pytest.mark.asyncio
async def test_expected_comp_foreign_currency_without_entry_never_leaks_inr():
    """A job in a country whose currency has no compensation-table entry (e.g.
    JPY) must NOT fall back to the INR min-salary figure."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
    with patch("autofill.rag.get_config", return_value=_cfg()):
        # Uncovered currency via job country -> no INR answer.
        assert (
            await rag.kb_answer(
                "What is your expected salary?",
                {"location": "Tokyo, Japan", "description": ""},
            )
            is None
        )
        # Explicit uncovered currency symbol -> no INR answer.
        assert await rag.kb_answer("What is your expected salary in JPY?") is None
        # Covered currencies still resolve.
        assert await rag.kb_answer("What is your expected salary in EUR?") == "€5,400"


@pytest.mark.asyncio
async def test_expected_comp_symbol_currency_ordering():
    """C$/A$/S$ symbols must not be classified as USD (they are CAD/AUD/SGD)."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert await rag.kb_answer("What is your expected salary in C$?") == "C$7,900"
        assert await rag.kb_answer("What is your expected salary in A$?") == "A$8,300"
        assert await rag.kb_answer("What is your expected salary in S$?") == "S$6,700"


@pytest.mark.asyncio
async def test_generate_cover_letter_strips_em_dashes():
    """Generated cover letters have em dashes removed deterministically."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="I built X\u2014and it worked.")
    rag = ScreenerRAG(
        context_manager=mock_cm,
        exact_answers={},
        profile=MagicMock(
            firstName="Aman",
            lastName="Aziz",
            email="a@b.com",
            linkedin="",
            github="",
            website="",
        ),
        store=None,
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        text = await rag.generate_cover_letter(
            job_context={"title": "Role", "company": "Acme", "description": "desc"}
        )
    assert "\u2014" not in text
    assert text == "I built X, and it worked."


@pytest.mark.asyncio
async def test_sensitive_questions_never_answered_by_llm():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(
        return_value='{"Why this role?": "I admire Twilio\'s developer-first API culture.", '
        '"Disability Status": "Yes, I have a disability", '
        '"Are you Hispanic/Latino?": "Yes"}'
    )

    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
    questions = [
        "Disability Status",
        "Are you Hispanic/Latino?",
        "Veteran Status",
        "Why this role?",
    ]

    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(questions)

    assert answers["Disability Status"] == "__ASK_USER__"
    assert answers["Are you Hispanic/Latino?"] == "__ASK_USER__"
    assert answers["Veteran Status"] == "__ASK_USER__"
    assert answers["Why this role?"] == "I admire Twilio's developer-first API culture."
    assert mock_cm.chat.called
    # Only the non-sensitive question was sent to the LLM.
    prompt = mock_cm.chat.call_args.args[0]
    assert "Disability Status" not in prompt
    assert "Veteran Status" not in prompt
    assert "Are you Hispanic/Latino?" not in prompt
    assert "Why this role?" in prompt


@pytest.mark.asyncio
async def test_persona_low_distance_used_high_distance_rejected():
    store = MagicMock()
    store.close = AsyncMock()
    store.search_similar_persona = AsyncMock(
        return_value=[
            {
                "category": "current_location",
                "distance": 0.12,
                "question": "Where do you live?",
                "answer": "Bhopal, India",
            }
        ]
    )
    rag = ScreenerRAG(store=store, exact_answers={})
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1, 0.2])):
        answers = await rag.answer_questions(["What is your current location?"])
    assert answers["What is your current location?"] == "Bhopal, India"

    store.search_similar_persona = AsyncMock(
        return_value=[
            {
                "category": "current_location",
                "distance": 0.55,
                "question": "Where do you live?",
                "answer": "Bhopal, India",
            }
        ]
    )
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1, 0.2])):
        answers = await rag.answer_questions(["What is your current location?"])
    assert answers["What is your current location?"] == "__ASK_USER__"
    assert rag.store.search_similar_persona.called


@pytest.mark.asyncio
async def test_sensitive_question_uses_confident_persona_answer():
    store = MagicMock()
    store.close = AsyncMock()
    store.search_similar_persona = AsyncMock(
        return_value=[
            {
                "category": "gender",
                "distance": 0.08,
                "question": "Gender",
                "answer": "Male",
            }
        ]
    )
    rag = ScreenerRAG(store=store, exact_answers={})
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1, 0.2])):
        answers = await rag.answer_questions(["Gender"])
    assert answers["Gender"] == "Male"


# ── exact-match tier (deterministic, no embeddings) ────────────────


def test_exact_answer_normalisation() -> None:
    rag = ScreenerRAG(exact_answers={"*Gender*": "Male", "where are you based?": "Delhi"})
    assert rag.exact_answer("Gender") == "Male"
    assert rag.exact_answer("  Gender  ") == "Male"
    assert rag.exact_answer("What is your Gender?") is None
    assert rag.exact_answer("Where are you based?") == "Delhi"
    assert rag.exact_answer("") is None


@pytest.mark.asyncio
async def test_exact_answer_tier_beats_embeddings() -> None:
    store = MagicMock()
    store.close = AsyncMock()
    store.search_similar_persona = AsyncMock(
        return_value=[
            {
                "category": "current_location",
                "distance": 0.12,
                "question": "Where do you live?",
                "answer": "Delhi, India",
            }
        ]
    )
    rag = ScreenerRAG(
        store=store, exact_answers={"what is your current location?": "Bhopal, India"}
    )
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1, 0.2])):
        answers = await rag.answer_questions(["What is your current location?"])

    assert answers["What is your current location?"] == "Bhopal, India"
    # The exact tier short-circuits before any embedding search.
    rag.store.search_similar_persona.assert_not_called()


@pytest.mark.asyncio
async def test_exact_answer_learns_into_map() -> None:
    rag = ScreenerRAG(exact_answers={}, store=None)
    assert rag.exact_answer("Are you a current employee?") is None

    rag._exact_answers["are you a current employee?"] = "No"
    assert rag.exact_answer("Are you a current employee?") == "No"


@pytest.mark.asyncio
async def test_scoped_authorization_never_leaks_across_countries() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(
        context_manager=mock_cm,
        exact_answers={},
        scoped_answers={("authorization", "india"): "No"},
    )

    with patch("autofill.rag.get_config", return_value=_cfg()):
        # India JD: the learned India answer applies.
        ind = await rag.kb_answer(
            "Are you authorized to work in the country?",
            job_context={"location": "Bengaluru, India", "description": "..."},
        )
        assert ind == "No"

        # US JD: the India "No" must NOT leak; no same-country answer exists.
        us = await rag.kb_answer(
            "Are you authorized to work in the country?",
            job_context={"location": "San Francisco, USA", "description": "..."},
        )
        assert us is None


@pytest.mark.asyncio
async def test_scoped_authorization_question_named_country_wins() -> None:
    rag = ScreenerRAG(
        exact_answers={},
        scoped_answers={("authorization", "united states"): "Yes"},
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        # The question itself names the country; JD context is irrelevant.
        ans = await rag.kb_answer(
            "Are you legally authorized to work in the United States?",
            job_context={"location": "Bengaluru, India"},
        )
        assert ans == "Yes"


@pytest.mark.asyncio
async def test_scoped_visa_scoped_to_jd_country() -> None:
    rag = ScreenerRAG(
        exact_answers={},
        scoped_answers={("visa", "india"): "No", ("visa", "united states"): "Yes"},
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        assert (
            await rag.kb_answer(
                "Do you require visa sponsorship?",
                job_context={"location": "India"},
            )
            == "No"
        )
        assert (
            await rag.kb_answer(
                "Do you require visa sponsorship?",
                job_context={"location": "USA"},
            )
            == "Yes"
        )


def test_resolve_authorization_policy_home_country() -> None:
    """Authorization policy decides from home vs job country when the persona
    has no country-scoped answer."""
    from autofill.profile import Profile

    profile = Profile(location="Bhopal, India")
    with patch("autofill.rag.get_config", return_value=_cfg()):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=profile, store=None)
        # Home India, job Germany -> not authorized -> "No".
        assert (
            rag.resolve_authorization_policy(
                "Are you authorized to work in Germany?",
                ["Yes", "No"],
                {"location": "Munich, Germany", "description": "..."},
            )
            == "No"
        )
        # Home India, job India -> authorized -> "Yes".
        assert (
            rag.resolve_authorization_policy(
                "Are you authorized to work in India?",
                ["Yes", "No"],
                {"location": "Bengaluru, India", "description": "..."},
            )
            == "Yes"
        )
        # Unknown job country -> conservative default: NOT authorized.
        assert (
            rag.resolve_authorization_policy(
                "Are you authorized to work in the country where this role is based?",
                ["Yes", "No"],
                {"location": "Remote", "description": "..."},
            )
            == "No"
        )
        # Non-authorization questions are ignored.
        assert (
            rag.resolve_authorization_policy(
                "Do you require visa sponsorship?",
                ["Yes", "No"],
                {"location": "Germany", "description": "..."},
            )
            is None
        )


def test_target_country_residence_phrasing_uses_home_country() -> None:
    """'the country you currently reside in' scopes to the candidate's home
    country, not the job's country (the Cohere inversion bug)."""
    from autofill.profile import Profile

    profile = Profile(location="Bhopal, India")
    with patch("autofill.rag.get_config", return_value=_cfg()):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=profile, store=None)
        job = {"location": "San Francisco, USA", "description": "..."}
        # Residence phrasing -> home (india) regardless of the job country.
        assert (
            rag.target_country(
                "Are you authorized to work in the country you currently reside in?",
                job,
            )
            == "india"
        )
        assert (
            rag.target_country(
                "Do you require sponsorship to work in the country you live in?", job
            )
            == "india"
        )
        assert (
            rag.target_country("Are you authorized to work in your home country?", job) == "india"
        )
        # A country named in the question still wins over residence phrasing.
        assert (
            rag.target_country(
                "Are you authorized to work in Germany where you currently reside?",
                job,
            )
            == "germany"
        )
        # No residence phrasing, no named country -> falls through to the job.
        assert (
            rag.target_country(
                "Are you authorized to work in the country where this role is based?",
                job,
            )
            == "united states"
        )
        # No residence phrasing and no job context -> unknown.
        assert rag.target_country("Are you authorized to work in this role?") is None


def test_pick_authorization_answer_sponsorship_list() -> None:
    """Xsolla-style three-way sponsorship options resolve to the right stance."""
    opts = [
        "Yes, I am authorized to work without sponsorship",
        "Yes, but I will require visa sponsorship in the future",
        "No, I will require immediate visa sponsorship",
    ]
    # Not authorized abroad -> the immediate-sponsorship option (leading No).
    assert (
        _pick_authorization_answer(opts, want_yes=False)
        == "No, I will require immediate visa sponsorship"
    )
    # Authorized at home -> the no-sponsorship option.
    assert _pick_authorization_answer(opts, want_yes=True) == opts[0]
    # Plain yes/no lists.
    assert _pick_authorization_answer(["Yes", "No"], want_yes=True) == "Yes"
    assert _pick_authorization_answer(["Yes", "No"], want_yes=False) == "No"


def test_authorization_classifies_uk_spelling_and_variants() -> None:
    """UK spelling and common right-to-work phrasings are country-scoped, so
    they resolve through the deterministic policy instead of the LLM."""
    from autofill.rag import is_scoped_question

    scoped = [
        "Are you authorised to work here in the country of job",
        "Are you authorised to work in the country of the job?",
        "Do you have work authorisation for this role?",
        "Do you have the right to live and work in the UK?",
        "Are you eligible to work in the location of the job?",
        "Are you legally entitled to work in Germany?",
        "Do you require a work permit to work in this country?",
        "Will you require immigration sponsorship?",
    ]
    for q in scoped:
        assert is_scoped_question(q), f"not scoped: {q!r}"


def test_country_from_text_city_fallback() -> None:
    """City-only job locations resolve to their country via the city map."""
    from autofill.rag import _country_from_text

    cases = {
        "San Francisco": "united states",
        "New York": "united states",
        "Boston": "united states",
        "Menlo Park, CA": "united states",
        "Barcelona": "spain",
        "London": "united kingdom",
        "Stockholm": "sweden",
        "Malmö": "sweden",
        "Montreal": "canada",
        "Bengaluru": "india",
        "Kuala Lumpur": "malaysia",
        "Budapest, Hungary (Hybrid)": "hungary",
        "Remote": None,
        "All locations": None,
        "London, Ontario, Canada": "canada",  # country name beats the city
    }
    for loc, expected in cases.items():
        assert _country_from_text(loc) == expected, f"{loc!r} -> {_country_from_text(loc)}"


def test_authorization_policy_city_locations() -> None:
    """The reported bug: 'authorised ... country of the job' with a city-only
    location must resolve to 'No' for a foreign job, 'Yes' for an India job."""
    from autofill.profile import Profile

    profile = Profile(location="Bhopal, India")
    with patch("autofill.rag.get_config", return_value=_cfg()):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=profile, store=None)
        q = "Are you authorised to work here in the country of the job?"
        assert (
            rag.resolve_authorization_policy(
                q, ["Yes", "No"], {"location": "San Francisco", "description": "..."}
            )
            == "No"
        )
        assert (
            rag.resolve_authorization_policy(
                q, ["Yes", "No"], {"location": "Barcelona", "description": "..."}
            )
            == "No"
        )
        # India-located job -> authorized.
        assert (
            rag.resolve_authorization_policy(
                q, ["Yes", "No"], {"location": "Bengaluru", "description": "..."}
            )
            == "Yes"
        )


def test_authorization_policy_question_names_country_wins() -> None:
    """A country named in the question beats the job context, and an unknown
    job country defaults to NOT authorized."""
    from autofill.profile import Profile

    profile = Profile(location="Bhopal, India")
    with patch("autofill.rag.get_config", return_value=_cfg()):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=profile, store=None)
        assert (
            rag.resolve_authorization_policy(
                "Are you legally authorized to work in the United States?",
                ["Yes", "No"],
                {"location": "", "description": ""},
            )
            == "No"
        )
        assert (
            rag.resolve_authorization_policy(
                "Are you legally authorized to work in India?",
                ["Yes", "No"],
                {"location": "San Francisco", "description": ""},
            )
            == "Yes"
        )


@pytest.mark.asyncio
async def test_learn_scoped_qualifies_question_text(tmp_path) -> None:
    """A country-scoped answer is stored under (category, country) — never the
    global exact tier — and the write is isolated from the real persona.json."""
    rag = ScreenerRAG(exact_answers={}, scoped_answers={}, store=None)
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    persona_txt = tmp_path / "persona.txt"
    persona_txt.write_text("Candidate Profile:\n- x: y\n\nFrom Resume:\n- stuff\n")
    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch("autofill.rag.PERSONA_TXT", persona_txt),
    ):
        ok = await rag.learn(
            "Are you authorized to work in the country?", "No", country="united states"
        )

    assert ok is True
    # Learned under the scoped map, not the global exact tier.
    assert rag._scoped_answers[("authorization", "united states")] == "No"
    assert rag.exact_answer("Are you authorized to work in the country?") is None
    data = json.loads(persona_json.read_text())
    assert any(
        e.get("question") == "Are you authorized to work in the country? (United States)"
        and e.get("country") == "united states"
        for e in data["answers"]
    )


def _persona_store(rows):
    store = MagicMock()
    store.close = AsyncMock()
    store.search_similar_persona = AsyncMock(return_value=rows)
    return store


@pytest.mark.asyncio
async def test_persona_scoped_fact_never_answers_unknown_country() -> None:
    # A country-qualified persona fact ("...in India?") must never answer a
    # scoped question when the target country cannot be established.
    store = _persona_store(
        [
            {
                "category": "work_authorization",
                "question": "Are you legally authorized to work in India?",
                "answer": "Yes",
                "distance": 0.1,
            }
        ]
    )
    rag = ScreenerRAG(store=store, exact_answers={}, scoped_answers={})
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1])):
        assert await rag.kb_answer("Are you authorized to work in the country?") is None
        # India JD: the India fact applies.
        assert (
            await rag.kb_answer(
                "Are you authorized to work in the country?",
                job_context={"location": "Bengaluru, India"},
            )
            == "Yes"
        )
        # US JD: the India fact must NOT apply.
        assert (
            await rag.kb_answer(
                "Are you authorized to work in the country?",
                job_context={"location": "San Francisco, USA"},
            )
            is None
        )


@pytest.mark.asyncio
async def test_persona_scoped_fact_guards_legacy_category_keys() -> None:
    # Entries indexed under the raw "authorization" category (pre-scoping)
    # get the same country guard as "work_authorization".
    store = _persona_store(
        [
            {
                "category": "authorization",
                "question": "Are you legally authorized to work in India?",
                "answer": "Yes",
                "distance": 0.1,
            }
        ]
    )
    rag = ScreenerRAG(store=store, exact_answers={}, scoped_answers={})
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1])):
        assert (
            await rag.kb_answer(
                "Are you legally authorized to work in the United States?",
                job_context={"location": "USA"},
            )
            is None
        )


@pytest.mark.asyncio
async def test_persona_unscoped_legacy_fact_requires_known_country() -> None:
    # A legacy unscoped fact ("...where the job is located?") is not
    # country-verifiable and must not answer a scoped question either.
    store = _persona_store(
        [
            {
                "category": "authorization",
                "question": (
                    "Are you legally authorized to work in the country where the job is located?"
                ),
                "answer": "No",
                "distance": 0.1,
            }
        ]
    )
    rag = ScreenerRAG(store=store, exact_answers={}, scoped_answers={})
    with patch("autofill.rag._embed_text", new=AsyncMock(return_value=[0.1])):
        assert (
            await rag.kb_answer(
                "Are you authorized to work in the country?",
                job_context={"location": "San Francisco, USA"},
            )
            is None
        )


@pytest.mark.asyncio
async def test_country_from_text_prefers_earliest_mention() -> None:
    from autofill.rag import _country_from_text

    assert _country_from_text("Are you authorized to work in the United States or Canada?") == (
        "united states"
    )
    assert _country_from_text("Remote in India or Singapore") == "india"
    assert _country_from_text("no country here") is None


def test_country_from_text_matches_us_abbreviations_at_end_and_before_punctuation() -> None:
    from autofill.rag import _country_from_text

    for q in (
        "Are you authorized to work in the U.S.?",
        "Will you require sponsorship in the U.S.?",
        "Are you authorized to work in U.S.A?",
        "eligible to work in the USA?",
        "work authorization in the U.S.",
        "based in U.S. or Canada",
    ):
        assert _country_from_text(q) == "united states", q
    assert _country_from_text("Are you authorized to work in the U.K.?") == "united kingdom"


def test_normalize_start_date_fixes_typo_and_relative_offsets() -> None:
    from autofill.rag import _normalize_start_date

    assert _normalize_start_date("Immeditely") == "Immediately"
    assert _normalize_start_date("immediate") == "Immediately"
    assert _normalize_start_date("ASAP") == "Immediately"
    assert _normalize_start_date("in 2 weeks") == "2 weeks"
    assert _normalize_start_date("within one month") == "1 month"
    assert _normalize_start_date("3 weeks") == "3 weeks"
    assert _normalize_start_date("any time") == "any time"


def test_start_date_question_is_recognized() -> None:
    from autofill.rag import _PERSONAL_RULES

    for q in (
        "What is the earliest date you are available to start this position?",
        "When can you start?",
        "How soon can you join?",
        "What is your notice period?",
    ):
        key = next((k for p, k in _PERSONAL_RULES if p.search(q)), None)
        assert key == "start_date", q


@pytest.mark.asyncio
async def test_start_date_persona_answer_is_normalized() -> None:
    from autofill.rag import ScreenerRAG

    rag = ScreenerRAG(
        context_manager=MagicMock(),
        exact_answers={},
        scoped_answers={},
        store=None,
    )
    # Persona store unavailable here (store=None); simulate via the exact tier
    # with the SAME question text so the exact match fires.
    rag._exact_answers["what is the earliest date you are available to start this position?"] = (
        "Immeditely"
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        ans = await rag.kb_answer(
            "What is the earliest date you are available to start this position?"
        )
    assert ans == "Immediately"


@pytest.mark.asyncio
async def test_authorization_us_scoped_answer_matched_when_question_uses_us() -> None:
    rag = ScreenerRAG(
        context_manager=MagicMock(),
        exact_answers={},
        scoped_answers={("authorization", "united states"): "No"},
        store=None,
    )
    ans = await rag.kb_answer("Are you authorized to work in the U.S.?")
    assert ans == "No"


@pytest.mark.asyncio
async def test_job_context_appears_in_llm_prompt() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(
        return_value='{"Why this role?": "Your infra platform fits my scaling work."}'
    )
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={}, store=None)
    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(
            ["Why this role?"],
            job_context={
                "title": "Senior Backend Engineer",
                "company": "Acme",
                "location": "Berlin, Germany",
                "description": "We scale distributed systems to millions of requests.",
            },
        )

    assert answers["Why this role?"] == "Your infra platform fits my scaling work."
    prompt = mock_cm.chat.await_args.args[0]
    assert "Senior Backend Engineer" in prompt
    assert "Acme" in prompt
    assert "We scale distributed systems" in prompt


@pytest.mark.asyncio
async def test_generate_cover_letter_grounds_on_resume_and_company() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="A real structured cover letter body.")
    store = MagicMock()
    store.search_similar_chunks = AsyncMock(
        return_value=[
            {"content": "Delivered 5 high-impact projects with a 4.8+ client rating."},
            {"content": "Larity: Tauri desktop app, pgvector, Redis, Groq, Gemini."},
            {"content": "Co-Founder and Lead Developer at Singularity Works."},
        ]
    )
    rag = ScreenerRAG(
        context_manager=mock_cm,
        exact_answers={},
        profile=MagicMock(
            firstName="Aman",
            lastName="Aziz",
            email="a@b.com",
            linkedin="https://linkedin.com/in/aman",
            github="https://github.com/aman",
            website="https://aman.dev",
        ),
        store=store,
    )
    with (
        patch("autofill.rag.get_config", return_value=_cfg()),
        patch("autofill.rag._embed_text", AsyncMock(return_value=[0.1, 0.2])),
    ):
        text = await rag.generate_cover_letter(
            job_context={
                "title": "Frontend Engineer",
                "company": "SingleStore",
                "location": "Portugal",
                "description": "Real-time data platform. React and TypeScript required.",
            }
        )

    assert text == "A real structured cover letter body."
    prompt = mock_cm.chat.await_args.args[0]
    assert "SingleStore" in prompt
    assert "Frontend Engineer" in prompt
    assert "4.8+" in prompt
    assert "Tauri" in prompt
    # Resume facts must be presented as grounding, and the JD as data.
    assert "Verified facts retrieved from the candidate's resume" in prompt
    assert "<job_description>" in prompt
    # The skills whitelist is passed and the inflated persona text is NOT
    # part of the grounding (persona.txt used to leak "Kubernetes: Yes").
    assert "STRICT SKILLS RULE" in prompt
    assert "Kubernetes" in prompt  # named in the whitelist as forbidden
    assert "Candidate Background & Persona" not in prompt
    system = mock_cm.chat.await_args.kwargs["system_prompt"]
    assert "four short paragraphs" in system
    assert "200 words" in system
    assert "Skills are strictly limited to the whitelist" in system


@pytest.mark.asyncio
async def test_generate_cover_letter_returns_empty_without_grounding() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="unused")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={}, store=None)
    with patch("autofill.rag.get_config", return_value=_cfg()):
        text = await rag.generate_cover_letter(job_context={})
    assert text == ""
    mock_cm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_generate_cover_letter_surfaces_llm_failure_as_empty() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(side_effect=RuntimeError("boom"))
    rag = ScreenerRAG(
        context_manager=mock_cm,
        exact_answers={},
        profile=MagicMock(
            firstName="Aman",
            lastName="Aziz",
            email="a@b.com",
            linkedin="",
            github="",
            website="",
        ),
        store=None,
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        text = await rag.generate_cover_letter(
            job_context={"title": "Role", "company": "Acme", "description": "desc"}
        )
    assert text == ""


# ── Tier 3: grounded LLM for selects ───────────────────────────────


@pytest.mark.asyncio
async def test_answer_select_validates_llm_option() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value='{"Question A?": "Yes"}')
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={}, store=None)
    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(
            [{"question": "Question A?", "kind": "select", "options": ["Yes", "No"]}]
        )
    assert answers["Question A?"] == "Yes"
    prompt = mock_cm.chat.await_args.args[0]
    assert "Options" in prompt and "Yes" in prompt


@pytest.mark.asyncio
async def test_answer_select_hallucinated_option_never_filled() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value='{"Question A?": "Banana"}')
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={}, store=None)
    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(
            [{"question": "Question A?", "kind": "select", "options": ["Yes", "No"]}]
        )
    assert answers["Question A?"] == ASK_USER


@pytest.mark.asyncio
async def test_answer_prompt_includes_decision_policy_rules() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={}, store=None)
    with patch("autofill.rag.get_config", return_value=_cfg()):
        await rag.answer_questions(
            [
                {
                    "question": "Do you agree to our data use?",
                    "kind": "select",
                    "options": ["Yes", "No"],
                },
                {
                    "question": "Email me about jobs?",
                    "kind": "select",
                    "options": ["Yes", "No"],
                },
                {
                    "question": "Have you ever worked for Cognizant?",
                    "kind": "select",
                    "options": ["Yes", "No"],
                },
            ]
        )
    system = mock_cm.chat.await_args.kwargs["system_prompt"]
    assert "CONSENT" in system
    assert "OPT-IN" in system
    assert "AFFILIATION" in system


@pytest.mark.asyncio
async def test_gather_context_unions_queries_and_drops_noise() -> None:
    store = MagicMock()
    store.search_similar_chunks = AsyncMock(
        return_value=[
            {"section": "projects", "content": "Larity: Tauri app, pgvector."},
            {"section": "table", "content": "| --- | --- |"},
            {"section": "", "content": "no section chunk"},
        ]
    )
    rag = ScreenerRAG(exact_answers={}, store=store)
    with patch("autofill.rag._embed_text", AsyncMock(return_value=[0.1, 0.2])):
        ctx = await rag._gather_context(["Describe your largest project"])
    assert "Larity" in ctx
    assert "| ---" not in ctx
    assert "no section chunk" in ctx


@pytest.mark.asyncio
async def test_answer_select_kb_value_must_map_or_falls_to_llm() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value='{"Work model?": "Remote"}')
    rag = ScreenerRAG(
        context_manager=mock_cm, exact_answers={"Work model?": "Xylophone"}, store=None
    )
    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(
            [{"question": "Work model?", "kind": "select", "options": ["Remote", "Onsite"]}]
        )
    # KB value "Xylophone" doesn't map to an option → LLM decides → "Remote".
    assert answers["Work model?"] == "Remote"


@pytest.mark.asyncio
async def test_answer_prompt_includes_motivation_rule() -> None:
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(
        return_value='{"Why are you looking for a new role?": "Growth and domain fit."}'
    )
    rag = ScreenerRAG(context_manager=mock_cm, exact_answers={}, store=None)
    with patch("autofill.rag.get_config", return_value=_cfg()):
        answers = await rag.answer_questions(
            [{"question": "Why are you looking for a new role?", "kind": "text", "options": []}]
        )
    assert answers["Why are you looking for a new role?"] == "Growth and domain fit."
    system_prompt = mock_cm.chat.await_args.kwargs.get("system_prompt", "")
    assert "MOTIVATION / INTENT" in system_prompt


# ── Authorization / sponsorship INTENT regression tests ─────────────────────
# These questions are verbatim from the 2026-08-05 overnight run, where the
# deterministic policy answered them WRONG (defaulting a foreign-country
# authorization question to "Yes" because the phrasing contained "visa").


def _geo_rag() -> ScreenerRAG:
    from autofill.profile import Profile

    profile = Profile(
        location="Bhopal, India",
        customAnswers={"What is your nationality?": "Indian"},
    )
    return ScreenerRAG(exact_answers={}, scoped_answers={}, profile=profile, store=None)


@pytest.mark.parametrize(
    "question,location,expected",
    [
        # Compound: authorization + "without requiring visa sponsorship" → No.
        (
            "Are you legally authorized to work in the country where you are applying, "
            "without requiring current or future visa sponsorship?",
            "Biassono, Italy",
            "No",
        ),
        ("Are you legally authorized to work in the United States?", "San Francisco", "No"),
        (
            "Are you authorized to work in the country where the job is located?",
            "San Francisco",
            "No",
        ),
        ("Are you currently authorized to work in the U.S.A?", "San Mateo", "No"),
        (
            "Are you legally eligible to work for any employer in the United States?",
            "New York",
            "No",
        ),
        (
            "Are you able to work legally in the United States without the need for "
            "visa sponsorship now or in the future?",
            "Austin",
            "No",
        ),
        # Home country authorization → Yes.
        ("Are you legally authorized to work in India?", "Bengaluru", "Yes"),
        (
            "Are you authorized to work in the country you currently reside in?",
            "Bhopal, India",
            "Yes",
        ),
        # Genuine sponsorship-requirement → Yes abroad, No at home.
        ("Do you require visa sponsorship now or in the future?", "San Francisco", "Yes"),
        (
            "Will you now or in the future require sponsorship for employment visa "
            "status (e.g., H-1B visa status)?",
            "New York",
            "Yes",
        ),
        (
            "Will you require sponsorship from Northwood for employment now or in "
            "the future (i.e. H1-B visa)",
            "Torrance, CA",
            "Yes",
        ),
        ("Do you now, or will you in the future, need US visa sponsorship?", "Concord", "Yes"),
        ("Do you require visa sponsorship now or in the future?", "Mumbai", "No"),
    ],
)
@pytest.mark.asyncio
async def test_authorization_visa_intent(question, location, expected) -> None:
    """Authorization-eligibility questions answer from the job/home country
    even when the phrasing contains "visa"/"sponsorship"; genuine
    sponsorship-requirement questions answer "Yes" abroad, "No" at home."""
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        job = {"location": location, "title": "Engineer", "description": ""}
        visa = rag.resolve_visa_policy(question, ["Yes", "No"], job)
        auth = rag.resolve_authorization_policy(question, ["Yes", "No"], job)
        picked = visa or auth
        assert picked == expected


@pytest.mark.asyncio
async def test_document_declarative_never_auto_answered() -> None:
    """'Are you able to provide a French/EU passport or valid visa?' is a
    personal-fact question about the candidate's actual documents — never a
    policy default. Previously it was answered 'Yes' (kb), fabricating a
    document the candidate does not hold."""
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        job = {"location": "Paris", "title": "Engineer", "description": ""}
        for q in (
            "Are you able to provide a French or EU ID/Passport or a valid visa that "
            "allows you to work for more than 6 months",
            "I am able to provide a French or EU ID/Passport or a valid visa that "
            "allows you to work for more than 6 months",
        ):
            assert rag.resolve_visa_policy(q, ["Yes", "No"], job) is None
            assert rag.resolve_authorization_policy(q, ["Yes", "No"], job) is None
            assert await rag.kb_answer(q, job_context=job) is None


# ── Relocation-willingness regression tests ─────────────────────────────────
# The candidate is willing to relocate to a first-world country, NOT to a
# third-world one. The run showed the same relocation question answered both
# "Yes" and "No" (LLM guessing); the deterministic policy fixes it.


@pytest.mark.parametrize(
    "location,expected",
    [
        ("San Francisco", "Yes"),
        ("New York", "Yes"),
        ("Munich Office", "Yes"),
        ("London - The River Building HQ", "Yes"),
        ("Singapore", "Yes"),
        ("Bangkok", "No"),
        ("Jakarta", "No"),
        ("Hyderabad (Telangana)", "Yes"),  # home country — no relocation needed
        ("Mumbai", "Yes"),  # home country
    ],
)
def test_relocation_policy_first_world_yes_third_world_no(location, expected) -> None:
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        job = {"location": location, "title": "Engineer", "description": ""}
        picked = rag.resolve_relocation_policy(
            "Are you willing to relocate for this role?", ["Yes", "No"], job
        )
        assert picked == expected


def test_relocation_unknown_country_returns_none() -> None:
    """A relocation question whose destination country cannot be determined is
    NOT guessed — it falls through to the LLM/user."""
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        job = {"location": "Remote", "title": "Engineer", "description": ""}
        assert (
            rag.resolve_relocation_policy(
                "Are you willing to relocate for this role?", ["Yes", "No"], job
            )
            is None
        )


def test_relocation_composite_residence_question_deferred_to_llm() -> None:
    """A composite 'based in X, or willing to relocate?' question is owned by
    the residence policy (which defers to the LLM when a willingness option
    exists) — a flat relocation Yes/No must not clobber the nuanced answer."""
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        job = {"location": "Munich, Germany", "title": "Engineer", "description": ""}
        assert (
            rag.resolve_relocation_policy(
                "Are you currently based in Munich, or willing to relocate?",
                ["Yes", "No", "Not based in Munich, but open to relocating"],
                job,
            )
            is None
        )


# ── Job-location extraction hardening ────────────────────────────────────────


@pytest.mark.parametrize(
    "location,expected",
    [
        ("Remote - United States", "united states"),
        ("US (Remote)", "united states"),
        ("Remote (US)", "united states"),
        ("MYS - Kuala Lumpur", "malaysia"),
        ("Concord Office", "united states"),
        ("Zürich Office", "switzerland"),
        ("Iasi Office", "romania"),
        ("London - The River Building HQ", "united kingdom"),
        ("Milton Keynes Office", "united kingdom"),
        ("RWC HQ", "united states"),
        ("Biassono, Italy", "italy"),
        ("US", "united states"),
        ("UK", "united kingdom"),
        ("Remote", None),
        ("All locations", None),
        ("Global", None),
        ("Remote - International", None),
        ("Cambridge Office", None),
    ],
)
def test_country_from_location(location, expected) -> None:
    from autofill.rag import _country_from_location

    assert _country_from_location(location) == expected


# ── Affiliation / employment fabrication regression tests ────────────────────
# The run fabricated prior employment: "Have you previously worked for one of
# our sister brand companies?" was answered "Agoda"/"KAYAK" — companies the
# candidate never worked at. These must resolve to the negative stance.


@pytest.mark.parametrize(
    "question,options,expected",
    [
        # Multi-select of sister brands: no "No"/"None" option → decline/blank.
        (
            "Have you previously worked or currently work for one of our sister brand companies?",
            ["Agoda", "Booking Holdings", "KAYAK", "OpenTable"],
            "",
        ),
        # Yes/No select → negative option.
        ("Are you a current or former employee of Deloitte?", ["Yes", "No"], "No"),
        (
            "Are you a family member (spouse, parent, child, sibling, in-law) of a current "
            "Deloitte employee?",
            ["Yes", "No"],
            "No",
        ),
        (
            "Do you know, or are you related to, anyone who currently works at Dominion Dynamics?",
            ["Yes", "No"],
            "No",
        ),
        (
            "Have you ever been employed by Anduril or any company that Anduril has acquired?",
            ["Yes", "No"],
            "No",
        ),
        ("Have you previously worked as an intern or co-op at SharkNinja?", ["Yes", "No"], "No"),
        ("Are you related to any current Saronic employees?", ["Yes", "No"], "No"),
        # A form offering "None of the above" picks it.
        (
            "Have you previously worked or currently work for one of our sister brand companies?",
            ["Agoda", "None of the above", "KAYAK"],
            "None of the above",
        ),
        # Skill questions must NOT be treated as affiliations.
        ("Have you worked professionally with Python and Docker?", ["Yes", "No"], None),
        ("Do you have prior experience using Neo4j?", ["Yes", "No"], None),
        # Duration / sourcing / generic-employment questions must NOT be
        # answered "No" or blanked (previously blanked as affiliations).
        ("How many years of professional experience do you have?", ["Yes", "No"], None),
        ("How did you hear about this position?", ["LinkedIn", "Referral", "Google"], None),
        ("Have you worked for the past 3 years?", ["Yes", "No"], None),
        ("Did someone refer you to this role?", ["Yes", "No"], "No"),
        # A decline-only negative option is still a valid fillable answer.
        (
            "Are you related to any current Acme employees?",
            ["Yes", "I do not wish to answer"],
            "I do not wish to answer",
        ),
    ],
)
def test_affiliation_policy(question, options, expected) -> None:
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        got = rag.resolve_affiliation_policy(question, options, {})
        assert got == expected


# ── Composite residence free-text: no LLM "Yes" fabrication ──────────────────


@pytest.mark.asyncio
async def test_residence_composite_free_text_no_yes_fabrication() -> None:
    """'Are you currently based in Europe (or willing to work from a European
    timezone)?' is free-text (no options). The residence facet answers 'No' —
    a truthful residence answer, never the LLM's fabricated 'Yes' (observed
    three times in the run for Somnia)."""
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        jc = {"location": "Remote", "title": "Engineer", "description": ""}
        assert (
            rag.resolve_residence_policy(
                "Are you currently based in Europe (or willing to work from a "
                "European timezone long-term)?",
                [],
                jc,
            )
            == "No"
        )


@pytest.mark.asyncio
async def test_residence_composite_with_willingness_option_deferred() -> None:
    """A composite residence question WITH a willingness option stays the
    LLM's to decide — never short-circuited by kb_answer's option-less call."""
    rag = _geo_rag()
    with patch("autofill.rag.get_config", return_value=_cfg()):
        jc = {"location": "Munich, Germany", "title": "Engineer", "description": ""}
        opts = ["Yes", "No", "Not based in Munich, but open to relocating"]
        assert (
            rag.resolve_residence_policy(
                "Are you currently based in Munich, or willing to relocate?", opts, jc
            )
            is None
        )
        # kb_answer must also defer (it cannot see the options).
        assert (
            await rag.kb_answer(
                "Are you currently based in Munich, or willing to relocate?", job_context=jc
            )
            is None
        )
