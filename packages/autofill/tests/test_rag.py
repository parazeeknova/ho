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
async def test_sensitive_questions_never_answered_by_llm(tmp_path):
    """Identity questions are never LLM-answered or guessed: without a grilled
    fact they resolve to ASK_USER; with one they answer deterministically."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(
        return_value='{"Why this role?": "I admire Twilio\'s developer-first API culture.", '
        '"Disability Status": "Yes, I have a disability", '
        '"Are you Hispanic/Latino?": "Yes"}'
    )

    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "", "version": 1, "answers": []}))
    questions = [
        "Disability Status",
        "Are you Hispanic/Latino?",
        "Veteran Status",
        "Why this role?",
    ]

    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch("autofill.rag.get_config", return_value=_cfg()),
    ):
        rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
        answers = await rag.answer_questions(questions)

    assert answers["Disability Status"] == "__ASK_USER__"
    assert answers["Are you Hispanic/Latino?"] == "__ASK_USER__"
    assert answers["Veteran Status"] == "__ASK_USER__"
    assert answers["Why this role?"] == "I admire Twilio's developer-first API culture."
    assert mock_cm.chat.called
    # Only the non-sensitive question was sent to the LLM.
    prompt = mock_cm.chat.call_args.args[0]
    assert "Disability Status" not in prompt


@pytest.mark.asyncio
async def test_sensitive_questions_answer_from_grilled_persona(tmp_path):
    """With grilled facts present, identity questions resolve deterministically
    and the LLM is never invoked for them."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value='{"Why this role?": "I admire Twilio."}')
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(
        json.dumps(
            {
                "name": "Test",
                "version": 1,
                "answers": [
                    {
                        "category": "disability",
                        "question": "Do you have a disability?",
                        "answer": "No",
                    },
                    {
                        "category": "veteran_status",
                        "question": "Are you a veteran?",
                        "answer": "No",
                    },
                ],
            }
        )
    )
    questions = ["Disability Status", "Veteran Status"]

    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch("autofill.rag.get_config", return_value=_cfg()),
    ):
        rag = ScreenerRAG(context_manager=mock_cm, exact_answers={})
        answers = await rag.answer_questions(questions)

    assert answers["Disability Status"] == "No"
    assert answers["Veteran Status"] == "No"
    # Both resolved from the grilled persona: the LLM was never invoked.
    assert mock_cm.chat.called is False


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
    store.search_similar_chunks = AsyncMock(return_value=[])
    rag = ScreenerRAG(store=store, exact_answers={})
    # Keep the LLM tier out of this test: the gate under test is the persona
    # distance threshold, so any unresolved question must stay unresolved.
    rag.cm.chat = AsyncMock(side_effect=Exception("LLM disabled in this test"))
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
    persona_json = tmp_path / "persona.json"
    rag = ScreenerRAG(exact_answers={}, scoped_answers={}, store=None)
    with patch("autofill.rag.PERSONA_JSON", persona_json):
        ok = await rag.learn("Are you authorized to work in the country?", "No", country="india")

    assert ok is True
    # Learned under the scoped map, not the global exact tier.
    assert rag._scoped_answers[("authorization", "india")] == "No"
    assert rag.exact_answer("Are you authorized to work in the country?") is None
    persisted = json.loads(persona_json.read_text())
    assert persisted["answers"] == [
        {
            "category": "authorization",
            "question": "Are you authorized to work in the country? (India)",
            "answer": "No",
            "country": "india",
        }
    ]


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
    system = mock_cm.chat.await_args.kwargs["system_prompt"]
    assert "four short paragraphs" in system
    assert "200 words" in system


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


@pytest.mark.asyncio
async def test_identity_answers_resolve_from_grilled_persona(tmp_path):
    """Identity facts answer deterministically from persona.json categories."""
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(
        json.dumps(
            {
                "name": "Harsh Sahu",
                "version": 2,
                "answers": [
                    {
                        "category": "gender_identity",
                        "question": "What is your gender?",
                        "answer": "Male",
                    },
                    {
                        "category": "disability",
                        "question": "Do you have a disability?",
                        "answer": "No",
                    },
                    {
                        "category": "veteran_status",
                        "question": "Are you a veteran?",
                        "answer": "No",
                    },
                    {
                        "category": "employee_relation",
                        "question": "Related to anyone employed?",
                        "answer": "No",
                    },
                ],
            }
        )
    )
    with patch("autofill.rag.PERSONA_JSON", persona_json):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=MagicMock(), store=None)
        assert await rag.kb_answer("Select your gender") == "Male"
        assert await rag.kb_answer("Do you identify as a person with a disability?") == "No"
        assert await rag.kb_answer("Are you a veteran of the U.S. Armed Forces?") == "No"
        assert await rag.kb_answer("Are you related to anyone employed at our company?") == "No"


@pytest.mark.asyncio
async def test_identity_answers_never_guessed_without_persona(tmp_path):
    """Without a grilled value, identity questions stay unresolved (ASK_USER)."""
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "", "version": 1, "answers": []}))
    with patch("autofill.rag.PERSONA_JSON", persona_json):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=MagicMock(), store=None)
        assert await rag.kb_answer("Select your gender") is None
        assert await rag.kb_answer("Are you a person with a disability?") is None
        assert await rag.kb_answer("What is your ethnicity?") is None


@pytest.mark.asyncio
async def test_identity_category_loader_first_entry_wins(tmp_path):
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(
        json.dumps(
            {
                "name": "",
                "version": 1,
                "answers": [
                    {"category": "gender_identity", "question": "A", "answer": "Male"},
                    {"category": "gender_identity", "question": "B", "answer": "Female"},
                    {"category": "general", "question": "C", "answer": "ignored"},
                ],
            }
        )
    )
    with patch("autofill.rag.PERSONA_JSON", persona_json):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, profile=MagicMock(), store=None)
        assert rag._category_answers["gender_identity"] == "Male"
        assert "general" not in rag._category_answers
