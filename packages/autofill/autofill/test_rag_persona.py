"""Unit tests for persona-grounded ScreenerRAG retrieval."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autofill.rag import ASK_USER, ScreenerRAG


def _rag(store=None, profile_location="Bhopal, India"):
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, store=store, exact_answers={}, scoped_answers={})
    rag.profile.location = profile_location
    return rag, mock_cm


def _store(results=None):
    store = MagicMock()
    store.search_similar_persona = AsyncMock(return_value=results or [])
    return store


def _embed_stub(emb=None):
    return patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=emb or [0.1] * 1024))


@pytest.mark.asyncio
async def test_persona_retrieval_answers_personal_question():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    store = _store(
        [
            {
                "category": "current_location",
                "question": "What is your current location?",
                "answer": "Delhi, India",
                "content": "Q: ...\nA: ...",
                "distance": 0.2,
            }
        ]
    )
    rag = ScreenerRAG(context_manager=mock_cm, store=store, exact_answers={})
    with _embed_stub():
        answers = await rag.answer_questions(["What is your current location?"])

    assert answers["What is your current location?"] == "Delhi, India"
    assert not mock_cm.chat.called


@pytest.mark.asyncio
async def test_persona_no_confident_match_falls_back_to_ask_user():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    # Distance above the threshold -> not a confident match.
    store = _store(
        [
            {
                "category": "current_location",
                "question": "What is your current location?",
                "answer": "Delhi, India",
                "content": "...",
                "distance": 0.95,
            }
        ]
    )
    rag = ScreenerRAG(context_manager=mock_cm, store=store, exact_answers={})
    with _embed_stub():
        answers = await rag.answer_questions(["What is your current location?"])

    # current_location is personal and non-deterministic -> prompt the user.
    assert answers["What is your current location?"] == ASK_USER


@pytest.mark.asyncio
async def test_persona_overrides_deterministic_visa_rule():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    store = _store(
        [
            {
                "category": "visa_sponsorship",
                "question": "Do you require visa sponsorship?",
                "answer": "Yes",
                "content": "...",
                "distance": 0.05,
            }
        ]
    )
    rag = ScreenerRAG(context_manager=mock_cm, store=store, exact_answers={})
    with _embed_stub():
        answers = await rag.answer_questions(["Do you require visa sponsorship?"])

    # Persona answer wins over the hardcoded "No" fallback.
    assert answers["Do you require visa sponsorship?"] == "Yes"


@pytest.mark.asyncio
async def test_work_authorization_respects_country():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    store = _store(
        [
            # India chunk is semantically closest but the question is about the US.
            {
                "category": "work_authorization",
                "question": "Are you legally authorized to work in India?",
                "answer": "Yes",
                "content": "...",
                "distance": 0.05,
            },
            {
                "category": "work_authorization",
                "question": "Are you legally authorized to work in the United States?",
                "answer": "No, I would require visa sponsorship.",
                "content": "...",
                "distance": 0.2,
            },
        ]
    )
    rag = ScreenerRAG(context_manager=mock_cm, store=store, exact_answers={})
    with _embed_stub():
        answers = await rag.answer_questions(
            ["Are you legally authorized to work in the United States?"]
        )

    assert (
        answers["Are you legally authorized to work in the United States?"]
        == "No, I would require visa sponsorship."
    )


@pytest.mark.asyncio
async def test_no_store_skips_persona_retrieval():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, store=None, exact_answers={})
    with _embed_stub():
        answers = await rag.answer_questions(["What is your current location?"])

    assert answers["What is your current location?"] == ASK_USER


@pytest.mark.asyncio
async def test_scoped_question_does_not_substring_match_unrelated_custom_answer():
    """A short scoped label ("Work Authorization") must not substring-match a
    longer custom key (a visa-sponsorship question) and leak its answer."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    store = _store(
        [
            {
                "category": "visa",
                "question": "Do you require visa sponsorship?",
                "answer": "Yes",
                "content": "...",
                "distance": 0.05,
            }
        ]
    )
    rag = ScreenerRAG(context_manager=mock_cm, store=store, exact_answers={})
    rag.profile.customAnswers = {
        "Will you now or in the future require company sponsorship to retain or "
        "extend your work authorization in the country where the job is located?": "Yes"
    }
    with _embed_stub():
        # No country named in the question or job context -> cannot answer a
        # scoped authorization question confidently -> prompt, never leak "Yes".
        answers = await rag.answer_questions(["Work Authorization"])

    assert answers["Work Authorization"] == ASK_USER


@pytest.mark.asyncio
async def test_general_visa_question_answered_by_exact_custom_match():
    """A self-contained general visa question ("Do you require visa
    sponsorship?") still resolves from an EXACT custom-answer match."""
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value="{}")
    rag = ScreenerRAG(context_manager=mock_cm, store=None, exact_answers={})
    rag.profile.customAnswers = {"Do you require visa sponsorship?": "Yes"}
    with _embed_stub():
        answers = await rag.answer_questions(["Do you require visa sponsorship?"])

    assert answers["Do you require visa sponsorship?"] == "Yes"


@pytest.mark.asyncio
async def test_config_persona_and_min_salary_used():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value=json.dumps({"Tell us about yourself": "grounded answer"}))
    candidate = MagicMock()
    candidate.persona = "A unique persona string for tests"
    candidate.min_salary = "9L INR per annum"
    cfg = MagicMock()
    cfg.candidate = candidate
    rag = ScreenerRAG(context_manager=mock_cm, store=None, exact_answers={})
    with patch("autofill.rag.get_config", return_value=cfg), _embed_stub():
        answers = await rag.answer_questions(["Tell us about yourself"])

    assert answers["Tell us about yourself"] == "grounded answer"
    assert "A unique persona string for tests" in mock_cm.chat.await_args.args[0]

    mock_cm.chat = AsyncMock(return_value="{}")
    with patch("autofill.rag.get_config", return_value=cfg), _embed_stub():
        answers = await rag.answer_questions(["What is your expected compensation?"])

    assert answers["What is your expected compensation?"] == "9L INR per annum"


# --- learn() self-learning knowledge base ---


def _learnable_rag(tmp_path):
    rag = ScreenerRAG(store=MagicMock())
    rag.store.persona_question_exists = AsyncMock(return_value=False)
    rag.store.index_persona_chunks = AsyncMock()
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    return rag, persona_json


@pytest.mark.asyncio
async def test_learn_indexes_and_appends(tmp_path):
    rag, pj = _learnable_rag(tmp_path)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        _embed_stub(),
    ):
        ok = await rag.learn("How many years of experience do you have?", "2 years")

    assert ok is True
    rag.store.index_persona_chunks.assert_awaited_once()
    record = rag.store.index_persona_chunks.await_args.args[0][0]
    assert record["category"] == "general"
    assert record["question"] == "How many years of experience do you have?"
    assert record["answer"] == "2 years"

    data = json.loads(pj.read_text())
    assert data["version"] == 2
    assert data["answers"][-1] == {
        "category": "general",
        "question": "How many years of experience do you have?",
        "answer": "2 years",
    }


@pytest.mark.asyncio
async def test_learn_skips_exact_duplicate(tmp_path):
    rag, pj = _learnable_rag(tmp_path)
    rag.store.persona_question_exists = AsyncMock(return_value=True)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        _embed_stub(),
    ):
        ok = await rag.learn("Do you require visa sponsorship?", "Yes")

    assert ok is False
    rag.store.index_persona_chunks.assert_not_called()
    assert json.loads(pj.read_text())["version"] == 1


@pytest.mark.asyncio
async def test_learn_blank_answer_noop():
    rag = ScreenerRAG(store=MagicMock())
    with _embed_stub():
        assert await rag.learn("", "") is False
        assert await rag.learn("Some question?", "   ") is False
    rag.store.index_persona_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_learn_embed_failure_still_appends_persona_json(tmp_path):
    rag, pj = _learnable_rag(tmp_path)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        ok = await rag.learn("Do you require visa sponsorship?", "Yes", country="india")

    assert ok is True
    rag.store.index_persona_chunks.assert_not_called()
    data = json.loads(pj.read_text())
    assert data["answers"][-1]["category"] == "visa"
    assert data["answers"][-1]["answer"] == "Yes"
    assert data["answers"][-1]["country"] == "india"


@pytest.mark.asyncio
async def test_learn_scoped_without_country_refuses_to_persist(tmp_path):
    rag, pj = _learnable_rag(tmp_path)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        # A "No" for an unknown country must never become a global fact.
        ok = await rag.learn("Are you authorized to work in the country?", "No")

    assert ok is False
    data = json.loads(pj.read_text())
    assert len(data["answers"]) == 0


@pytest.mark.asyncio
async def test_learn_scoped_skips_same_answer_duplicate(tmp_path):
    rag = ScreenerRAG(store=MagicMock(), scoped_answers={})
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        ok1 = await rag.learn("Are you authorized to work in the country?", "No", country="india")
        # Same category + country + answer must not append again.
        ok2 = await rag.learn("Are you authorized to work in the country?", "No", country="india")

    assert ok1 is True
    assert ok2 is False
    data = json.loads(persona_json.read_text())
    assert len(data["answers"]) == 1


# --- deterministic residence / office-geography policies ---


@pytest.mark.asyncio
async def test_residence_question_answered_no_deterministically():
    """'Are you currently based in Europe?' is a residence fact: the candidate
    is based in India, so the answer is the 'No' option — never the LLM (which
    has answered 'Yes' to this in the past)."""
    rag, mock_cm = _rag()
    with _embed_stub():
        answers = await rag.answer_questions(
            [
                {
                    "question": "Are you currently based in Europe?",
                    "kind": "select",
                    "options": ["Yes", "No"],
                }
            ]
        )

    assert answers["Are you currently based in Europe?"] == "No"
    mock_cm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_residence_composite_with_willingness_option_stays_llm_driven():
    """A 'based in X or willing to relocate?' question with an option that
    expresses willingness separately is the LLM's to decide (the candidate is
    willing to relocate, so a blanket 'No' would be wrong)."""
    rag, mock_cm = _rag()
    mock_cm.chat = AsyncMock(
        return_value=json.dumps(
            {
                "Are you currently based in Munich, or willing to relocate?": (
                    "Not based in Munich, but open to relocating"
                )
            }
        )
    )
    with _embed_stub():
        answers = await rag.answer_questions(
            [
                {
                    "question": "Are you currently based in Munich, or willing to relocate?",
                    "kind": "select",
                    "options": ["Yes", "No", "Not based in Munich, but open to relocating"],
                }
            ]
        )

    assert (
        answers["Are you currently based in Munich, or willing to relocate?"]
        == "Not based in Munich, but open to relocating"
    )


@pytest.mark.asyncio
async def test_residence_composite_yes_no_only_answers_residence_facet():
    """A composite with plain Yes/No options cannot express willingness, so the
    residence facet answers deterministically ('No' — not based there)."""
    rag, mock_cm = _rag()
    with _embed_stub():
        answers = await rag.answer_questions(
            [
                {
                    "question": (
                        "Are you currently based in Europe (or willing to work "
                        "from a European timezone long-term)?"
                    ),
                    "kind": "select",
                    "options": ["Yes", "No"],
                }
            ]
        )

    assert (
        answers[
            (
                "Are you currently based in Europe (or willing to work from a "
                "European timezone long-term)?"
            )
        ]
        == "No"
    )
    mock_cm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_which_country_question_answers_home_country():
    """'Which country are you currently based in?' resolves to the home
    country, not the poisoned 'India +91' persona value."""
    rag, mock_cm = _rag()
    with _embed_stub():
        answers = await rag.answer_questions(["Which country are you currently based in?"])

    assert answers["Which country are you currently based in?"] == "India"
    mock_cm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_office_commute_question_beats_persona_leak():
    """The Netic regression: a persona row ('open to working in person in SF
    2-3 times/week' => Yes) must never answer 'able to work from our SF office
    five days per week?' — commuting to a foreign office is impossible."""
    store = _store(
        [
            {
                "category": "general",
                "question": (
                    "Are you open to working in person in our San Francisco or "
                    "New York office 2-3 times a week?"
                ),
                "answer": "Yes",
                "content": "...",
                "distance": 0.05,
            }
        ]
    )
    rag, mock_cm = _rag(store=store)
    with _embed_stub():
        answers = await rag.answer_questions(
            [
                {
                    "question": "Are you able to work from our SF office five days per week?",
                    "kind": "select",
                    "options": ["Yes", "No"],
                }
            ]
        )

    assert answers["Are you able to work from our SF office five days per week?"] == "No"
    mock_cm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_office_question_ignores_willingness_phrasing():
    """'Willing to work onsite' is intent, not a commute fact: it stays
    LLM-driven so the candidate's relocation willingness can answer Yes."""
    rag, mock_cm = _rag()
    mock_cm.chat = AsyncMock(
        return_value=json.dumps(
            {
                (
                    "We have an all-day, in-office policy. Are you willing to "
                    "work onsite, either in Paris or Tel Aviv?"
                ): "Yes"
            }
        )
    )
    with _embed_stub():
        answers = await rag.answer_questions(
            [
                {
                    "question": (
                        "We have an all-day, in-office policy. Are you willing "
                        "to work onsite, either in Paris or Tel Aviv?"
                    ),
                    "kind": "select",
                    "options": ["Yes", "No"],
                }
            ]
        )

    assert (
        answers[
            (
                "We have an all-day, in-office policy. Are you willing to work "
                "onsite, either in Paris or Tel Aviv?"
            )
        ]
        == "Yes"
    )


@pytest.mark.asyncio
async def test_position_statement_is_not_a_residence_question():
    """'This position is based in Bangkok...' describes the JOB, not the
    candidate's residence — the residence policy must not fire, and the
    learned exact answer ("Yes" — willing to relocate there) still wins."""
    rag, mock_cm = _rag()
    rag._exact_answers = {
        "this position is based in bangkok, thailand and requires full-time, "
        "on-site presence.": "Yes"
    }
    with _embed_stub():
        answers = await rag.answer_questions(
            [
                {
                    "question": (
                        "This position is based in Bangkok, Thailand and "
                        "requires full-time, on-site presence."
                    ),
                    "kind": "select",
                    "options": ["Yes", "No"],
                }
            ]
        )

    assert (
        answers[
            "This position is based in Bangkok, Thailand and requires full-time, on-site presence."
        ]
        == "Yes"
    )
    mock_cm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_learn_guard_refuses_negative_home_authorization(tmp_path):
    """The candidate is authorized in India; a learned 'No' for India is a
    self-contradictory fact and must be refused (this exact poisoning filled
    the persona with four 'No' rows)."""
    rag, _ = _rag()
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        ok = await rag.learn("Are you authorized to work in the country?", "No", country="india")

    assert ok is False
    data = json.loads(persona_json.read_text())
    assert len(data["answers"]) == 0
    assert rag._scoped_answers.get(("authorization", "india")) is None


@pytest.mark.asyncio
async def test_learn_guard_refuses_positive_foreign_authorization(tmp_path):
    """The candidate is only authorized in India; a learned 'Yes' for a
    foreign country must be refused."""
    rag, _ = _rag()
    persona_json = tmp_path / "persona.json"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        ok = await rag.learn(
            "Are you legally authorized to work in the United States?",
            "Yes",
            country="united states",
        )

    assert ok is False
    assert json.loads(persona_json.read_text())["answers"] == []


@pytest.mark.asyncio
async def test_learn_on_disk_dedupe_across_instances(tmp_path):
    """Dedupe must hold across process restarts: a fresh instance with an
    empty in-memory index still refuses to append an identical row."""
    persona_json = tmp_path / "persona.json"
    persona_txt = tmp_path / "persona.txt"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    persona_txt.write_text("Candidate Profile:\n- x: y\n\nFrom Resume:\n- stuff\n")

    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch("autofill.rag.PERSONA_TXT", persona_txt),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        first = ScreenerRAG(exact_answers={}, scoped_answers={}, store=None)
        ok1 = await first.learn("Do you require visa sponsorship?", "Yes", country="germany")
        second = ScreenerRAG(exact_answers={}, scoped_answers={}, store=None)
        ok2 = await second.learn("Do you require visa sponsorship?", "Yes", country="germany")

    assert ok1 is True
    assert ok2 is False
    data = json.loads(persona_json.read_text())
    assert len(data["answers"]) == 1
    assert data["version"] == 2


@pytest.mark.asyncio
async def test_learn_scoped_replace_updates_row_on_disk(tmp_path):
    """A changed scoped answer replaces the old row instead of appending a
    second one — no duplicate versions can accumulate."""
    persona_json = tmp_path / "persona.json"
    persona_txt = tmp_path / "persona.txt"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    persona_txt.write_text("Candidate Profile:\n- x: y\n\nFrom Resume:\n- stuff\n")

    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch("autofill.rag.PERSONA_TXT", persona_txt),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, store=None)
        ok1 = await rag.learn("Do you require visa sponsorship?", "Yes", country="germany")
        ok2 = await rag.learn(
            "Do you require visa sponsorship?",
            "Yes, I will require visa sponsorship",
            country="germany",
        )

    assert ok1 is True
    assert ok2 is True
    data = json.loads(persona_json.read_text())
    assert len(data["answers"]) == 1
    assert data["answers"][0]["answer"] == "Yes, I will require visa sponsorship"


@pytest.mark.asyncio
async def test_append_persona_txt_dedupes_across_calls(tmp_path):
    """Repeated learn() calls must never pile up duplicate lines in
    persona.txt, and a changed answer for the same question replaces the old
    line instead of adding a conflicting second one."""
    persona_json = tmp_path / "persona.json"
    persona_txt = tmp_path / "persona.txt"
    persona_json.write_text(json.dumps({"name": "T", "version": 1, "answers": []}))
    persona_txt.write_text("Candidate Profile:\n\nFrom Resume:\n- stuff\n")

    with (
        patch("autofill.rag.PERSONA_JSON", persona_json),
        patch("autofill.rag.PERSONA_TXT", persona_txt),
        patch.object(ScreenerRAG, "_embed", new=AsyncMock(return_value=None)),
    ):
        rag = ScreenerRAG(exact_answers={}, scoped_answers={}, store=None)
        await rag.learn("Do you require visa sponsorship?", "Yes", country="germany")
        await rag.learn("Do you require visa sponsorship?", "Yes", country="germany")
        await rag.learn(
            "Do you require visa sponsorship?",
            "Yes, I will require visa sponsorship",
            country="germany",
        )

    txt = persona_txt.read_text()
    assert txt.count("Do you require visa sponsorship?") == 1
    assert "Yes, I will require visa sponsorship" in txt
    assert "Yes, I will require visa sponsorship" in txt.split("From Resume:")[0]
