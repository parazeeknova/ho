"""Unit tests for persona-grounded ScreenerRAG retrieval."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autofill.rag import ASK_USER, ScreenerRAG


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
    persona_txt = tmp_path / "persona.txt"
    persona_txt.write_text("Candidate Profile:\n- x: y\n\nFrom Resume:\n- stuff\n")
    return rag, persona_json, persona_txt


@pytest.mark.asyncio
async def test_learn_indexes_and_appends(tmp_path):
    rag, pj, ptxt = _learnable_rag(tmp_path)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        patch("autofill.rag.PERSONA_TXT", ptxt),
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
    txt = ptxt.read_text()
    assert "- How many years of experience do you have?: 2 years" in txt
    assert txt.index("From Resume:") > txt.index("How many years")


@pytest.mark.asyncio
async def test_learn_skips_exact_duplicate(tmp_path):
    rag, pj, ptxt = _learnable_rag(tmp_path)
    rag.store.persona_question_exists = AsyncMock(return_value=True)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        patch("autofill.rag.PERSONA_TXT", ptxt),
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
    rag, pj, ptxt = _learnable_rag(tmp_path)
    with (
        patch("autofill.rag.PERSONA_JSON", pj),
        patch("autofill.rag.PERSONA_TXT", ptxt),
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
    rag, pj, _ = _learnable_rag(tmp_path)
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
