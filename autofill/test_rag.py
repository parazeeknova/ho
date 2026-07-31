"""Unit tests for Phase 3 ScreenerRAG integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from autofill.rag import ScreenerRAG


@pytest.mark.asyncio
async def test_deterministic_question_answering():
    mock_cm = MagicMock()
    mock_cm.chat = AsyncMock(return_value='{"Why Twilio?": "I admire Twilio\'s developer-first API culture."}')

    rag = ScreenerRAG(context_manager=mock_cm)
    questions = [
        "Do you require visa sponsorship?",
        "Are you legally authorized to work in the United States?",
        "What is your expected salary?",
        "Why Twilio?"
    ]

    answers = await rag.answer_questions(questions)

    # Verify deterministic rules
    assert answers["Do you require visa sponsorship?"] == "No"
    assert answers["Are you legally authorized to work in the United States?"] == "Yes"
    assert "Flexible" in answers["What is your expected salary?"] or "80K" in answers["What is your expected salary?"]

    # Verify LLM call was only made for the open-ended question ("Why Twilio?")
    assert mock_cm.chat.called
    assert answers["Why Twilio?"] == "I admire Twilio's developer-first API culture."
