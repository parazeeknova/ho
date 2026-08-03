"""Unit tests for deterministic profile resolution (no LLM involvement)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autofill.profile import build_profile

PERSONA_DATA = {
    "name": "Aman Aziz",
    "identity": {
        "firstName": "Aman",
        "lastName": "Aziz",
        "email": "amanaziz2020@gmail.com",
        "phone": "+91 93159 78211",
        "linkedin": "linkedin.com/in/aman-aziz",
        "github": "github.com/tutankhaman",
        "website": "aamn.dev/tldr",
    },
    "answers": [
        {
            "category": "visa_sponsorship",
            "question": "Do you require visa sponsorship?",
            "answer": "Yes",
        },
        {
            "category": "current_location",
            "question": "What is your current location?",
            "answer": "Delhi, India",
        },
    ],
}


def _identity_field_resolver():
    return AsyncMock(side_effect=lambda store, field: PERSONA_DATA["identity"].get(field))


@pytest.mark.asyncio
async def test_lookup_identity_field_matches_identity_chunk():
    from autofill.profile import _lookup_identity_field

    store = MagicMock()
    store.search_similar_persona = AsyncMock(
        return_value=[
            {"category": "general", "question": "?", "answer": "wrong", "distance": 0.1},
            {"category": "identity", "question": "?", "answer": "Aman", "distance": 0.15},
            {"category": "identity", "question": "?", "answer": "too far", "distance": 0.9},
        ]
    )
    assert await _lookup_identity_field(store, "firstName") == "Aman"


@pytest.mark.asyncio
async def test_build_profile_resolves_identity_from_persona_store():
    with (
        patch("autofill.profile._load_persona_json", return_value=PERSONA_DATA),
        patch("autofill.profile._lookup_identity_field", new=_identity_field_resolver()),
        patch("autofill.profile._lookup_resume_header", new=AsyncMock(return_value="")),
    ):
        profile = await build_profile(MagicMock())

    assert profile.firstName == "Aman"
    assert profile.lastName == "Aziz"
    assert profile.email == "amanaziz2020@gmail.com"
    assert profile.phone == "+91 93159 78211"
    assert profile.linkedin == "linkedin.com/in/aman-aziz"
    assert profile.github == "github.com/tutankhaman"
    assert profile.website == "aamn.dev/tldr"


@pytest.mark.asyncio
async def test_build_profile_resume_header_regex_fallback():
    header = (
        "+91 93159 78211 amanaziz2020@gmail.com "
        "linkedin.com/in/aman-aziz github.com/tutankhaman aamn.dev/tldr"
    )
    data = dict(PERSONA_DATA)
    data["identity"] = {}
    with (
        patch("autofill.profile._load_persona_json", return_value=data),
        patch("autofill.profile._lookup_identity_field", new=AsyncMock(return_value=None)),
        patch("autofill.profile._lookup_resume_header", new=AsyncMock(return_value=header)),
    ):
        profile = await build_profile(MagicMock())

    assert profile.email == "amanaziz2020@gmail.com"
    assert profile.phone == "+91 93159 78211"
    assert profile.linkedin == "linkedin.com/in/aman-aziz"
    assert profile.github == "github.com/tutankhaman"
    assert profile.website == "aamn.dev/tldr"


@pytest.mark.asyncio
async def test_build_profile_persona_json_fallback_without_store():
    with patch("autofill.profile._load_persona_json", return_value=PERSONA_DATA):
        profile = await build_profile(store=None)

    assert profile.firstName == "Aman"
    assert profile.lastName == "Aziz"
    assert profile.email == "amanaziz2020@gmail.com"
    assert profile.github == "github.com/tutankhaman"


@pytest.mark.asyncio
async def test_build_profile_custom_answers_from_persona():
    with patch("autofill.profile._load_persona_json", return_value=PERSONA_DATA):
        profile = await build_profile(store=None)

    assert profile.customAnswers["Do you require visa sponsorship?"] == "Yes"
    assert profile.customAnswers["What is your current location?"] == "Delhi, India"


@pytest.mark.asyncio
async def test_build_profile_deterministic_across_runs():
    with (
        patch("autofill.profile._load_persona_json", return_value=PERSONA_DATA),
        patch("autofill.profile._lookup_identity_field", new=_identity_field_resolver()),
        patch("autofill.profile._lookup_resume_header", new=AsyncMock(return_value="")),
    ):
        first = await build_profile(MagicMock())
        second = await build_profile(MagicMock())

    assert first.model_dump() == second.model_dump()
