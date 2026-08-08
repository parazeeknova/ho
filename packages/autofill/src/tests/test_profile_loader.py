"""Unit tests for deterministic profile resolution (no LLM involvement)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autofill.src.screener.profile import build_profile

PERSONA_DATA = {
    "name": "Test Candidate",
    "identity": {
        "firstName": "Aman",
        "lastName": "Candidate",
        "email": "candidate@example.com",
        "phone": "+1 555 0100",
        "linkedin": "linkedin.com/in/test-candidate",
        "github": "github.com/test-user-github",
        "website": "example.com",
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
            "answer": "Bangalore, India",
        },
    ],
}


def _identity_field_resolver():
    return AsyncMock(side_effect=lambda store, field: PERSONA_DATA["identity"].get(field))


@pytest.mark.asyncio
async def test_lookup_identity_field_matches_identity_chunk():
    from autofill.src.screener.profile import _lookup_identity_field

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
        patch("autofill.src.screener.profile._load_persona_json", return_value=PERSONA_DATA),
        patch(
            "autofill.src.screener.profile._lookup_identity_field", new=_identity_field_resolver()
        ),
        patch(
            "autofill.src.screener.profile._lookup_resume_header", new=AsyncMock(return_value="")
        ),
    ):
        profile = await build_profile(MagicMock())

    assert profile.firstName == "Aman"
    assert profile.lastName == "Candidate"
    assert profile.email == "candidate@example.com"
    assert profile.phone == "+1 555 0100"
    assert profile.linkedin == "linkedin.com/in/test-candidate"
    assert profile.github == "github.com/test-user-github"
    assert profile.website == "example.com"


@pytest.mark.asyncio
async def test_build_profile_resume_header_regex_fallback():
    header = (
        "+1 555 0100 candidate@example.com "
        "linkedin.com/in/test-candidate github.com/test-user-github example.com"
    )
    data = dict(PERSONA_DATA)
    data["identity"] = {}
    with (
        patch("autofill.src.screener.profile._load_persona_json", return_value=data),
        patch(
            "autofill.src.screener.profile._lookup_identity_field", new=AsyncMock(return_value=None)
        ),
        patch(
            "autofill.src.screener.profile._lookup_resume_header",
            new=AsyncMock(return_value=header),
        ),
    ):
        profile = await build_profile(MagicMock())

    assert profile.email == "candidate@example.com"
    assert profile.phone == "+1 555 0100"
    assert profile.linkedin == "linkedin.com/in/test-candidate"
    assert profile.github == "github.com/test-user-github"
    assert profile.website == "example.com"


@pytest.mark.asyncio
async def test_build_profile_persona_json_fallback_without_store():
    with patch("autofill.src.screener.profile._load_persona_json", return_value=PERSONA_DATA):
        profile = await build_profile(store=None)

    assert profile.firstName == "Aman"
    assert profile.lastName == "Candidate"
    assert profile.email == "candidate@example.com"
    assert profile.github == "github.com/test-user-github"


@pytest.mark.asyncio
async def test_build_profile_custom_answers_from_persona():
    with patch("autofill.src.screener.profile._load_persona_json", return_value=PERSONA_DATA):
        profile = await build_profile(store=None)

    assert profile.customAnswers["Do you require visa sponsorship?"] == "Yes"
    assert profile.customAnswers["What is your current location?"] == "Bangalore, India"


@pytest.mark.asyncio
async def test_build_profile_deterministic_across_runs():
    with (
        patch("autofill.src.screener.profile._load_persona_json", return_value=PERSONA_DATA),
        patch(
            "autofill.src.screener.profile._lookup_identity_field", new=_identity_field_resolver()
        ),
        patch(
            "autofill.src.screener.profile._lookup_resume_header", new=AsyncMock(return_value="")
        ),
    ):
        first = await build_profile(MagicMock())
        second = await build_profile(MagicMock())

    assert first.model_dump() == second.model_dump()


def test_load_persona_resolves_from_worker_cwd():
    """Regression: _load_persona_json must resolve data/persona.json even when
    the CWD is packages/ingest (the worker's working dir). It previously
    computed the base as 3 parents up (-> packages/data, wrong), so the persona
    was never loaded and identity fell back to semantic-search garbage."""
    import json
    import os
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    # Point the loader at a deterministic fixture (never the user's real
    # persona.json) via CANDIDATE_PERSONA_FILE.
    fixture = Path(__file__).resolve().parent / "_fixture_persona.json"
    fixture.write_text(
        json.dumps(
            {
                "name": "Test Candidate",
                "identity": {"linkedin": "linkedin.com/in/test-candidate"},
            }
        )
    )
    code = (
        "from pathlib import Path; "
        "import sys; "
        "sys.path.insert(0, str(Path.cwd().parent)); "  # repo root
        "sys.path.insert(0, str(Path.cwd())); "
        "sys.path.insert(0, str(Path.cwd().parent / 'packages' / 'autofill')); "
        "from autofill.src.screener.profile import _load_persona_json; "
        "d = _load_persona_json(); "
        "print(d.get('identity', {}).get('linkedin', ''))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(repo / "packages" / "ingest") + os.pathsep + str(repo / "packages" / "autofill")
    )
    env["CANDIDATE_PERSONA_FILE"] = str(fixture)
    try:
        r = subprocess.run(
            ["uv", "run", "python", "-c", code],
            cwd=str(repo / "packages" / "ingest"),
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        out = (r.stdout or "").strip()
        assert "linkedin.com/in" in out, (
            f"persona not loaded from worker cwd: out={out!r} err={r.stderr[:300]}"
        )
    finally:
        fixture.unlink(missing_ok=True)
