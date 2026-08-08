"""Unit tests for the non-LLM pre-submit consistency check."""

import pytest

from autofill.src.filling.consistency import _classify, _values_match, check_payload


class FakeProfile:
    def __init__(self, **kw) -> None:
        self.firstName = kw.get("firstName", "Harsh")
        self.lastName = kw.get("lastName", "Sahu")
        self.email = kw.get("email", "harshsahu049@gmail.com")
        self.phone = kw.get("phone", "+91 7000127001")
        self.location = kw.get("location", "Bhopal, Madhya Pradesh, India")
        self.linkedin = kw.get("linkedin", "https://linkedin.com/in/hashk")
        self.github = kw.get("github", "https://github.com/parazeeknova")
        self.website = kw.get("website", "https://przknv.cc")
        self.twitter = kw.get("twitter", "https://x.com/parazeeknova")
        self.school = kw.get("school", "")
        self.university = kw.get("university", "")


def test_classify_maps_labels_to_profile_fields():
    assert _classify("Location") == ("location", True)
    assert _classify("City") == ("location", True)
    assert _classify("Email") == ("email", True)
    assert _classify("First Name") == ("firstName", True)
    assert _classify("LinkedIn URL") == ("linkedin", True)
    assert _classify("University") == ("school", False)  # soft
    assert _classify("Something random") == (None, False)


def test_values_match_exact_and_partial():
    assert _values_match("Bhopal, Madhya Pradesh, India", "Bhopal, Madhya Pradesh, India") == 1.0
    # Substring containment (form truncated the value).
    assert _values_match("Bhopal, Madhya Pradesh, India", "Bhopal") >= 0.9
    # Emails must match exactly.
    assert _values_match("harshsahu049@gmail.com", "other@gmail.com") == 0.0
    assert _values_match("harshsahu049@gmail.com", "harshsahu049@gmail.com") == 1.0
    # Different location -> low score.
    assert _values_match("Bhopal, India", "United Kingdom") < 0.45


@pytest.mark.asyncio
async def test_check_payload_blocks_critical_mismatch():
    profile = FakeProfile()
    filled = {
        "Location": "United Kingdom",  # wrong — should block
        "Email": "harshsahu049@gmail.com",  # right
        "First Name": "Harsh",  # right
    }
    report = await check_payload(filled, profile, store=None, rag=None)
    assert report["ok"] is False
    assert any(m["label"] == "Location" for m in report["critical_mismatches"])


@pytest.mark.asyncio
async def test_check_payload_passes_when_consistent():
    profile = FakeProfile()
    filled = {
        "Location": "Bhopal, Madhya Pradesh, India",
        "Email": "harshsahu049@gmail.com",
        "First Name": "Harsh",
        "Last Name": "Sahu",
        "Phone Number": "+91 7000127001",
    }
    report = await check_payload(filled, profile, store=None, rag=None)
    assert report["ok"] is True
    assert report["critical_mismatches"] == []
    assert report["checked"] == 5


@pytest.mark.asyncio
async def test_check_payload_soft_mismatch_warns_but_passes():
    profile = FakeProfile(school="IIITDM Jabalpur")
    filled = {"University": "Some Other University"}
    report = await check_payload(filled, profile, store=None, rag=None)
    # School is a soft field — warns, does not block.
    assert report["ok"] is True
    assert any(w["label"] == "University" for w in report["soft_warnings"])


@pytest.mark.asyncio
async def test_check_payload_empty_fields_is_ok():
    report = await check_payload({}, FakeProfile(), store=None, rag=None)
    assert report["ok"] is True
    assert report["checked"] == 0


def test_match_persona_question_finds_rephrased_field():
    from autofill.src.filling.consistency import _match_persona_question

    answers = {
        "How soon can you start if selected?": "Tomorrow",
        "What are your salary expectations? (include currency)": "60000 USD",
        "Are you legally authorized to work in India?": "Yes",
    }
    matched = _match_persona_question("When could you start?", answers)
    assert matched is not None
    assert "start" in matched[0].lower()
    assert matched[1] >= 0.35


def test_match_persona_question_salary():
    from autofill.src.filling.consistency import _match_persona_question

    answers = {"What are your salary expectations? (include currency)": "60000 USD"}
    matched = _match_persona_question("Expected salary", answers)
    assert matched is not None
    assert "salary" in matched[0].lower()


@pytest.mark.asyncio
async def test_check_payload_verifies_new_field_via_persona():
    """A field NOT in the static registry (e.g. 'When could you start?') is
    matched to the persona's grilled question and verified."""
    profile = FakeProfile()
    profile.customAnswers = {
        "How soon can you start if selected?": "Tomorrow",
    }
    filled = {"When could you start?": "Tomorrow"}
    report = await check_payload(filled, profile, store=None, rag=None)
    assert report["ok"] is True
    assert report["checked"] == 1
    assert report["unchecked"] == 0


@pytest.mark.asyncio
async def test_check_payload_new_field_wrong_value_flags():
    """A new field whose committed value contradicts the persona answer is
    caught (soft warning for a non-critical label, not silent)."""
    profile = FakeProfile()
    profile.customAnswers = {
        "How soon can you start if selected?": "Tomorrow",
    }
    filled = {"When could you start?": "6 months"}
    report = await check_payload(filled, profile, store=None, rag=None)
    # Availability hint is in _CRITICAL_FIELD_HINT_RE -> blocks.
    assert report["ok"] is False
    assert any(m["label"] == "When could you start?" for m in report["critical_mismatches"])


@pytest.mark.asyncio
async def test_check_payload_unverifiable_field_counts_unchecked(monkeypatch):
    profile = FakeProfile()
    profile.customAnswers = {}
    import autofill.src.filling.consistency as consistency

    monkeypatch.setattr(consistency, "_live_persona_answers", lambda: {})
    filled = {"Why are you interested in this role?": "I like AI"}
    report = await check_payload(filled, profile, store=None, rag=None)
    assert report["ok"] is True
    assert report["unchecked"] == 1
    assert report["rag_flags"] == []  # no store -> no rag check


@pytest.mark.asyncio
async def test_critical_mismatch_expected_is_correctable():
    """A critical mismatch must expose the 'expected' persona value so the
    worker can auto-correct the field before submission."""
    profile = FakeProfile()
    filled = {"Location": "United Kingdom", "Email": "harshsahu049@gmail.com"}
    report = await check_payload(filled, profile, store=None, rag=None)
    assert report["ok"] is False
    loc = next(m for m in report["critical_mismatches"] if m["label"] == "Location")
    assert loc["expected"] == "Bhopal, Madhya Pradesh, India"
    # The worker turns this into a correction map: {label: expected}.
    corrections = {m["label"]: m["expected"] for m in report["critical_mismatches"]}
    assert corrections["Location"] == "Bhopal, Madhya Pradesh, India"


@pytest.mark.asyncio
async def test_check_payload_enforces_live_persona_answers(monkeypatch):
    """A question learned mid-run (after the profile snapshot) must be enforced
    by the pre-submit gate even though the profile.customAnswers is stale."""
    profile = FakeProfile()
    profile.customAnswers = {}  # stale: built before the mid-run learn

    # Simulate a question the user answered mid-run and persisted to persona.json.
    live = {
        "Are you legally authorized to work in Germany?": "No",
        "Expected salary": "₹18,00,000",
    }
    import autofill.src.filling.consistency as consistency

    monkeypatch.setattr(consistency, "_live_persona_answers", lambda: live)

    # The form filled "Yes" for work authorization in Germany — must be a
    # critical mismatch (label touches authorization).
    filled = {
        "Are you legally authorized to work in Germany?": "Yes",
        "Expected salary": "₹18,00,000",
    }
    report = await check_payload(filled, profile, store=None, rag=None)
    assert report["ok"] is False
    auth = next((m for m in report["critical_mismatches"] if "authorized" in m["label"]), None)
    assert auth is not None
    assert auth["expected"] == "No"

    # A matching answer passes.
    report2 = await check_payload(
        {"Are you legally authorized to work in Germany?": "No"}, profile, store=None, rag=None
    )
    assert report2["ok"] is True


def test_match_option_gender_synonyms():
    """Persona gender answers (Male/Female) must map to the option labels
    ATS boards actually use (Man/Woman) so a required DEI question resolves
    instead of declining and blocking submission."""
    from autofill.src.screener.resolve import match_option

    opts = ["Man", "Non-binary", "Woman", "I prefer to self-describe", "I don't wish to answer"]
    assert match_option("Male", opts) == "Man"
    assert match_option("Female", opts) == "Woman"
    assert match_option("Non-binary", opts) == "Non-binary"
    assert match_option("Nonbinary", opts) == "Non-binary"
