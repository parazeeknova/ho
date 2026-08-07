"""Unit tests for the non-LLM pre-submit consistency check."""

import pytest

from autofill.consistency import _classify, _values_match, check_payload


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
