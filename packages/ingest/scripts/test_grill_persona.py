"""Unit tests for grill_persona identity helpers."""

import grill_persona


def test_norm_strips_scheme_and_www():
    assert grill_persona._norm("https://github.com/test-user") == "github.com/test-user"
    assert (
        grill_persona._norm("www.LinkedIn.com/in/test-candidate")
        == "linkedin.com/in/test-candidate"
    )
    assert grill_persona._norm("+91 93159 78211") == "+91 93159 78211"


def test_no_mismatch_when_equal_after_normalization():
    saved = {"github": "https://github.com/test-user"}
    resume = {"github": "github.com/test-user"}
    assert grill_persona.identity_mismatches(saved, resume) == []


def test_mismatch_on_different_value():
    saved = {"github": "https://github.com/test-candidate"}
    resume = {"github": "github.com/test-user"}
    assert grill_persona.identity_mismatches(saved, resume) == [
        ("github", "https://github.com/test-candidate", "github.com/test-user")
    ]


def test_case_difference_is_not_a_mismatch():
    saved = {"github": "https://github.com/test-user"}
    resume = {"github": "github.com/test-user"}
    assert grill_persona.identity_mismatches(saved, resume) == []


def test_empty_saved_value_is_not_a_mismatch():
    saved = {"github": ""}
    resume = {"github": "github.com/test-user"}
    assert grill_persona.identity_mismatches(saved, resume) == []


def test_empty_resume_extraction_yields_no_mismatches():
    saved = {"email": "candidate@example.com"}
    assert grill_persona.identity_mismatches(saved, {}) == []
