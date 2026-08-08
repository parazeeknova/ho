"""Unit tests for grill_persona identity helpers."""

import grill_persona


def test_norm_strips_scheme_and_www():
    assert grill_persona._norm("https://github.com/Tutankhaman") == "github.com/tutankhaman"
    assert grill_persona._norm("www.LinkedIn.com/in/aman-aziz") == "linkedin.com/in/aman-aziz"
    assert grill_persona._norm("+91 93159 78211") == "+91 93159 78211"


def test_no_mismatch_when_equal_after_normalization():
    saved = {"github": "https://github.com/tutankhaman"}
    resume = {"github": "github.com/tutankhaman"}
    assert grill_persona.identity_mismatches(saved, resume) == []


def test_mismatch_on_different_value():
    saved = {"github": "https://github.com/aman-aziz"}
    resume = {"github": "github.com/tutankhaman"}
    assert grill_persona.identity_mismatches(saved, resume) == [
        ("github", "https://github.com/aman-aziz", "github.com/tutankhaman")
    ]


def test_case_difference_is_not_a_mismatch():
    saved = {"github": "https://github.com/tutankhAman"}
    resume = {"github": "github.com/tutankhaman"}
    assert grill_persona.identity_mismatches(saved, resume) == []


def test_empty_saved_value_is_not_a_mismatch():
    saved = {"github": ""}
    resume = {"github": "github.com/tutankhaman"}
    assert grill_persona.identity_mismatches(saved, resume) == []


def test_empty_resume_extraction_yields_no_mismatches():
    saved = {"email": "amanaziz2020@gmail.com"}
    assert grill_persona.identity_mismatches(saved, {}) == []
