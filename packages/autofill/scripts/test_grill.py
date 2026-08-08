"""Unit tests for the persona grill's answer normalization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from grill_persona import _normalize_answer  # noqa: E402


def test_ask_strips_trailing_punctuation_from_label():
    """LLM-generated questions ending in '.'/'?' must not render '. :'."""
    import re

    def strip_label(label):
        return re.sub(r"[.!?]+\s*$", "", label).strip()

    assert strip_label("What's a project you're most proud of.") == (
        "What's a project you're most proud of"
    )
    assert strip_label("How did you grow it?") == "How did you grow it"
    assert strip_label("proud of!") == "proud of"
    assert strip_label("no trailing punctuation") == "no trailing punctuation"


def test_normalize_question_punctuation():
    """Trailing sentence punctuation is stripped for clean dedupe."""
    import re

    def norm(q):
        return re.sub(r"[.!?]+\s*$", "", q).strip()

    assert norm("proud of.") == "proud of"
    assert norm("proud of?") == "proud of"
    assert norm("proud of") == "proud of"
    assert norm("  proud of.  ") == "proud of"


def test_normalize_yes_no_aliases():
    assert _normalize_answer("yes") == "Yes"
    assert _normalize_answer(" yep ") == "Yes"
    assert _normalize_answer("n") == "No"
    assert _normalize_answer("nah") == "No"
    assert _normalize_answer("N") == "No"


def test_normalize_prefer_not_family():
    for alias in ("prefer not to answer", "Prefer not to say", "decline", "skip", "pnta"):
        assert _normalize_answer(alias) == "Prefer not to answer"


def test_normalize_na_family():
    assert _normalize_answer("n/a") == "N/A"
    assert _normalize_answer("not applicable") == "N/A"
    assert _normalize_answer("none") == "N/A"


def test_normalize_collapses_whitespace():
    assert _normalize_answer("  remote   hybrid  ") == "remote hybrid"


def test_normalize_free_text_passes_through():
    assert _normalize_answer("60,000 USD") == "60,000 USD"
    assert _normalize_answer("Tomorrow") == "Tomorrow"
    assert _normalize_answer("36") == "36"


def test_normalize_empty():
    assert _normalize_answer("") == ""
    assert _normalize_answer("   ") == ""
