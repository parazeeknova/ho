"""Unit tests for ATS platform classification."""

from autofill.ats import classify_ats


def test_classify_greenhouse():
    assert classify_ats("https://boards.greenhouse.io/neo4j/jobs/123") == "greenhouse"
    assert classify_ats("https://job-boards.greenhouse.io/acme/456") == "greenhouse"


def test_classify_ashby():
    assert classify_ats("https://jobs.ashbyhq.com/replit/abc") == "ashby"


def test_classify_lever():
    assert classify_ats("https://jobs.lever.co/acme/dev") == "lever"


def test_classify_workday():
    assert classify_ats("https://acme.wd12.myworkdayjobs.com/en-US/role") == "workday"


def test_classify_unknown_is_generic():
    assert classify_ats("https://some-company.com/careers/123") == "generic"


def test_classify_empty_is_generic():
    assert classify_ats("") == "generic"


def test_classify_case_insensitive():
    assert classify_ats("HTTPS://JOBS.ASHBYHQ.COM/REPLIT/X") == "ashby"
