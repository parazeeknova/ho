import json

from autofill.profile import Profile


def test_profile_serialization():
    profile = Profile(
        first_name="Jane", last_name="Smith", email="jane.smith@example.com", phone="+9876543210"
    )
    data = profile.model_dump()

    assert data["firstName"] == "Jane"
    assert data["lastName"] == "Smith"
    assert data["email"] == "jane.smith@example.com"
    assert data["phone"] == "+9876543210"


def test_profile_special_characters_injection_safe():
    profile = Profile(
        first_name="O'Brien",
        last_name="D'Angelo `injected`",
        email="test+tag@example.com",
        phone="+1234567890",
        custom_answers={"sponsorship": "No 'sponsorship' needed"},
    )
    data = profile.model_dump()

    assert data["firstName"] == "O'Brien"
    assert data["lastName"] == "D'Angelo `injected`"
    assert data["customAnswers"]["sponsorship"] == "No 'sponsorship' needed"

    # Ensure JSON dump works without escaping issues
    dumped = json.dumps(data)
    loaded = json.loads(dumped)
    assert loaded["firstName"] == "O'Brien"


def test_profile_optional_fields_null():
    profile = Profile(
        first_name="Alice",
        last_name="Bob",
        linkedin=None,
        github=None,
        website=None,
        resume_path=None,
    )
    data = profile.model_dump()

    assert data["linkedin"] is None
    assert data["github"] is None
    assert data["website"] is None
    assert data["resumePath"] is None


def test_job_payload_schema():
    profile = Profile()
    payload = {
        "jobId": "test-job-123",
        "url": "https://boards.greenhouse.io/test/jobs/123",
        "profile": profile.model_dump(),
        "mode": "review",
    }

    serialized = json.dumps(payload)
    deserialized = json.loads(serialized)

    assert deserialized["jobId"] == "test-job-123"
    assert deserialized["url"] == "https://boards.greenhouse.io/test/jobs/123"
    assert deserialized["profile"]["firstName"] == "John"
