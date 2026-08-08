"""Unit tests for the anti-fraud worker helpers (proxy template / resume copy)."""

import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autofill.src.core.worker import (
    _VOICE_SEEDS,
    AutofillWorker,
    _new_session_id,
    _per_job_proxy,
    _per_job_resume,
    _pick_voice,
    _start_proxy_relay,
)


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOFILL_PROXY", raising=False)
    monkeypatch.delenv("AUTOFILL_PROXY_TEMPLATE", raising=False)


def test_session_id_is_hex_and_unique() -> None:
    ids = {_new_session_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 16 for i in ids)


def test_proxy_template_substitutes_sid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "AUTOFILL_PROXY_TEMPLATE",
        "http://user-country-in-session-{SID}:pass@geo.iproyal.com:12321",
    )
    sid = _new_session_id()
    url = _per_job_proxy(sid)
    assert url is not None
    assert "{SID}" not in url
    assert sid in url


def test_proxy_template_wins_over_static(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOFILL_PROXY", "socks5://127.0.0.1:9050")
    monkeypatch.setenv("AUTOFILL_PROXY_TEMPLATE", "http://{SID}@geo:1")
    assert "geo" in (_per_job_proxy("abc") or "")


def test_static_proxy_returned_without_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOFILL_PROXY", "socks5://127.0.0.1:9050")
    assert _per_job_proxy("abc") == "socks5://127.0.0.1:9050"


def test_no_proxy_when_unset() -> None:
    assert _per_job_proxy("abc") is None


@pytest.mark.asyncio
async def test_start_proxy_relay_returns_credfree_local_url() -> None:
    relay = await _start_proxy_relay(
        "http://user-country-in-session-sid123:secret@geo.iproyal.com:12321"
    )
    assert relay is not None
    try:
        url = relay.local_url
        assert url.startswith("http://127.0.0.1:")
        assert "secret" not in url and "user-country" not in url
    finally:
        await relay.stop()


@pytest.mark.asyncio
async def test_start_proxy_relay_invalid_url_returns_none() -> None:
    relay = await _start_proxy_relay("http://user:pass@")
    assert relay is None


def test_voice_seed_from_pool() -> None:
    for _ in range(50):
        assert _pick_voice() in _VOICE_SEEDS


def test_profile_pool_never_reallocates_same_dir_concurrently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    import asyncio
    from unittest.mock import MagicMock

    monkeypatch.setenv("AUTOFILL_PROFILE_POOL_SIZE", "3")
    worker = AutofillWorker(MagicMock(), max_concurrent=1)

    async def run() -> None:
        # Allocate three slots; they must be distinct (no concurrent reuse).
        p1 = await worker._acquire_profile()
        p2 = await worker._acquire_profile()
        p3 = await worker._acquire_profile()
        # Pool is exhausted.
        assert await worker._acquire_profile() is None
        # Releasing one frees it for reuse.
        worker._release_profile(p1)
        p4 = await worker._acquire_profile()
        assert p4 == p1
        worker._release_profile(p2)
        worker._release_profile(p3)
        worker._release_profile(p4)

    asyncio.run(run())


def test_per_job_resume_copies_to_name_named_file(tmp_path: pytest.TempPathFactory) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4 fake")
    out = _per_job_resume(str(resume), first_name="Aman", last_name="Aziz", job_id="job-abc12345")
    assert out is not None
    dest = pathlib.Path(out)
    assert dest.exists()
    assert dest.name == "Aman_Aziz_Resume.pdf"
    assert dest.read_bytes() == b"%PDF-1.4 fake"
    # Idempotent: second call returns the same path.
    assert (
        _per_job_resume(str(resume), first_name="Aman", last_name="Aziz", job_id="job-abc12345")
        == out
    )


def test_per_job_resume_basename_constant_across_jobs(tmp_path: pytest.TempPathFactory) -> None:
    """The uploaded basename is <First>_<Last>_Resume.pdf for every job; the
    per-job subdirectory (not the basename) isolates concurrent jobs."""
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"%PDF-1.4 fake")
    a = _per_job_resume(str(resume), first_name="Aman", last_name="Aziz", job_id="job-aaa")
    b = _per_job_resume(str(resume), first_name="Aman", last_name="Aziz", job_id="job-bbb")
    assert a and b
    assert pathlib.Path(a).name == pathlib.Path(b).name == "Aman_Aziz_Resume.pdf"
    assert pathlib.Path(a).parent != pathlib.Path(b).parent


def test_per_job_resume_falls_back_to_stem_without_names(
    tmp_path: pytest.TempPathFactory,
) -> None:
    resume = tmp_path / "cv.pdf"
    resume.write_bytes(b"%PDF-1.4 fake")
    out = _per_job_resume(str(resume))
    assert out is not None
    assert pathlib.Path(out).name == "cv_Resume.pdf"


def test_per_job_resume_none_when_no_resume() -> None:
    assert _per_job_resume(None) is None


def test_per_job_resume_missing_source_passthrough(tmp_path: pytest.TempPathFactory) -> None:
    missing = str(tmp_path / "nope.pdf")
    assert _per_job_resume(missing) == missing


def test_format_digest_accepts_custom_title() -> None:
    text = AutofillWorker._format_digest(
        [
            {
                "job_id": "job-1",
                "company": "Acme",
                "role": "Backend Engineer",
                "apply_link": "https://boards.greenhouse.io/acme/123",
                "pending_questions": ["Are you authorized to work in the country?"],
            }
        ],
        title="🏁 Overnight run finished — deferred jobs",
    )
    assert "Overnight run finished" in text


def test_format_digest_is_markdown_not_html() -> None:
    """Discord renders Markdown, not Telegram HTML — no <b>/<a>/<code> tags."""
    text = AutofillWorker._format_digest(
        [
            {
                "job_id": "job-1",
                "company": "Acme",
                "role": "Backend Engineer",
                "apply_link": "https://boards.greenhouse.io/acme/123",
                "pending_questions": ["Are you authorized to work in the country?"],
            }
        ],
        title="🏁 Overnight run finished — deferred jobs",
    )
    assert "Overnight run finished" in text
    # No Telegram HTML
    assert "<b>" not in text
    assert "<a href" not in text
    assert "<code>" not in text
    assert "&lt;" not in text
    # Discord Markdown
    assert "**1. Acme** — Backend Engineer" in text
    assert "[Open posting →](https://boards.greenhouse.io/acme/123)" in text
    assert "`python -m autofill.src.filling.resume <job_id>`" in text
    assert "• Are you authorized to work in the country?" in text
    assert "Acme" in text
    assert "https://boards.greenhouse.io/acme/123" in text


@pytest.mark.asyncio
async def test_send_end_of_run_summary_lists_all_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERNIGHT_LOOP", "true")
    db = MagicMock()
    db.get_deferred_jobs = AsyncMock(
        return_value=[
            {
                "job_id": "job-1",
                "company": "Acme",
                "role": "Backend Engineer",
                "apply_link": "https://boards.greenhouse.io/acme/1",
                "pending_questions": ["Q1"],
            },
            {
                "job_id": "job-2",
                "company": "Globex",
                "role": "Frontend Engineer",
                "apply_link": "https://boards.lever.co/globex/2",
                "pending_questions": ["Q2", "Q3"],
            },
        ]
    )
    bridge = MagicMock()
    bridge.is_configured = True
    bridge.send = AsyncMock(return_value=True)

    worker = AutofillWorker(db, max_concurrent=4)
    with patch("autofill.src.core.worker.DiscordQuestionBridge", return_value=bridge):
        await worker.send_end_of_run_summary()

    bridge.send.assert_called_once()
    sent = bridge.send.await_args.args[0]
    assert "Acme" in sent and "Globex" in sent
    assert "https://boards.greenhouse.io/acme/1" in sent
    assert "https://boards.lever.co/globex/2" in sent
    assert "Overnight run finished" in sent


@pytest.mark.asyncio
async def test_send_end_of_run_summary_skipped_in_day_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OVERNIGHT_LOOP", raising=False)
    db = MagicMock()
    db.get_deferred_jobs = AsyncMock(return_value=[{"job_id": "job-1", "pending_questions": []}])
    worker = AutofillWorker(db, max_concurrent=4)
    await worker.send_end_of_run_summary()
    db.get_deferred_jobs.assert_not_called()


@pytest.mark.asyncio
async def test_send_end_of_run_summary_no_jobs_no_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERNIGHT_LOOP", "true")
    db = MagicMock()
    db.get_deferred_jobs = AsyncMock(return_value=[])
    bridge = MagicMock()
    worker = AutofillWorker(db, max_concurrent=4)
    with patch("autofill.src.core.worker.DiscordQuestionBridge", return_value=bridge):
        await worker.send_end_of_run_summary()
    bridge.send.assert_not_called()


@pytest.mark.asyncio
async def test_record_fill_helper_is_best_effort() -> None:
    db = MagicMock()
    db.record_fill = AsyncMock(side_effect=RuntimeError("db down"))
    worker = AutofillWorker(db, max_concurrent=4)
    # Must not raise even when the store fails.
    await worker._record_fill("job-1", "Q?", "No", source="kb")
    db.record_fill.assert_called_once()


def test_node_dir_points_to_ts_package() -> None:
    """The autofill Node runner lives at the repo root's packages/node.

    Regression: node_dir used ``dirname(__file__)/node`` which resolved to
    ``autofill/autofill/node`` (no runner.ts), so every fill failed with
    "Runner exited with code 127". The package now lives at ``packages/node``.
    """
    import os

    import autofill.src.core.worker as w

    node_dir = w._runner_dir()
    assert node_dir.endswith("packages/node")
    assert os.path.isfile(os.path.join(node_dir, "runner.ts"))
    assert os.path.isfile(os.path.join(node_dir, "package.json"))


def test_profiles_dir_points_to_ts_package() -> None:
    import autofill.src.core.worker as w

    base = w._NODE_DIR / "artifacts" / "profiles"
    assert str(base).endswith("packages/node/artifacts/profiles")
    assert base.exists()


def test_normalize_batch_specs_coerces_and_drops():
    from autofill.src.core.worker import _normalize_batch_specs

    out = _normalize_batch_specs(
        [
            {"question": "  Are you authorized?  ", "kind": "select", "options": ["Yes", "No"]},
            {"question": "Radio?", "kind": "radio", "options": ["A", "B"]},
            {"question": "Check?", "kind": "checkbox", "options": ["X", "Y"]},
            {"question": "", "kind": "text"},
            "not-a-dict",
        ]
    )
    assert len(out) == 3
    assert out[0]["question"] == "Are you authorized?"
    assert out[0]["kind"] == "select"
    assert out[1]["kind"] == "select"  # radio coerced
    assert out[2]["kind"] == "multi"  # checkbox coerced
    assert out[0]["required"] is True  # default


def test_normalize_batch_specs_required_flag():
    from autofill.src.core.worker import _normalize_batch_specs

    out = _normalize_batch_specs([{"question": "Q", "required": False}])
    assert out[0]["required"] is False
