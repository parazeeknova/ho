"""Unit tests for Phase 2 PostgreSQL queue database & worker IPC contracts."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from autofill.db import AutofillDB


@pytest_asyncio.fixture
async def db():
    """Fixture providing a fresh AutofillDB connected to Postgres.

    Uses a DEDICATED test database (``agent_memory_test``) so these tests
    never touch the live batch queue. The table is truncated before/after each
    test; a blanket DELETE here would otherwise wipe every in-flight job
    application in the shared ``agent_memory`` database.
    """
    from src.configuration import PostgresConfig, get_config

    live = get_config().postgres.dsn
    test_dsn = live.rsplit("/", 1)[0] + "/agent_memory_test"

    try:
        instance = await AutofillDB.create(config=PostgresConfig(dsn=test_dsn))
    except Exception as e:
        pytest.skip(f"PostgreSQL connection unavailable: {e}")

    # Clean up test table records before test
    async with instance._pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE autofill_queue, autofill_fills, autofill_site_knowledge, "
            "site_health, discord_question_mailbox"
        )

    yield instance

    # Clean up test table records after test
    async with instance._pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE autofill_queue, autofill_fills, autofill_site_knowledge, "
            "site_health, discord_question_mailbox"
        )
    await instance.close()


@pytest.mark.asyncio
async def test_enqueue_and_get_job(db: AutofillDB):
    job_id = await db.enqueue_job(
        apply_link="https://job-boards.greenhouse.io/test/jobs/123",
        role="Senior Backend Engineer",
        company="TestCorp",
        ats_platform="greenhouse",
        apply_mode="review",
    )
    assert job_id.startswith("job-")

    job = await db.get_job(job_id)
    assert job is not None
    assert job["apply_link"] == "https://job-boards.greenhouse.io/test/jobs/123"
    assert job["role"] == "Senior Backend Engineer"
    assert job["company"] == "TestCorp"
    assert job["status"] == "pending"


@pytest.mark.asyncio
async def test_claim_next_job_atomic_lock(db: AutofillDB):
    # Enqueue 1 test job
    job_id = await db.enqueue_job(
        apply_link="https://jobs.lever.co/test/456",
        role="Fullstack Engineer",
        company="LeverCorp",
        apply_mode="review",
    )

    # Concurrently attempt to claim the job from 2 async tasks
    job_1, job_2 = await asyncio.gather(
        db.claim_next_job(lease_seconds=600),
        db.claim_next_job(lease_seconds=600),
    )

    # Exactly one task should claim the job, the other receives None (due to FOR UPDATE SKIP LOCKED)
    claimed_jobs = [j for j in (job_1, job_2) if j is not None]
    assert len(claimed_jobs) == 1
    assert claimed_jobs[0]["job_id"] == job_id
    assert claimed_jobs[0]["status"] == "filling"


@pytest.mark.asyncio
async def test_update_job_status_and_payload(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://jobs.ashbyhq.com/test/789")

    # Claim job
    claimed = await db.claim_next_job(lease_seconds=600)
    assert claimed is not None

    # Update to awaiting_review with payload and screenshot
    payload = {"firstName": "Test", "lastName": "User"}
    screenshot = "/tmp/test.png"
    updated = await db.update_status(
        job_id,
        status="awaiting_review",
        filled_payload=payload,
        screenshot_path=screenshot,
    )
    assert updated is True

    # Verify updated record
    job = await db.get_job(job_id)
    assert job["status"] == "awaiting_review"
    assert job["filled_payload"] == payload
    assert job["screenshot_path"] == screenshot


@pytest.mark.asyncio
async def test_expired_lease_reclaim(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://job-boards.greenhouse.io/test/lease")

    # Claim job with 1 second lease
    claimed = await db.claim_next_job(lease_seconds=1)
    assert claimed is not None

    # Wait for lease to expire
    await asyncio.sleep(1.5)

    # Claim again — should reclaim the expired lease
    reclaimed = await db.claim_next_job(lease_seconds=600)
    assert reclaimed is not None
    assert reclaimed["job_id"] == job_id
    assert reclaimed["retries"] == 2


@pytest.mark.asyncio
async def test_enqueue_dedup_active_link(db: AutofillDB):
    link = "https://job-boards.greenhouse.io/test/dedup"
    job_id_1 = await db.enqueue_job(apply_link=link)
    job_id_2 = await db.enqueue_job(apply_link=link)
    assert job_id_2 == job_id_1


@pytest.mark.asyncio
async def test_mark_deferred_round_trip(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://jobs.lever.co/test/defer")

    updated = await db.mark_deferred(job_id, questions=["Q1?", "Q2?"], reason="needs user input")
    assert updated is True

    job = await db.get_job(job_id)
    assert job["status"] == "deferred"
    assert job["pending_questions"] == ["Q1?", "Q2?"]
    assert job["error"] == "needs user input"


@pytest.mark.asyncio
async def test_deferred_job_not_reclaimed(db: AutofillDB):
    deferred_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/deferred")
    await db.mark_deferred(deferred_id, questions=["Q?"])
    fresh_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/fresh")

    claimed = await db.claim_next_job(lease_seconds=600)
    assert claimed is not None
    assert claimed["job_id"] == fresh_id


@pytest.mark.asyncio
async def test_summary_query_and_mark(db: AutofillDB):
    d1 = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/s1")
    await db.mark_deferred(d1, questions=["Q1"])
    d2 = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/s2")
    await db.mark_deferred(d2, questions=["Q2"])

    pending = await db.get_pending_summary_jobs()
    assert {r["job_id"] for r in pending} == {d1, d2}
    assert pending[0]["pending_questions"] == ["Q2"]  # newest first

    updated = await db.mark_summary_sent([d1])
    assert updated == 1
    remaining = await db.get_pending_summary_jobs()
    assert [r["job_id"] for r in remaining] == [d2]


@pytest.mark.asyncio
async def test_clear_pending_questions(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/clear")
    await db.mark_deferred(job_id, questions=["Q?"])

    assert await db.clear_pending_questions(job_id) is True
    job = await db.get_job(job_id)
    assert job["pending_questions"] == []


@pytest.mark.asyncio
async def test_get_deferred_jobs(db: AutofillDB):
    job_id = await db.enqueue_job(
        apply_link="https://boards.greenhouse.io/test/list",
        role="Backend Engineer",
        company="ListCorp",
    )
    await db.mark_deferred(job_id, questions=["Q1?"])

    rows = await db.get_deferred_jobs()
    assert rows[0]["job_id"] == job_id
    assert rows[0]["role"] == "Backend Engineer"
    assert rows[0]["pending_questions"] == ["Q1?"]


@pytest.mark.asyncio
async def test_record_fill_round_trip(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/fill")

    assert (
        await db.record_fill(
            job_id,
            "Are you legally authorized to work in the United States?",
            "No",
            source="kb",
            options=["Yes", "No"],
        )
        is True
    )
    assert (
        await db.record_fill(
            job_id, "Why are you interested in this role?", "Grounded LLM answer", source="llm"
        )
        is True
    )
    assert (
        await db.record_fill(
            job_id, "Are you authorized to work in the country?", None, source="deferred"
        )
        is True
    )

    fills = await db.get_fills(job_id)
    assert len(fills) == 3
    assert fills[0]["question"] == "Are you legally authorized to work in the United States?"
    assert fills[0]["answer"] == "No"
    assert fills[0]["source"] == "kb"
    assert fills[0]["options"] == ["Yes", "No"]
    assert fills[2]["answer"] is None
    assert fills[2]["source"] == "deferred"


@pytest.mark.asyncio
async def test_record_fill_rejects_blank_question(db: AutofillDB):
    assert await db.record_fill("job-x", "", "No") is False
    assert await db.record_fill("", "Q?", "No") is False


@pytest.mark.asyncio
async def test_fills_scoped_per_job(db: AutofillDB):
    a = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/filla")
    b = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/fillb")
    await db.record_fill(a, "Q for A", "Ans A")
    await db.record_fill(b, "Q for B", "Ans B")

    assert [f["question"] for f in await db.get_fills(a)] == ["Q for A"]
    assert [f["question"] for f in await db.get_fills(b)] == ["Q for B"]


@pytest.mark.asyncio
async def test_fill_ttl_expires_and_purges(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/ttl")
    await db.record_fill(job_id, "Q1", "A1")
    await db.record_fill(job_id, "Q2", "A2")
    assert len(await db.get_fills(job_id)) == 2

    # Backdate Q1 past the 2-day TTL. The opportunistic purge inside get_fills
    # removes it; Q2 (still live) is returned.
    async with db._pool.acquire() as conn:
        await conn.execute(
            "UPDATE autofill_fills SET expires_at = NOW() - INTERVAL '1 second' "
            "WHERE question = 'Q1'"
        )
    assert [f["question"] for f in await db.get_fills(job_id)] == ["Q2"]

    # Backdate the remaining row; the standalone purge now has work to do.
    async with db._pool.acquire() as conn:
        await conn.execute(
            "UPDATE autofill_fills SET expires_at = NOW() - INTERVAL '1 second' "
            "WHERE question = 'Q2'"
        )
    assert await db.purge_expired_fills() >= 1
    assert len(await db.get_fills(job_id)) == 0


@pytest.mark.asyncio
async def test_update_status_sets_applied_at(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/applied")
    assert (await db.get_job(job_id))["applied_at"] is None

    ok = await db.update_status(job_id, status="submitted")
    assert ok
    row = await db.get_job(job_id)
    assert row["status"] == "submitted"
    assert row["applied_at"] is not None

    summary = await db.queue_summary()
    assert summary["applied"] == 1
    assert summary["open"] == 0


@pytest.mark.asyncio
async def test_update_status_failed_bumps_error_count(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/error")
    await db.update_status(job_id, status="failed", error="CAPTCHA_DETECTED")
    await db.update_status(job_id, status="failed", error="TIMEOUT")

    row = await db.get_job(job_id)
    assert row["error_count"] == 2
    assert row["last_error"] == "TIMEOUT"
    assert row["last_error_at"] is not None

    summary = await db.queue_summary()
    assert summary["errored"] == 1
    assert summary["failed"] == 1


@pytest.mark.asyncio
async def test_link_known_covers_terminal_rows(db: AutofillDB):
    link = "https://boards.greenhouse.io/test/known"
    assert await db.link_known(link) is False

    job_id = await db.enqueue_job(apply_link=link, source="radar")
    assert await db.link_known(link) is True

    # Terminal status still counts: bridge must never re-enqueue an applied job.
    await db.update_status(job_id, status="submitted")
    assert await db.link_known(link) is True


@pytest.mark.asyncio
async def test_queue_summary_counts(db: AutofillDB):
    a = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/sum-a")
    b = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/sum-b")
    c = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/sum-c")

    await db.update_status(a, status="submitted")
    await db.update_status(b, status="failed", error="boom")
    await db.update_status(c, status="deferred", error="needs user input")

    summary = await db.queue_summary()
    assert summary["applied"] == 1
    assert summary["errored"] == 1
    assert summary["deferred"] == 1
    assert summary["failed"] == 1


@pytest.mark.asyncio
async def test_mailbox_round_trip(db: AutofillDB):
    await db.heartbeat_poller("ingest")
    assert await db.poller_alive() is True

    await db.open_mailbox_question("q-1", "123", [111, 112], "Question?")
    await db.append_mailbox_message_ids("q-1", [113])
    assert await db.poll_mailbox_question("q-1") == ("pending", None)

    assert await db.answer_mailbox_message(113, "Yes") is True
    assert await db.poll_mailbox_question("q-1") == ("answered", "Yes")
    # An unrelated message must not match any pending question.
    assert await db.answer_mailbox_message(999, "stray") is False


@pytest.mark.asyncio
async def test_mailbox_skip_callback_routes(db: AutofillDB):
    await db.open_mailbox_question("q-2", "123", [201], "Pick one")
    assert await db.answer_mailbox_message(201, "skip") is True
    assert await db.poll_mailbox_question("q-2") == ("answered", "skip")


@pytest.mark.asyncio
async def test_poller_alive_staleness(db: AutofillDB):
    await db.heartbeat_poller("ingest")
    assert await db.poller_alive() is True
    async with db._pool.acquire() as conn:
        await conn.execute(
            "UPDATE discord_poller_state SET last_seen = NOW() - INTERVAL '60 seconds'"
        )
    assert await db.poller_alive() is False
    assert await db.poller_alive(max_age_seconds=120) is True


@pytest.mark.asyncio
async def test_close_mailbox_question(db: AutofillDB):
    await db.open_mailbox_question("q-3", "123", [301], "Done?")
    await db.close_mailbox_question("q-3", "timed_out")
    assert await db.poll_mailbox_question("q-3") == ("timed_out", None)
    # A late reply to a closed question must not resurrect it.
    assert await db.answer_mailbox_message(301, "late") is False


# ── site knowledge (procedural memory) ─────────────────────────────────


@pytest.mark.asyncio
async def test_site_knowledge_upsert_and_get(db: AutofillDB):
    assert await db.get_site_knowledge("jobs.ashbyhq.com", "form-a") is None
    await db.upsert_site_knowledge(
        "jobs.ashbyhq.com",
        "form-a",
        platform="ashby",
        selectors={"location": 'input[role="combobox"]'},
        flow="wizard",
        strategies={"location": "two-keystrokes-arrow-down"},
    )
    rec = await db.get_site_knowledge("jobs.ashbyhq.com", "form-a")
    assert rec is not None
    assert rec["platform"] == "ashby"
    assert rec["selectors"]["location"] == 'input[role="combobox"]'
    assert rec["flow"] == "wizard"
    assert rec["success_count"] == 1


@pytest.mark.asyncio
async def test_site_knowledge_success_increments(db: AutofillDB):
    await db.upsert_site_knowledge("x.com", "f", success=True)
    await db.upsert_site_knowledge("x.com", "f", success=True)
    rec = await db.get_site_knowledge("x.com", "f")
    assert rec["success_count"] == 2


@pytest.mark.asyncio
async def test_site_knowledge_failure_increments_without_overwriting_selectors(
    db: AutofillDB,
):
    await db.upsert_site_knowledge("x.com", "f", selectors={"name": "#name"}, flow="single")
    await db.upsert_site_knowledge("x.com", "f", success=False)
    rec = await db.get_site_knowledge("x.com", "f")
    assert rec["fail_count"] == 1
    # last-known-good selectors preserved on failure (drift detection)
    assert rec["selectors"]["name"] == "#name"


# ── site health + circuit breaker ──────────────────────────────────────


@pytest.mark.asyncio
async def test_site_health_success_resets_failures(db: AutofillDB):
    await db.record_site_failure("blocked-ats.com", "captcha")
    await db.record_site_failure("blocked-ats.com", "captcha")
    health = await db.site_health("blocked-ats.com")
    assert health is not None and health["fail_count"] == 2
    await db.record_site_success("blocked-ats.com")
    health = await db.site_health("blocked-ats.com")
    assert health is not None and health["fail_count"] == 0


@pytest.mark.asyncio
async def test_site_health_quarantines_after_threshold(db: AutofillDB, monkeypatch):
    monkeypatch.setenv("SITE_HEALTH_QUARANTINE", "2")
    await db.record_site_failure("bad.com", "selector_drift", cooldown_seconds=60)
    await db.record_site_failure("bad.com", "selector_drift", cooldown_seconds=60)
    health = await db.site_health("bad.com")
    assert health is not None and health["cooldown_until"] is not None
    assert await db.domain_quarantined("bad.com") is True


@pytest.mark.asyncio
async def test_site_health_not_quarantined_below_threshold(db: AutofillDB):
    await db.record_site_failure("ok.com", "network", cooldown_seconds=60)
    assert await db.domain_quarantined("ok.com") is False


@pytest.mark.asyncio
async def test_failure_label_taxonomy():
    from autofill.db import _failure_label

    assert _failure_label("selector not found") == "selector_drift"
    assert _failure_label("CAPTCHA_DETECTED") == "captcha"
    assert _failure_label("blocked by anti-bot") == "ban"
    assert _failure_label("network timeout") == "network"
    assert _failure_label("Runner infra failure (exit 127)") == "infra"
    assert _failure_label("something else") == "unknown"


@pytest.mark.asyncio
async def test_expired_status_is_terminal(db: AutofillDB):
    """An expired posting is terminal: a later status update must not overwrite it."""
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/expired")
    ok = await db.update_status(job_id, status="expired", error="expired posting (404)")
    assert ok
    row = await db.get_job(job_id)
    assert row["status"] == "expired"
    assert "expired posting" in row["last_error"]

    # A later non-terminal update is refused.
    await db.update_status(job_id, status="pending")
    assert (await db.get_job(job_id))["status"] == "expired"


@pytest.mark.asyncio
async def test_expired_counted_in_summary_and_not_open(db: AutofillDB):
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/expired2")
    await db.update_status(job_id, status="expired", error="gone")
    summary = await db.queue_summary()
    assert summary["expired"] == 1
    assert summary["open"] == 0


@pytest.mark.asyncio
async def test_expired_claim_never_picked(db: AutofillDB):
    """An expired job must never be re-claimed by the worker."""
    job_id = await db.enqueue_job(apply_link="https://boards.greenhouse.io/test/expired3")
    await db.update_status(job_id, status="expired", error="gone")
    # claim_next_job must not return it (only pending / expired-lease filling).
    claimed = await db.claim_next_job(lease_seconds=60)
    assert claimed is None or claimed["job_id"] != job_id


# ── active discord thread state ────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_and_read_active_thread(db: AutofillDB):
    assert await db.active_thread() is None
    await db.set_active_thread("12345")
    assert await db.active_thread() == "12345"
    # overwrite
    await db.set_active_thread("67890")
    assert await db.active_thread() == "67890"


@pytest.mark.asyncio
async def test_active_thread_stale(db: AutofillDB):
    await db.set_active_thread("999")
    async with db._pool.acquire() as conn:
        await conn.execute(
            "UPDATE discord_thread_state SET updated_at = NOW() - INTERVAL '2 hours'"
        )
    assert await db.active_thread(max_age_seconds=3600) is None


@pytest.mark.asyncio
async def test_clean_question_strips_jd_dump():
    from autofill.worker import AutofillWorker

    q = (
        "Expected Graduation Year\n"
        "Software Engineering- Internship (Fall 2026/Summer 2027)\n"
        "COMPANY OVERVIEW\n"
        "Deepgram is the leading platform..."
    )
    cleaned = AutofillWorker._clean_question(q)
    assert "Deepgram" not in cleaned
    assert "COMPANY OVERVIEW" not in cleaned
    assert "Expected Graduation Year" in cleaned
    assert len(cleaned) <= 140


@pytest.mark.asyncio
async def test_clean_question_caps_long_single_line():
    from autofill.worker import AutofillWorker

    long = "What is your expected graduation year?" + " extra words " * 30
    cleaned = AutofillWorker._clean_question(long)
    assert len(cleaned) <= 140
    assert cleaned.endswith("...")
