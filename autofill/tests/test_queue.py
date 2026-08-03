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
        await conn.execute("TRUNCATE autofill_queue, autofill_fills")

    yield instance

    # Clean up test table records after test
    async with instance._pool.acquire() as conn:
        await conn.execute("TRUNCATE autofill_queue, autofill_fills")
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
