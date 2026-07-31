"""Unit tests for Phase 2 PostgreSQL queue database & worker IPC contracts."""

from __future__ import annotations

import asyncio
import pytest
import pytest_asyncio
import asyncpg
from autofill.db import AutofillDB


@pytest_asyncio.fixture
async def db():
    """Fixture providing a fresh AutofillDB connected to Postgres."""
    try:
        instance = await AutofillDB.create()
    except Exception as e:
        pytest.skip(f"PostgreSQL connection unavailable: {e}")

    # Clean up test table records before test
    async with instance._pool.acquire() as conn:
        await conn.execute("DELETE FROM autofill_queue WHERE job_id LIKE 'test-job-%'")

    yield instance

    # Clean up test table records after test
    async with instance._pool.acquire() as conn:
        await conn.execute("DELETE FROM autofill_queue WHERE job_id LIKE 'test-job-%'")
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
