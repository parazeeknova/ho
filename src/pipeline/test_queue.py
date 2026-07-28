"""Tests for async job queue: producer-consumer with asyncio.Queue."""

import asyncio

import pytest

from src.pipeline.queue import JobPipeline, QueuedJob


class TestQueuedJob:
    def test_defaults(self) -> None:
        job = QueuedJob(markdown="x")
        assert job.url == ""
        assert job.title == ""
        assert job.snippet == ""

    def test_all_fields(self) -> None:
        job = QueuedJob(markdown="md", url="http://x.com", title="Dev", snippet="snip")
        assert job.markdown == "md"
        assert job.url == "http://x.com"
        assert job.title == "Dev"
        assert job.snippet == "snip"


class TestJobPipeline:
    @pytest.mark.asyncio
    async def test_push_and_pop(self) -> None:
        pipeline = JobPipeline()
        job = QueuedJob(markdown="test", url="u", title="t")
        await pipeline.push(job)

        popped = await pipeline.pop(timeout=0.5)
        assert popped is not None
        assert popped.markdown == "test"
        assert popped.url == "u"
        assert popped.title == "t"

    @pytest.mark.asyncio
    async def test_pop_timeout_returns_none(self) -> None:
        pipeline = JobPipeline()
        result = await pipeline.pop(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_signal_done(self) -> None:
        pipeline = JobPipeline()
        assert not pipeline.is_done
        pipeline.signal_done()
        assert pipeline.is_done

    @pytest.mark.asyncio
    async def test_pending_count(self) -> None:
        pipeline = JobPipeline()
        await pipeline.push(QueuedJob(markdown="a"))
        await pipeline.push(QueuedJob(markdown="b"))
        assert pipeline.pending == 2

    @pytest.mark.asyncio
    async def test_scraped_matched_counts(self) -> None:
        pipeline = JobPipeline()
        await pipeline.push(QueuedJob(markdown="a"))
        await pipeline.push(QueuedJob(markdown="b"))
        assert pipeline.scraped_count == 2
        assert pipeline.matched_count == 0

        await pipeline.pop(timeout=0.1)
        await pipeline.task_done()
        assert pipeline.matched_count == 1

    @pytest.mark.asyncio
    async def test_log_status(self) -> None:
        pipeline = JobPipeline()
        await pipeline.push(QueuedJob(markdown="a"))
        await pipeline.push(QueuedJob(markdown="b"))
        await pipeline.push(QueuedJob(markdown="c"))

        await pipeline.pop(timeout=0.1)
        await pipeline.task_done()

        status = pipeline.log_status()
        assert "2 pending" in status
        assert "3 scraped" in status
        assert "1 matched" in status

    @pytest.mark.asyncio
    async def test_consumer_drain_pattern(self) -> None:
        pipeline = JobPipeline()

        async def producer() -> None:
            for i in range(5):
                await pipeline.push(QueuedJob(markdown=f"item_{i}"))
            pipeline.signal_done()

        async def consumer() -> list[QueuedJob]:
            items: list[QueuedJob] = []
            while True:
                job = await pipeline.pop(timeout=0.5)
                if job is None:
                    if pipeline.is_done:
                        break
                    continue
                items.append(job)
                await pipeline.task_done()
            return items

        await asyncio.gather(producer(), asyncio.sleep(0))

        result = await consumer()
        assert len(result) == 5
        assert result[0].markdown == "item_0"
        assert pipeline.matched_count == 5
