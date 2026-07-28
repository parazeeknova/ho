"""Tests for job queue: Redis-backed producer-consumer."""

from unittest.mock import MagicMock, patch

from src.pipeline.queue import JobPipeline, QueuedJob


class TestQueuedJob:
    def test_to_from_json_roundtrip(self) -> None:
        job = QueuedJob(markdown="test md", url="http://example.com", title="Dev")
        json_str = job.to_json()
        parsed = QueuedJob.from_json(json_str)
        assert parsed.markdown == "test md"
        assert parsed.url == "http://example.com"
        assert parsed.title == "Dev"

    def test_from_json_minimal(self) -> None:
        parsed = QueuedJob.from_json('{"markdown": "hello"}')
        assert parsed.markdown == "hello"
        assert parsed.url == ""
        assert parsed.title == ""

    def test_defaults(self) -> None:
        job = QueuedJob(markdown="x")
        assert job.url == ""
        assert job.title == ""
        assert job.snippet == ""


class TestJobPipeline:
    @patch("src.pipeline.queue.redis")
    def test_push_and_pop(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r
        mock_r.blpop.return_value = (b"key", b'{"markdown":"test","url":"u","title":"t"}')

        pipeline = JobPipeline()
        job = QueuedJob(markdown="test", url="u", title="t")
        pipeline.push(job)
        mock_r.rpush.assert_called_once()
        mock_r.hincrby.assert_called_with("ho:stats", "scraped", 1)

        popped = pipeline.pop()
        assert popped is not None
        assert popped.markdown == "test"

    @patch("src.pipeline.queue.redis")
    def test_pop_timeout_returns_none(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r
        mock_r.blpop.return_value = None

        pipeline = JobPipeline()
        result = pipeline.pop(timeout=1)
        assert result is None

    @patch("src.pipeline.queue.redis")
    def test_signal_done(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r
        mock_r.exists.return_value = False  # before signal
        mock_r.exists.side_effect = [False, True]  # after signal returns True

        pipeline = JobPipeline()
        assert not pipeline.is_done
        pipeline.signal_done()
        mock_r.set.assert_called_with("ho:stop", "1")

    @patch("src.pipeline.queue.redis")
    def test_pending_count(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r
        mock_r.llen.return_value = 5

        pipeline = JobPipeline()
        assert pipeline.pending == 5

    @patch("src.pipeline.queue.redis")
    def test_scraped_matched_counts(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r
        mock_r.hget.side_effect = [b"10", b"3"]

        pipeline = JobPipeline()
        assert pipeline.scraped_count == 10
        assert pipeline.matched_count == 3

    @patch("src.pipeline.queue.redis")
    def test_task_done(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r

        pipeline = JobPipeline()
        pipeline.task_done()
        mock_r.hincrby.assert_called_with("ho:stats", "matched", 1)

    @patch("src.pipeline.queue.redis")
    def test_log_status(self, mock_redis_module: MagicMock) -> None:
        mock_r = MagicMock()
        mock_redis_module.from_url.return_value = mock_r
        mock_r.llen.return_value = 3
        mock_r.hget.side_effect = [b"12", b"8"]

        pipeline = JobPipeline()
        status = pipeline.log_status()
        assert "3 pending" in status
        assert "12 scraped" in status
        assert "8 matched" in status
