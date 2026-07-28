"""Redis-backed job queue for producer-consumer decoupling."""

import json
import threading
from dataclasses import dataclass

import redis

QUEUE_KEY = "ho:job_queue"
STATS_KEY = "ho:stats"
STOP_KEY = "ho:stop"


@dataclass
class QueuedJob:
    markdown: str
    url: str = ""
    title: str = ""
    snippet: str = ""

    def to_json(self) -> str:
        return json.dumps({"markdown": self.markdown, "url": self.url, "title": self.title})

    @classmethod
    def from_json(cls, data: str) -> QueuedJob:
        d = json.loads(data)
        return cls(markdown=d["markdown"], url=d.get("url", ""), title=d.get("title", ""))


class JobPipeline:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = redis.from_url(redis_url)
        self.done = threading.Event()
        self._flush()

    def _flush(self) -> None:
        self.r.delete(QUEUE_KEY, STATS_KEY, STOP_KEY)

    def push(self, job: QueuedJob) -> None:
        self.r.rpush(QUEUE_KEY, job.to_json())
        self.r.hincrby(STATS_KEY, "scraped", 1)

    def signal_done(self) -> None:
        self.r.set(STOP_KEY, "1")
        self.done.set()

    @property
    def is_done(self) -> bool:
        return bool(self.r.exists(STOP_KEY))

    def pop(self, timeout: int = 2) -> QueuedJob | None:
        result = self.r.blpop(QUEUE_KEY, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        data = raw.decode() if isinstance(raw, bytes) else raw
        return QueuedJob.from_json(data)

    def task_done(self) -> None:
        self.r.hincrby(STATS_KEY, "matched", 1)

    @property
    def pending(self) -> int:
        return self.r.llen(QUEUE_KEY)

    @property
    def scraped_count(self) -> int:
        val = self.r.hget(STATS_KEY, "scraped")
        return int(val) if val else 0

    @property
    def matched_count(self) -> int:
        val = self.r.hget(STATS_KEY, "matched")
        return int(val) if val else 0

    def log_status(self) -> str:
        return (
            f"Queue: {self.pending} pending | "
            f"{self.scraped_count} scraped | "
            f"{self.matched_count} matched"
        )
