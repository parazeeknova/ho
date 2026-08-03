"""Centralized, strongly-typed configuration with env-var overrides.

All magic numbers and scattered constants live here. Every module consumes
this one config object. Environment variables override defaults at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is not None:
        return val.strip().lower() in ("1", "true", "yes", "on")
    return default


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Configuration


@dataclass
class HttpConfig:
    """Shared HTTP client settings for all connectors and internal services."""

    default_timeout: float = field(default_factory=lambda: _env_float("HTTP_DEFAULT_TIMEOUT", 30.0))
    connect_timeout: float = field(default_factory=lambda: _env_float("HTTP_CONNECT_TIMEOUT", 10.0))
    max_keepalive: int = field(default_factory=lambda: _env_int("HTTP_MAX_KEEPALIVE", 20))
    max_connections: int = field(default_factory=lambda: _env_int("HTTP_MAX_CONNECTIONS", 100))
    cache_enabled: bool = field(default_factory=lambda: _env_bool("HTTP_CACHE_ENABLED", True))
    cache_max_body_bytes: int = field(
        default_factory=lambda: _env_int("HTTP_CACHE_MAX_BODY_BYTES", 524288)
    )
    cache_ttl_default: int = field(default_factory=lambda: _env_int("HTTP_CACHE_TTL_DEFAULT", 900))


@dataclass
class RetryConfig:
    """Global retry policy."""

    max_retries: int = field(default_factory=lambda: _env_int("RETRY_MAX", 3))
    base_delay: float = field(default_factory=lambda: _env_float("RETRY_BASE_DELAY", 1.0))
    max_delay: float = field(default_factory=lambda: _env_float("RETRY_MAX_DELAY", 30.0))
    jitter: bool = field(default_factory=lambda: _env_bool("RETRY_JITTER", True))


@dataclass
class RateLimitConfig:
    """Per-connector rate limiting (seconds between calls)."""

    yc: float = field(default_factory=lambda: _env_float("RATE_YC", 2.0))
    producthunt: float = field(default_factory=lambda: _env_float("RATE_PRODUCTHUNT", 1.5))
    github: float = field(default_factory=lambda: _env_float("RATE_GITHUB", 3.0))
    hn: float = field(default_factory=lambda: _env_float("RATE_HN", 2.0))
    vc: float = field(default_factory=lambda: _env_float("RATE_VC", 2.0))
    founder_social: float = field(default_factory=lambda: _env_float("RATE_FOUNDER_SOCIAL", 1.5))


@dataclass
class SchedulerConfig:
    """WorkScheduler / CrawlFrontier settings."""

    worker_count: int = field(default_factory=lambda: _env_int("SCHEDULER_WORKERS", 4))
    max_queue_size: int = field(default_factory=lambda: _env_int("MAX_QUEUE_SIZE", 500))
    lease_ttl: float = field(default_factory=lambda: _env_float("LEASE_TTL", 120.0))
    heartbeat_interval: float = field(
        default_factory=lambda: _env_float("HEARTBEAT_INTERVAL", 30.0)
    )
    drain_timeout: float = field(default_factory=lambda: _env_float("DRAIN_TIMEOUT", 30.0))
    batch_max: int = field(default_factory=lambda: _env_int("BATCH_MAX", 3))
    consecutive_failure_threshold: int = field(
        default_factory=lambda: _env_int("CONSECUTIVE_FAILURE_THRESHOLD", 5)
    )
    consecutive_empty_threshold: int = field(
        default_factory=lambda: _env_int("CONSECUTIVE_EMPTY_THRESHOLD", 20)
    )


@dataclass
class EventBusConfig:
    """EventBus TTL-cache settings."""

    cache_maxsize: int = field(default_factory=lambda: _env_int("EVENT_CACHE_MAXSIZE", 10000))
    cache_ttl: float = field(default_factory=lambda: _env_float("EVENT_CACHE_TTL", 3600.0))


@dataclass
class PipelineConfig:
    """Orchestrator pipeline settings."""

    target: int = field(default_factory=lambda: _env_int("PIPELINE_TARGET", 15))
    max_scrape_workers: int = field(default_factory=lambda: _env_int("MAX_SCRAPE_WORKERS", 18))
    match_concurrency: int = field(default_factory=lambda: _env_int("MATCH_CONCURRENCY", 24))
    verify_concurrency: int = field(default_factory=lambda: _env_int("VERIFY_CONCURRENCY", 20))
    sweep_interval: float = field(default_factory=lambda: _env_float("SWEEP_INTERVAL", 300.0))


@dataclass
class Neo4jConfig:
    """Neo4j graph database settings."""

    uri: str = field(default_factory=lambda: _env_str("NEO4J_URI", "bolt://127.0.0.1:7687"))
    username: str = field(default_factory=lambda: _env_str("NEO4J_USERNAME", "neo4j"))
    password: str = field(default_factory=lambda: _env_str("NEO4J_PASSWORD", "password"))
    max_connection_lifetime: int = field(
        default_factory=lambda: _env_int("NEO4J_MAX_CONN_LIFETIME", 3600)
    )


@dataclass
class PostgresConfig:
    """PostgreSQL / pgvector settings."""

    dsn: str = field(
        default_factory=lambda: _env_str(
            "POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5433/agent_memory"
        )
    )
    min_pool: int = field(default_factory=lambda: _env_int("POSTGRES_MIN_POOL", 2))
    max_pool: int = field(default_factory=lambda: _env_int("POSTGRES_MAX_POOL", 25))
    vector_dim: int = field(default_factory=lambda: _env_int("VECTOR_DIM", 1024))


@dataclass
class SearXNGCongfig:
    """SearXNG metasearch settings."""

    url: str = field(
        default_factory=lambda: _env_str("SEARXNG_URL", "http://localhost:8080/search")
    )
    timeout: float = field(default_factory=lambda: _env_float("SEARXNG_TIMEOUT", 6.0))
    semaphore: int = field(default_factory=lambda: _env_int("SEARXNG_SEMAPHORE", 5))


@dataclass
class FirecrawlConfig:
    """Firecrawl service settings."""

    url: str = field(default_factory=lambda: _env_str("FIRECRAWL_URL", "http://127.0.0.1:3002"))
    timeout: float = field(default_factory=lambda: _env_float("FIRECRAWL_TIMEOUT", 60.0))
    map_limit: int = field(default_factory=lambda: _env_int("FIRECRAWL_MAP_LIMIT", 200))
    scrape_limit: int = field(default_factory=lambda: _env_int("FIRECRAWL_SCRAPE_LIMIT", 200))


@dataclass
class EmbedConfig:
    """Embedding server settings."""

    url: str = field(default_factory=lambda: _env_str("EMBED_URL", "http://127.0.0.1:8900/v1"))
    model: str = field(default_factory=lambda: _env_str("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"))
    timeout: float = field(default_factory=lambda: _env_float("EMBED_TIMEOUT", 4.0))


@dataclass
class LLMConfig:
    """LLM / GeneralCompute settings."""

    api_key: str = field(default_factory=lambda: _env_str("GENERALCOMPUTE_API_KEY", ""))
    model: str = field(default_factory=lambda: _env_str("GENERALCOMPUTE_MODEL", "deepseek-v3.2"))
    context_length: int = field(default_factory=lambda: _env_int("LLM_CONTEXT_LENGTH", 32768))
    token_rate: float = field(default_factory=lambda: _env_float("LLM_TOKEN_RATE", 1.4))
    token_max: int = field(default_factory=lambda: _env_int("LLM_TOKEN_MAX", 30))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    retry_delay: float = field(default_factory=lambda: _env_float("LLM_RETRY_DELAY", 2.0))
    rate_penalty_secs: float = field(
        default_factory=lambda: _env_float("LLM_RATE_PENALTY_SECS", 60.0)
    )
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))


@dataclass
class LinkedinGuestConfig:
    """LinkedIn guest API scraping settings."""

    timeout: float = field(default_factory=lambda: _env_float("LINKEDIN_TIMEOUT", 12.0))
    max_pages: int = field(default_factory=lambda: _env_int("LINKEDIN_MAX_PAGES", 4))
    delay_min: float = field(default_factory=lambda: _env_float("LINKEDIN_DELAY_MIN", 1.5))
    delay_max: float = field(default_factory=lambda: _env_float("LINKEDIN_DELAY_MAX", 3.5))


def _load_persona_file() -> str:
    """Load persona.txt if it exists, otherwise fall back to env var."""
    if os.path.exists("persona.txt"):
        with open("persona.txt") as f:
            return f.read()
    return _env_str("CANDIDATE_PERSONA", "early-career / new-grad / intern based in India")


@dataclass
class CandidateConfig:
    """Candidate-specific job-matching parameters — configurable by persona."""

    persona: str = field(default_factory=_load_persona_file)
    min_salary: str = field(
        default_factory=lambda: _env_str("CANDIDATE_MIN_SALARY", "70K INR/month")
    )


@dataclass
class LlmQueueConfig:
    """LLM work-queue rate-limiting and budget controls for the radar pipeline."""

    requests_per_minute: int = field(default_factory=lambda: _env_int("LLM_QUEUE_RPM", 240))
    estimated_tokens_per_minute: int = field(
        default_factory=lambda: _env_int("LLM_QUEUE_TPM", 400000)
    )
    max_in_flight: int = field(default_factory=lambda: _env_int("LLM_QUEUE_MAX_IN_FLIGHT", 30))
    match_token_budget: int = field(
        default_factory=lambda: _env_int("LLM_QUEUE_MATCH_TOKENS", 2000)
    )
    vector_gate_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_QUEUE_VECTOR_GATE", True)
    )
    vector_gate_threshold: float = field(
        default_factory=lambda: _env_float("LLM_QUEUE_VECTOR_THRESHOLD", 0.35)
    )
    cooldown_seconds: float = field(default_factory=lambda: _env_float("LLM_QUEUE_COOLDOWN", 30.0))
    jitter_seconds: float = field(default_factory=lambda: _env_float("LLM_QUEUE_JITTER", 5.0))
    # Shared provider budget: the provider (and Firecrawl's LLM calls) share one
    # per-minute quota. Radar reserves this fraction of it; Redis makes the
    # budget hold across radar processes (master + workers) atomically.
    budget_radar_rpm: int = field(default_factory=lambda: _env_int("LLM_BUDGET_RADAR_RPM", 70))
    budget_radar_tpm: int = field(default_factory=lambda: _env_int("LLM_BUDGET_RADAR_TPM", 140000))
    budget_redis_url: str = field(
        default_factory=lambda: _env_str("LLM_BUDGET_REDIS_URL", "redis://127.0.0.1:6379/1")
    )
    budget_redis_enabled: bool = field(
        default_factory=lambda: _env_bool("LLM_BUDGET_REDIS_ENABLED", True)
    )


@dataclass
class RadarConfig:
    """Job radar v2 pipeline settings."""

    poll_high_freq_seconds: float = field(
        default_factory=lambda: _env_float("RADAR_HIGH_FREQ_POLL", 180.0)
    )
    poll_low_freq_seconds: float = field(
        default_factory=lambda: _env_float("RADAR_LOW_FREQ_POLL", 1800.0)
    )
    max_candidates_per_sweep: int = field(
        default_factory=lambda: _env_int("RADAR_MAX_CANDIDATES", 300)
    )
    urgent_window_hours: int = field(
        default_factory=lambda: _env_int("RADAR_URGENT_WINDOW_HOURS", 48)
    )
    stale_days: int = field(default_factory=lambda: _env_int("RADAR_STALE_DAYS", 30))
    source_min_confidence: float = field(
        default_factory=lambda: _env_float("RADAR_SOURCE_MIN_CONFIDENCE", 0.3)
    )
    # Discovery source for new companies. "azure" = only the Azure relic's
    # company index blobs; "all" = local adapters (dealroom/YC/VC/HN/etc);
    # "none" = disable company discovery entirely.
    discovery_source: str = field(
        default_factory=lambda: _env_str("DISCOVERY_SOURCE", "all").lower()
    )


@dataclass
class ATSCrawlerConfig:
    """ATS-specific crawling parameters."""

    greenhouse_base: str = "https://boards.greenhouse.io"
    lever_base: str = "https://jobs.lever.co"
    ashby_base: str = "https://jobs.ashbyhq.com"
    workable_base: str = "https://apply.workable.com"
    smartrecruiters_base: str = "https://jobs.smartrecruiters.com"
    rippling_base: str = "https://app.rippling.com/careers"
    max_pages_per_board: int = field(default_factory=lambda: _env_int("ATS_MAX_PAGES", 10))
    snapshot_ttl_hours: int = field(default_factory=lambda: _env_int("ATS_SNAPSHOT_TTL_HOURS", 6))


@dataclass
class Config:
    """Root configuration aggregating all subsystems."""

    http: HttpConfig = field(default_factory=HttpConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    searxng: SearXNGCongfig = field(default_factory=SearXNGCongfig)
    firecrawl: FirecrawlConfig = field(default_factory=FirecrawlConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    linkedin_guest: LinkedinGuestConfig = field(default_factory=LinkedinGuestConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    llm_queue: LlmQueueConfig = field(default_factory=LlmQueueConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    ats: ATSCrawlerConfig = field(default_factory=ATSCrawlerConfig)

    def validate(self) -> list[str]:
        """Run startup validation. Returns list of problems (empty = all good)."""
        problems: list[str] = []

        if self.http.default_timeout <= 0:
            problems.append("http.default_timeout must be > 0")
        if self.http.max_connections <= 0:
            problems.append("http.max_connections must be > 0")
        if self.retry.max_retries < 0:
            problems.append("retry.max_retries must be >= 0")
        if self.scheduler.worker_count <= 0:
            problems.append("scheduler.worker_count must be > 0")
        if self.postgres.min_pool > self.postgres.max_pool:
            problems.append("postgres.min_pool must be <= max_pool")
        if self.event_bus.cache_maxsize <= 0:
            problems.append("event_bus.cache_maxsize must be > 0")

        return problems

    def as_dict(self) -> dict[str, dict[str, Any]]:
        """Return a flat dict suitable for health-check reporting."""
        return {
            "http": {
                "default_timeout": self.http.default_timeout,
                "max_connections": self.http.max_connections,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "base_delay": self.retry.base_delay,
            },
            "scheduler": {
                "worker_count": self.scheduler.worker_count,
                "max_queue_size": self.scheduler.max_queue_size,
                "lease_ttl": self.scheduler.lease_ttl,
            },
            "event_bus": {
                "cache_maxsize": self.event_bus.cache_maxsize,
                "cache_ttl": self.event_bus.cache_ttl,
            },
            "pipeline": {
                "max_scrape_workers": self.pipeline.max_scrape_workers,
                "match_concurrency": self.pipeline.match_concurrency,
            },
            "postgres": {
                "min_pool": self.postgres.min_pool,
                "max_pool": self.postgres.max_pool,
            },
            "llm": {
                "model": self.llm.model,
            },
            "llm_queue": {
                "rpm": self.llm_queue.requests_per_minute,
                "tpm": self.llm_queue.estimated_tokens_per_minute,
                "max_in_flight": self.llm_queue.max_in_flight,
            },
            "radar": {
                "urgent_window_hours": self.radar.urgent_window_hours,
                "stale_days": self.radar.stale_days,
            },
        }


_config: Config | None = None


def get_config() -> Config:
    """Return the singleton Config, creating it on first call."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Override the singleton (useful for tests)."""
    global _config
    _config = config
