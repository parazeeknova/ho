"""Centralized, strongly-typed configuration with env-var overrides.

All magic numbers and scattered constants live here. Every module consumes
this one config object. Environment variables override defaults at startup.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
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


def _env_list(key: str, default: list[str]) -> list[str]:
    """Parse a comma/pipe-separated env var into a list, filtering empties."""
    raw = os.environ.get(key, "")
    if not raw.strip():
        return default
    return [item.strip() for item in re.split(r"[,\|]", raw) if item.strip()]


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
class RenderConfig:
    """In-process page-rendering budget + anti-blocking controls.

    The in-process renderer (static httpx + pooled Chromium) is used instead
    of an external scraping service. Env names use the RENDER_* prefix; the
    legacy FIRECRAWL_* names are still honored for backward compatibility.
    """

    url: str = field(default_factory=lambda: _env_str("FIRECRAWL_URL", "http://127.0.0.1:3002"))
    timeout: float = field(default_factory=lambda: _env_float("FIRECRAWL_TIMEOUT", 60.0))
    map_limit: int = field(default_factory=lambda: _env_int("FIRECRAWL_MAP_LIMIT", 200))
    scrape_limit: int = field(default_factory=lambda: _env_int("FIRECRAWL_SCRAPE_LIMIT", 200))
    scrape_timeout: float = field(
        default_factory=lambda: _env_float("FIRECRAWL_SCRAPE_TIMEOUT", 15.0)
    )
    # Politeness: minimum seconds between requests to the SAME host (jittered).
    host_delay: float = field(default_factory=lambda: _env_float("RENDER_HOST_DELAY", 0.5))
    # SOCKS5 proxy (torproxy on 9050) for JS renders and static fetches when
    # set — rotates egress IP so a site can't block the datacenter IP.
    socks_proxy: str = field(
        default_factory=lambda: _env_str("RENDER_SOCKS_PROXY", "socks5://127.0.0.1:9050")
    )
    # Master switch: route through the proxy by default (masking). Set to
    # false to disable proxying entirely.
    use_proxy: bool = field(default_factory=lambda: _env_bool("RENDER_USE_PROXY", True))
    # Retry a blocked request (429/403/Cloudflare challenge) via the proxy.
    proxy_on_block: bool = field(default_factory=lambda: _env_bool("RENDER_PROXY_ON_BLOCK", True))
    # JS-shell heuristic: a page whose visible text is below this many chars is
    # treated as an un-rendered SPA shell and sent to the browser. Real job
    # pages carry a description body far above this.
    shell_threshold_chars: int = field(
        default_factory=lambda: _env_int("RENDER_SHELL_THRESHOLD_CHARS", 200)
    )
    # Client-side render settle time after domcontentloaded (JS-SPA hydration).
    render_settle_ms: int = field(default_factory=lambda: _env_int("RENDER_SETTLE_MS", 2500))
    proxied_settle_ms: int = field(
        default_factory=lambda: _env_int("RENDER_PROXIED_SETTLE_MS", 3000)
    )
    # In-memory + (optional) Postgres-backed render cache: TTL seconds and the
    # max in-memory entries before stale eviction.
    cache_ttl: int = field(default_factory=lambda: _env_int("RENDER_CACHE_TTL", 600))
    cache_max_entries: int = field(
        default_factory=lambda: _env_int("RENDER_CACHE_MAX_ENTRIES", 4000)
    )
    # Whether to persist the render cache in Postgres (survives restarts and is
    # shared across ingest workers). Falls back to the in-memory dict when the
    # DB is unreachable.
    cache_persist: bool = field(default_factory=lambda: _env_bool("RENDER_CACHE_PERSIST", True))
    # Chromium pool: max concurrent JS renders and idle-close window (ms).
    concurrency: int = field(default_factory=lambda: _env_int("RENDER_CONCURRENCY", 4))
    pool_idle_ms: int = field(default_factory=lambda: _env_int("RENDER_IDLE_MS", 30000))


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
    # Fallback chain tried in order when the primary model is overloaded,
    # rate-limited, or errors out. Comma/pipe-separated in LLM_FALLBACK_MODELS.
    fallback_models: list[str] = field(
        default_factory=lambda: _env_list("LLM_FALLBACK_MODELS", ["deepseek-v3.1", "minimax-m2.7"])
    )
    context_length: int = field(default_factory=lambda: _env_int("LLM_CONTEXT_LENGTH", 32768))
    token_rate: float = field(default_factory=lambda: _env_float("LLM_TOKEN_RATE", 1.4))
    token_max: int = field(default_factory=lambda: _env_int("LLM_TOKEN_MAX", 30))
    max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    retry_delay: float = field(default_factory=lambda: _env_float("LLM_RETRY_DELAY", 2.0))
    # Hard per-attempt timeout: a model that hangs (no error, no response) must
    # not stall the caller forever. When an attempt exceeds this, it counts as
    # a failure and the next fallback model is tried.
    per_call_timeout_s: float = field(
        default_factory=lambda: _env_float("LLM_PER_CALL_TIMEOUT_S", 30.0)
    )
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


def _candidate_persona_paths() -> list[str]:
    """Candidate persona.json locations, most specific first."""
    paths: list[str] = []
    env_file = os.environ.get("CANDIDATE_PERSONA_FILE")
    if env_file:
        paths.append(env_file)
    for cwd in (os.getcwd(), str(Path(__file__).resolve().parents[3])):
        paths.append(os.path.join(cwd, "data", "persona.json"))
    return paths


def _load_persona_file() -> str:
    """Render the flat candidate grounding text from persona.json.

    The persona is a single autogenerated JSON file (identity + grilled
    answers + resume_summary); the radar matcher consumes it as one text
    block. Falls back to the CANDIDATE_PERSONA env var when no persona.json
    has been built yet (`npm run init-memory`).
    """
    for path in _candidate_persona_paths():
        if not os.path.exists(path):
            continue
        try:
            data = json.loads(Path(path).read_text())
        except OSError, json.JSONDecodeError:
            continue
        blocks: list[str] = []
        identity = data.get("identity") or {}
        for key, value in identity.items():
            if value:
                blocks.append(f"- {key}: {value}")
        for a in data.get("answers") or []:
            q, ans = a.get("question"), a.get("answer")
            if q and ans:
                blocks.append(f"- {q}: {ans}")
        resume = (data.get("resume_summary") or "").strip()
        if resume:
            blocks.append("")
            blocks.append("From Resume:")
            blocks.append(resume)
        if blocks:
            return "\n".join(blocks)
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
    """LLM work-queue rate-limiting and budget controls for the radar pipeline.

    Defaults stay under the provider's real cap (GeneralCompute: 100 RPM /
    200K TPM) so the sweep never trips a 429 and stalls a single application
    on a 30-40s cooldown. The queue reserves most of it; radar's shared budget
    is a fraction, and autofill's per-question answers use the interactive
    reserved lane.
    """

    requests_per_minute: int = field(default_factory=lambda: _env_int("LLM_QUEUE_RPM", 85))
    estimated_tokens_per_minute: int = field(
        default_factory=lambda: _env_int("LLM_QUEUE_TPM", 180000)
    )
    max_in_flight: int = field(default_factory=lambda: _env_int("LLM_QUEUE_MAX_IN_FLIGHT", 12))
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
    # Shared provider budget: the provider's per-minute quota (100 RPM / 200K
    # TPM). Radar reserves a fraction of it; Redis makes the budget hold across
    # radar processes (master + workers) atomically.
    budget_radar_rpm: int = field(default_factory=lambda: _env_int("LLM_BUDGET_RADAR_RPM", 60))
    budget_radar_tpm: int = field(default_factory=lambda: _env_int("LLM_BUDGET_RADAR_TPM", 120000))
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
    # Stop-after: when the gate has passed at least this many candidates in a
    # sweep, the master loop ends (overnight runs usually want a bounded batch
    # — e.g. 20 gated jobs — not an endless loop). 0 = no early stop.
    stop_after_gated: int = field(default_factory=lambda: _env_int("RADAR_STOP_AFTER_GATED", 0))
    # Work-session epoch target (the review's pivot #3): the radar keeps
    # discovering/ranking into the application reservoir, but a SESSION completes
    # when at least this many applications are CONFIRMED SUBMITTED (applied_at
    # set — NOT attempts, deferred, captcha, or duplicates). On completion the
    # loop finalizes the session (learning update + re-rank of the remaining
    # reservoir). 0 = disable (bounded by stop_after_gated instead).
    session_application_target: int = field(
        default_factory=lambda: _env_int("RADAR_SESSION_APPLICATION_TARGET", 0)
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
    # US roles are only reachable for this candidate when remote; onsite US
    # postings (which require visa sponsorship to attend) are rejected.
    us_only_remote: bool = field(default_factory=lambda: _env_bool("RADAR_US_ONLY_REMOTE", True))


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
    render: RenderConfig = field(default_factory=RenderConfig)
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
