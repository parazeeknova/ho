"""ML package configuration — env-driven, safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v.strip() if v and v.strip() else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class GmailPushConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("GMAIL_PUSH", False))
    client_id: str = field(default_factory=lambda: _env_str("GOOGLE_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: _env_str("GOOGLE_CLIENT_SECRET", ""))
    refresh_token: str = field(default_factory=lambda: _env_str("GMAIL_REFRESH_TOKEN", ""))
    project_id: str = field(default_factory=lambda: _env_str("GCP_PUBSUB_PROJECT", ""))
    topic: str = field(default_factory=lambda: _env_str("GCP_PUBSUB_TOPIC", "ho-gmail-events"))
    subscription: str = field(
        default_factory=lambda: _env_str("GCP_PUBSUB_SUBSCRIPTION", "ho-gmail-events-sub")
    )
    poll_interval_s: int = field(
        default_factory=lambda: _env_int("GMAIL_PUSH_POLL_INTERVAL_S", 180)
    )
    watch_ttl_days: int = field(default_factory=lambda: _env_int("GMAIL_WATCH_TTL_DAYS", 6))


@dataclass
class LearningConfig:
    gamma: float = field(default_factory=lambda: _env_float("ML_GAMMA", 0.99))
    exploration_initial: float = field(
        default_factory=lambda: _env_float("ML_EXPLORATION_INITIAL", 0.30)
    )
    exploration_mature: float = field(
        default_factory=lambda: _env_float("ML_EXPLORATION_MATURE", 0.05)
    )
    # Asymmetric: recommendation can explore more than autonomous application.
    exploration_application: float = field(
        default_factory=lambda: _env_float("ML_EXPLORATION_APPLICATION", 0.02)
    )
    # Hierarchical source rewards — quick tier for source bandit learning.
    # Long-term hiring reward is maintained separately.
    shadow_mode: bool = field(default_factory=lambda: _env_bool("ML_SHADOW_MODE", True))


@dataclass
class TrainingConfig:
    train_window_days: int = field(default_factory=lambda: _env_int("ML_TRAIN_WINDOW_DAYS", 90))
    val_window_days: int = field(default_factory=lambda: _env_int("ML_VAL_WINDOW_DAYS", 14))
    test_window_days: int = field(default_factory=lambda: _env_int("ML_TEST_WINDOW_DAYS", 14))
    min_positive_samples: int = field(
        default_factory=lambda: _env_int("ML_MIN_POSITIVE_SAMPLES", 20)
    )
    # Outcome maturity window (P0 censoring fix): impressions younger than this
    # many days are CENSORED — no outcome has had time to mature, so they are
    # excluded from supervised label training instead of being labeled as
    # negatives merely because no interview/offer has appeared yet.
    label_maturity_days: float = field(
        default_factory=lambda: _env_float("ML_LABEL_MATURITY_DAYS", 7.0)
    )


@dataclass
class MlConfig:
    gmail_push: GmailPushConfig = field(default_factory=GmailPushConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifact_dir: str = field(default_factory=lambda: _env_str("ML_ARTIFACT_DIR", "artifacts/ml"))
    dataset_dir: str = field(
        default_factory=lambda: _env_str("ML_DATASET_DIR", "artifacts/datasets")
    )


_ml_config: MlConfig | None = None


def get_ml_config() -> MlConfig:
    global _ml_config
    if _ml_config is None:
        _ml_config = MlConfig()
    return _ml_config
