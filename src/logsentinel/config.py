"""Runtime configuration from environment variables (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str
    kafka_topic: str
    kafka_group: str
    database_url: str | None
    model_path: str
    log_format: str
    anthropic_api_key: str | None
    llm_model: str
    metrics_port: int


def get_settings() -> Settings:
    """Read settings at call time, so tests and containers can override them."""
    return Settings(
        kafka_bootstrap=_env("KAFKA_BOOTSTRAP", "localhost:19092"),
        kafka_topic=_env("KAFKA_TOPIC", "logs"),
        kafka_group=_env("KAFKA_GROUP", "logsentinel-detector"),
        database_url=_env("DATABASE_URL"),
        model_path=_env("MODEL_PATH", "models/model.pkl"),
        log_format=_env("LOG_FORMAT", "app"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        llm_model=_env("LLM_MODEL", "claude-haiku-4-5-20251001"),
        metrics_port=int(_env("METRICS_PORT", "8001")),
    )
