"""Runtime configuration, read from environment variables (prefix ``COPILOT_``) or a ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COPILOT_", env_file=".env", extra="ignore")

    # --- Claude -------------------------------------------------------------------------------
    # Read without the prefix so the standard Anthropic variable works as-is.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    model: str = "claude-opus-5-5"
    ask_effort: Effort = "low"
    agent_effort: Effort = "medium"
    # Hard ceiling per model call (thinking + answer). Keeps a single request's cost bounded.
    max_tokens: int = 4096
    max_agent_steps: int = 6
    # Retry a declined request on Anthropic's recommended fallback model instead of failing.
    refusal_fallbacks: bool = True

    # --- HireFlow ------------------------------------------------------------------------------
    # "fixtures" serves the bundled demo dataset; "graphql" talks to a running HireFlow API.
    source: Literal["fixtures", "graphql"] = "fixtures"
    hireflow_url: str = "http://localhost:5230/graphql"
    # A bearer token for HireFlow, or leave empty and set dev_token=true to mint one from /auth/token.
    hireflow_token: SecretStr | None = None
    hireflow_dev_token: bool = True

    # --- Retrieval -----------------------------------------------------------------------------
    # "fastembed" runs a small ONNX model locally; "hashing" is dependency-free and used by tests.
    embedder: Literal["fastembed", "hashing"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_path: Path = Path("data/index")
    top_k: int = 5

    # --- Guardrails ----------------------------------------------------------------------------
    rate_limit_per_minute: int = 5
    rate_limit_per_day: int = 40
    # Global spend ceiling across all callers. Once reached, LLM endpoints return 429 until UTC midnight.
    daily_budget_usd: float = 1.00
    max_question_chars: int = 1000
    # Optional shared secret. When set, LLM endpoints require the X-Copilot-Key header.
    access_key: SecretStr | None = None
    # Trust X-Forwarded-For for client IPs (only behind a proxy you control, e.g. Azure Container Apps).
    trust_proxy: bool = False

    # --- Observability -------------------------------------------------------------------------
    otel_exporter: Literal["none", "console", "otlp", "azure"] = "none"
    log_level: str = "INFO"

    @property
    def demo_mode(self) -> bool:
        """No API key means no model calls: answers are extractive and the agent is disabled."""
        return self.anthropic_api_key is None or not self.anthropic_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
