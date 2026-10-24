from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from anthropic.types.beta import BetaMessage
from fastapi.testclient import TestClient

from hireflow_copilot.config import Settings
from hireflow_copilot.hireflow.client import FixtureHireFlowClient
from hireflow_copilot.main import create_app
from hireflow_copilot.rag.embeddings import HashingEmbedder

APP_ANA = "40000000-0000-4000-8000-000000000001"  # Ana Popescu, Senior .NET Engineer, INTERVIEW
APP_VICTORIA = "40000000-0000-4000-8000-000000000006"  # Victoria Sandu, AI Engineer, SCREENING


def message(content: list[dict[str, Any]], stop_reason: str = "end_turn", **usage: int) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 1000),
                "output_tokens": usage.get("output_tokens", 200),
                "cache_read_input_tokens": usage.get("cache_read", 0),
                "cache_creation_input_tokens": 0,
            },
        }
    )


class FakeMessages:
    """Replays scripted responses and records every request, standing in for client.beta.messages."""

    def __init__(self, responses: list[BetaMessage]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> BetaMessage:
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeAnthropic:
    def __init__(self, responses: list[BetaMessage]) -> None:
        self.messages = FakeMessages(responses)
        self.beta = self

    def with_options(self, **_: Any) -> FakeAnthropic:
        return self


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"embedder": "hashing", "anthropic_api_key": None, "otel_exporter": "none"}
    return Settings(**(base | overrides))


@pytest.fixture
def demo_client() -> Iterator[TestClient]:
    app = create_app(settings(), hireflow=FixtureHireFlowClient(), embedder=HashingEmbedder())
    with TestClient(app) as client:
        yield client


def llm_client(responses: list[BetaMessage], **overrides: Any) -> tuple[TestClient, FakeAnthropic]:
    fake = FakeAnthropic(responses)
    app = create_app(
        settings(anthropic_api_key="test-key", **overrides),
        hireflow=FixtureHireFlowClient(),
        embedder=HashingEmbedder(),
        anthropic_client=fake,  # type: ignore[arg-type]
    )
    return TestClient(app), fake
