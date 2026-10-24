"""Thin wrapper over the Anthropic SDK: one place for model, effort, caps, fallbacks and cost accounting."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage
from opentelemetry import trace

from hireflow_copilot.config import Effort, Settings
from hireflow_copilot.guard import SpendLedger
from hireflow_copilot.llm.pricing import Usage

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Claude:
    def __init__(self, settings: Settings, ledger: SpendLedger, client: AsyncAnthropic | None = None) -> None:
        self.settings = settings
        self.ledger = ledger
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        # SDK retries 429/5xx twice with backoff; a 60s timeout keeps a stuck call from pinning a worker.
        self.client = client or AsyncAnthropic(api_key=api_key, timeout=60.0, max_retries=2)

    async def create(
        self,
        *,
        system: str,
        messages: Sequence[Any],
        effort: Effort,
        usage: Usage,
        tools: Sequence[Any] = (),
    ) -> BetaMessage:
        extra: dict[str, Any] = {}
        if tools:
            extra["tools"] = list(tools)
        if self.settings.refusal_fallbacks:
            # If a safety classifier declines, Anthropic re-runs the request on its recommended
            # fallback model inside the same call instead of returning an empty refusal.
            extra["betas"] = [FALLBACK_BETA]
            extra["fallbacks"] = "default"

        with tracer.start_as_current_span("claude.messages.create") as span:
            started = time.perf_counter()
            response: BetaMessage = await self.client.beta.messages.create(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                # The system prompt and tool list never change between calls, so they form a cacheable
                # prefix. Automatic caching places the breakpoint on the last cacheable block.
                cache_control={"type": "ephemeral"},
                system=system,
                messages=list(messages),
                output_config={"effort": effort},
                **extra,
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000)

            before = usage.cost_usd
            usage.add(response.model, response.usage)
            call_cost = usage.cost_usd - before
            self.ledger.record(call_cost)

            span.set_attributes(
                {
                    "gen_ai.request.model": self.settings.model,
                    "gen_ai.response.model": response.model,
                    "gen_ai.usage.input_tokens": response.usage.input_tokens,
                    "gen_ai.usage.output_tokens": response.usage.output_tokens,
                    "copilot.cache_read_tokens": response.usage.cache_read_input_tokens or 0,
                    "copilot.cost_usd": call_cost,
                    "copilot.stop_reason": response.stop_reason or "",
                }
            )
            log.info(
                "claude call",
                extra={
                    # Set by the SDK from the request-id header; quote it when reporting issues to Anthropic.
                    "request_id": getattr(response, "_request_id", None),
                    "model": response.model,
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": response.usage.cache_read_input_tokens,
                    "cost_usd": round(call_cost, 6),
                    "latency_ms": elapsed_ms,
                },
            )
            return response
