"""Token usage to dollars. Rates are USD per million tokens (Anthropic list prices, October 2026)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# model -> (input, output, cache read). Cache writes (5-minute TTL) cost 1.25x input.
RATES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
}
CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, model: str, usage: Any) -> None:
        """Accumulate one response's ``usage`` block, priced at the model that actually served it."""
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += cache_read
        self.cache_write_tokens += cache_write
        self.cost_usd += cost(model, usage.input_tokens, usage.output_tokens, cache_read, cache_write)
        self.calls += 1


def cost(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0) -> float:
    # Unknown models are priced as Opus so the budget errs on the side of stopping early.
    rate_in, rate_out, rate_cached = RATES.get(model, RATES["claude-opus-5-5"])
    total = (
        input_tokens * rate_in
        + output_tokens * rate_out
        + cache_read * rate_cached
        + cache_write * rate_in * CACHE_WRITE_MULTIPLIER
    )
    return total / 1_000_000
