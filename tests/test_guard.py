import pytest

from hireflow_copilot.guard import LimitExceeded, RateLimiter, SpendLedger
from hireflow_copilot.llm.pricing import cost


def test_rate_limiter_per_minute_window() -> None:
    limiter = RateLimiter(per_minute=2, per_day=100)
    limiter.check("a", now=0)
    limiter.check("a", now=1)
    with pytest.raises(LimitExceeded) as exc:
        limiter.check("a", now=2)
    assert exc.value.retry_after == 59
    limiter.check("b", now=2)  # other clients are unaffected
    limiter.check("a", now=61)  # window has moved on


def test_rate_limiter_per_day() -> None:
    limiter = RateLimiter(per_minute=100, per_day=3)
    for t in (0, 100, 200):
        limiter.check("a", now=t)
    with pytest.raises(LimitExceeded):
        limiter.check("a", now=300)
    limiter.check("a", now=86_401)


def test_ledger_blocks_once_budget_is_spent() -> None:
    ledger = SpendLedger(daily_budget_usd=0.05)
    ledger.check()
    ledger.record(0.03)
    ledger.check()
    ledger.record(0.03)
    with pytest.raises(LimitExceeded):
        ledger.check()
    assert ledger.snapshot()["remaining_usd"] == 0


def test_cost_uses_cache_rates() -> None:
    # 1M fresh input + 1M output on Opus 5.5 = $4 + $20; cached reads are 20x cheaper than fresh input.
    assert cost("claude-opus-5-5", 1_000_000, 1_000_000) == pytest.approx(24.0)
    assert cost("claude-opus-5-5", 0, 0, cache_read=1_000_000) == pytest.approx(0.20)
    assert cost("unknown-model", 1_000_000, 0) == pytest.approx(4.0)
