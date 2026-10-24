"""Cost guardrails for a public demo running on someone's personal API key.

Three independent limits, checked before any model call:
1. per-client request rate (sliding windows: per minute and per day),
2. a global daily spend ceiling in USD, fed by real ``usage`` numbers after each call,
3. per-request caps (question length, ``max_tokens``, agent steps) enforced elsewhere.

State is in-process. That is correct for a single replica; with more than one, move the counters to
Redis (``INCR`` + ``EXPIRE``) so the limits are shared.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from threading import Lock


class LimitExceeded(Exception):
    def __init__(self, reason: str, retry_after: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


def _seconds_to_utc_midnight() -> int:
    now = datetime.now(UTC)
    return 86_400 - (now.hour * 3600 + now.minute * 60 + now.second)


@dataclass
class RateLimiter:
    per_minute: int
    per_day: int
    _hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: Lock = field(default_factory=Lock)

    def check(self, client: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= 86_400:
                hits.popleft()
            last_minute = sum(1 for t in hits if now - t < 60)
            if last_minute >= self.per_minute:
                oldest = next(t for t in hits if now - t < 60)
                raise LimitExceeded(f"Rate limit: {self.per_minute} requests per minute.", int(60 - (now - oldest)) + 1)
            if len(hits) >= self.per_day:
                raise LimitExceeded(f"Rate limit: {self.per_day} requests per day.", int(86_400 - (now - hits[0])) + 1)
            hits.append(now)


@dataclass
class SpendLedger:
    daily_budget_usd: float
    _day: date = field(default_factory=lambda: datetime.now(UTC).date())
    _spent: float = 0.0
    _requests: int = 0
    _lock: Lock = field(default_factory=Lock)

    def _roll(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._day:
            self._day, self._spent, self._requests = today, 0.0, 0

    def check(self) -> None:
        with self._lock:
            self._roll()
            if self._spent >= self.daily_budget_usd:
                raise LimitExceeded(
                    f"Daily demo budget of ${self.daily_budget_usd:.2f} is used up. It resets at 00:00 UTC.",
                    _seconds_to_utc_midnight(),
                )

    def record(self, cost_usd: float) -> None:
        with self._lock:
            self._roll()
            self._spent += cost_usd
            self._requests += 1

    def snapshot(self) -> dict[str, float | int | str]:
        with self._lock:
            self._roll()
            return {
                "day": self._day.isoformat(),
                "spent_usd": round(self._spent, 4),
                "budget_usd": self.daily_budget_usd,
                "remaining_usd": round(max(self.daily_budget_usd - self._spent, 0.0), 4),
                "requests": self._requests,
            }
