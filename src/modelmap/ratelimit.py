"""Tiny in-memory token bucket, keyed by client. Enough for a single-process
deployment; put a real limiter at the edge for anything bigger."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, per_min: int, burst: int):
        self.rate = per_min / 60.0
        self.burst = float(max(1, burst))
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                self._prune(now)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            return False, (1.0 - tokens) / self.rate

    def _prune(self, now: float) -> None:
        if len(self._buckets) < 5000:
            return
        stale = [k for k, (_, ts) in self._buckets.items() if now - ts > 600]
        for k in stale:
            del self._buckets[k]
