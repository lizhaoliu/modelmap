"""Retry wrapper for Hub metadata calls.

Header sweeps make dozens-to-hundreds of ranged requests; an occasional
connection gets refused or times out mid-burst. Retry transport-level
failures with exponential backoff — never HTTP-level errors (404/403/gated
are real answers, not flakiness).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def _transport_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import httpx

        errs.append(httpx.TransportError)
    except ImportError:
        pass
    try:
        import requests

        errs.append(requests.exceptions.ConnectionError)
        errs.append(requests.exceptions.Timeout)
    except ImportError:
        pass
    return tuple(errs)


def with_retries(fn: Callable[[], T], attempts: int = 3, base_delay: float = 0.5) -> T:
    retryable = _transport_errors()
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except retryable as e:  # noqa: PERF203
            last = e
            if i < attempts - 1:
                delay = base_delay * (2**i)
                log.warning("transient hub error (%s); retrying in %.1fs", type(e).__name__, delay)
                time.sleep(delay)
    assert last is not None
    raise last
