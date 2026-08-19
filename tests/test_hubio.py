import pytest

from modelmap.hubio import with_retries


def test_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr("modelmap.hubio.time.sleep", lambda _: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("refused")
        return "ok"

    assert with_retries(flaky) == "ok"
    assert len(calls) == 3


def test_gives_up_after_attempts(monkeypatch):
    monkeypatch.setattr("modelmap.hubio.time.sleep", lambda _: None)

    def always_down():
        raise ConnectionError("refused")

    with pytest.raises(ConnectionError):
        with_retries(always_down, attempts=3)


def test_non_transport_errors_raise_immediately():
    calls = []

    def not_found():
        calls.append(1)
        raise ValueError("404 is an answer, not flakiness")

    with pytest.raises(ValueError):
        with_retries(not_found)
    assert len(calls) == 1


def test_short_hub_rate_limits_are_waited_out(monkeypatch):
    from modelmap import hubio

    monkeypatch.setattr(hubio.time, "sleep", lambda s: None)
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 Too Many Requests: rate limit. Retry after 3 seconds")
        return "ok"

    assert hubio.with_retries(fn) == "ok" and len(calls) == 2
    # long waits are not absorbed: the server reports them instead
    def long_wait():
        raise RuntimeError("429 Too Many Requests: rate limit. Retry after 86 seconds")

    import pytest
    with pytest.raises(RuntimeError):
        hubio.with_retries(long_wait)
