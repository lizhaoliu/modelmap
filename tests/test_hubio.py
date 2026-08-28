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


# ---------------------------------------------------------------- is_auth_error


def test_auth_error_by_exception_type():
    from modelmap.hubio import is_auth_error

    GatedRepoError = type("GatedRepoError", (Exception,), {})
    assert is_auth_error(GatedRepoError("403 Client Error"))


def test_auth_error_through_the_cause_chain():
    """transformers wraps the hub's GatedRepoError in a plain OSError; the
    detector must walk __cause__/__context__ to find it."""
    from modelmap.hubio import is_auth_error

    GatedRepoError = type("GatedRepoError", (Exception,), {})
    try:
        try:
            raise GatedRepoError("401 Client Error")
        except GatedRepoError as inner:
            raise OSError("could not load config") from inner
    except OSError as e:
        assert is_auth_error(e)


def test_auth_error_by_message():
    from modelmap.hubio import is_auth_error

    assert is_auth_error(OSError("You are trying to access a gated repo."))
    assert is_auth_error(OSError("401 Client Error for url https://huggingface.co/x/y"))
    assert not is_auth_error(OSError("404 Repository Not Found"))
    assert not is_auth_error(ValueError("model type `breeze` is unknown"))
    assert not is_auth_error(None)
