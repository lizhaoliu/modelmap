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
