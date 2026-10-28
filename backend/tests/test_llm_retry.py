import pytest
from fixtures.fakes import connection_error, status_error

from ghostline.llm import ModelNotFoundError, PermanentLLMError, TransientLLMError
from ghostline.llm.retry import (
    RetryableError,
    backoff_delay,
    call_with_retry,
    parse_retry_after,
)


def flaky(*steps):
    """A function that raises or returns each step in turn, and counts its calls."""
    remaining = list(steps)
    calls = []

    def attempt():
        calls.append(1)
        step = remaining.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return attempt, calls


def run(attempt, **kwargs):
    slept = []
    result = call_with_retry(attempt, sleep=slept.append, rng=lambda: 0.5, **kwargs)
    return result, slept


def test_backoff_starts_at_500_ms_and_doubles():
    delays = [backoff_delay(n, None, lambda: 0.5) for n in (1, 2, 3)]

    assert delays == pytest.approx([0.5, 1.0, 2.0])


def test_backoff_jitter_stays_within_a_quarter():
    low, high = backoff_delay(2, None, lambda: 0.0), backoff_delay(2, None, lambda: 1.0)

    assert (low, high) == pytest.approx((0.75, 1.25))


def test_retry_after_raises_the_delay_but_never_lowers_it():
    assert backoff_delay(1, 7.0, lambda: 0.5) == 7.0
    assert backoff_delay(3, 0.1, lambda: 0.5) == pytest.approx(2.0)


def test_retry_after_header_forms():
    assert parse_retry_after({"Retry-After": "12"}) == 12.0
    assert parse_retry_after({"retry-after-ms": "1500"}) == 1.5
    assert parse_retry_after({}) is None
    assert parse_retry_after({"retry-after": "soon"}) is None


def test_429_is_retried_and_honors_retry_after():
    attempt, calls = flaky(status_error(429, {"retry-after": "3"}), "ok")

    result, slept = run(attempt)

    assert result == "ok" and len(calls) == 2
    assert slept == [3.0]


@pytest.mark.parametrize("status", [500, 502, 503])
def test_5xx_is_retried(status):
    attempt, calls = flaky(status_error(status), status_error(status), "ok")

    result, slept = run(attempt)

    assert result == "ok" and len(calls) == 3
    assert slept == pytest.approx([0.5, 1.0])


def test_timeouts_and_dropped_connections_are_retried():
    attempt, calls = flaky(connection_error(), "ok")

    assert run(attempt)[0] == "ok" and len(calls) == 2


def test_a_streamed_server_failure_is_retried():
    attempt, calls = flaky(RetryableError("service_overloaded"), "ok")

    assert run(attempt)[0] == "ok" and len(calls) == 2


def test_at_most_four_attempts():
    attempt, calls = flaky(*[status_error(503)] * 10)

    with pytest.raises(TransientLLMError):
        run(attempt)

    assert len(calls) == 4


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 504])
def test_permanent_errors_are_never_retried(status):
    attempt, calls = flaky(status_error(status), "ok")

    with pytest.raises(PermanentLLMError) as caught:
        run(attempt)

    assert len(calls) == 1
    assert caught.value.status_code == status


def test_model_not_found_has_a_clear_error():
    attempt, _ = flaky(status_error(404, code="model_not_found"))

    with pytest.raises(ModelNotFoundError, match="no access|unknown"):
        run(attempt)


def test_other_exceptions_pass_through():
    attempt, calls = flaky(ValueError("bug"), "ok")

    with pytest.raises(ValueError):
        run(attempt)

    assert len(calls) == 1
