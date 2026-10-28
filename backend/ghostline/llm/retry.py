"""Retries with exponential backoff (build plan Part 23).

429, 5xx and timeouts are retried: delay 500 ms doubling each attempt, with jitter, and never
shorter than a `Retry-After` header. A 400-class error is permanent and is never retried
unchanged. So is a 504, which Meta's docs say needs a smaller request, not a repeat.
"""

from __future__ import annotations

import email.utils
import logging
import random
import time
from collections.abc import Callable, Mapping

import httpx
import httpx2
import openai

from .errors import LLMError, ModelNotFoundError, PermanentLLMError, TransientLLMError

log = logging.getLogger("ghostline.llm")

MAX_ATTEMPTS = 4
BASE_DELAY_S = 0.5
MAX_DELAY_S = 90.0
_TRANSPORT_ERRORS = (httpx.TransportError, httpx2.TransportError)  # the SDK uses httpx2
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 529}


class RetryableError(Exception):
    """Raised inside an attempt for a failure that is worth retrying, such as a streamed
    `response.failed` event with a server-side error code."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Seconds to wait from `retry-after-ms` or `Retry-After` (seconds or an HTTP date)."""
    if not headers:
        return None
    lowered = {k.lower(): v for k, v in headers.items()}
    if "retry-after-ms" in lowered:
        try:
            return float(lowered["retry-after-ms"]) / 1000
        except ValueError:
            pass
    value = lowered.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(parsed.timestamp() - time.time(), 0.0)


def _error_code(error: openai.APIStatusError) -> str | None:
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict):
            return inner.get("code")
    return getattr(error, "code", None)


def to_permanent(error: openai.APIStatusError) -> PermanentLLMError:
    code = _error_code(error)
    message = f"{error.status_code}: {error.message}"
    if error.status_code == 404 and code == "model_not_found":
        return ModelNotFoundError(
            f"{message}. The model is unknown or this key has no access to it "
            "(muse-wonder-1.0 is in preview).",
            status_code=404,
            code=code,
        )
    return PermanentLLMError(message, status_code=error.status_code, code=code)


def classify(error: BaseException) -> tuple[bool, float | None]:
    """(retryable, Retry-After seconds) for an exception raised by one attempt."""
    if isinstance(error, RetryableError):
        return True, error.retry_after
    if isinstance(error, openai.APIStatusError):
        retry_after = parse_retry_after(dict(error.response.headers))
        return error.status_code in _RETRYABLE_STATUS, retry_after
    if isinstance(error, (openai.APIConnectionError, *_TRANSPORT_ERRORS)):
        return True, None  # includes timeouts
    return False, None


def backoff_delay(attempt: int, retry_after: float | None, rng: Callable[[], float]) -> float:
    """Delay before retry number `attempt` (1-based): 0.5 s, 1 s, 2 s... with +-25% jitter,
    but never below the server's Retry-After."""
    delay = BASE_DELAY_S * 2 ** (attempt - 1) * (0.75 + 0.5 * rng())
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, MAX_DELAY_S)


def call_with_retry[T](
    attempt_fn: Callable[[], T],
    *,
    attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Run `attempt_fn`, retrying transient failures. Raises `PermanentLLMError` at once for a
    permanent API error and `TransientLLMError` once the attempts run out."""
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return attempt_fn()
        except LLMError:
            raise
        except (
            RetryableError,
            openai.APIStatusError,
            openai.APIConnectionError,
            *_TRANSPORT_ERRORS,
        ) as error:
            retryable, retry_after = classify(error)
            if not retryable:
                if isinstance(error, openai.APIStatusError):
                    raise to_permanent(error) from error
                raise
            last = error
            if attempt == attempts:
                break
            delay = backoff_delay(attempt, retry_after, rng)
            log.warning(
                "model call failed (%s); retry %d/%d in %.1f s",
                type(error).__name__,
                attempt,
                attempts - 1,
                delay,
            )
            sleep(delay)
    raise TransientLLMError(f"gave up after {attempts} attempts: {last}") from last
