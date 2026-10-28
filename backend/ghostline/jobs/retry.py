"""Automatic retry of a stage on TransientStageError (build plan Part 21: cached calls make
retries cheap)."""

import logging
import time
from collections.abc import Callable

from ghostline.jobs.errors import TransientStageError

log = logging.getLogger("ghostline.jobs")


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying with exponential backoff while it raises TransientStageError. The last
    error propagates. Any other exception is not retried."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except TransientStageError as exc:
            if attempt == attempts:
                raise
            delay = base_delay_s * 2 ** (attempt - 1)
            log.warning(
                "transient stage error, retrying",
                extra={"attempt": attempt, "delay_s": delay, "error": str(exc)},
            )
            sleep(delay)
    raise AssertionError("unreachable")
