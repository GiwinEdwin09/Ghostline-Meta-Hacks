"""Exceptions that pipeline stages and the job runner use to describe how a job ended."""


class StageError(Exception):
    """A failure with a message the user can read. Raise it from a stage instead of a bare
    exception. `retryable=False` hides the Retry button, for problems only a new clip can fix
    (the object can't be measured, the sheet isn't visible)."""

    def __init__(self, message: str, *, code: str = "stage_error", retryable: bool = True):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable


class TransientStageError(StageError):
    """A failure worth trying again automatically (a 429 or 5xx that outlasted the client's own
    retries, a flaky download). The runner retries the stage before failing the job."""

    def __init__(self, message: str, *, code: str = "transient"):
        super().__init__(message, code=code, retryable=True)


class JobAbandoned(Exception):
    """The job was marked failed while its worker was still running (no heartbeat for too long).
    The worker stops quietly; the job row already says what happened."""
