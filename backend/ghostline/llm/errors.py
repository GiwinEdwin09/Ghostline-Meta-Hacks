"""Errors raised by the model clients."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every error the model clients raise."""


class PermanentLLMError(LLMError):
    """The request will never succeed unchanged (HTTP 400, 401, 402, 403, 404, 413, 504)."""

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class ModelNotFoundError(PermanentLLMError):
    """The model ID is unknown, or the key has no access to it (Wonder is in preview)."""


class TransientLLMError(LLMError):
    """Retries were exhausted on a 429, 5xx or timeout."""


class ReplayMissError(LLMError):
    """Replay mode found no recorded response for this request."""

    def __init__(self, kind: str, request_hash: str, model: str | None = None):
        super().__init__(
            f"no recorded {kind} response for request {request_hash[:12]} (model {model}); "
            "record it with LLM_MODE=record"
        )
        self.kind = kind
        self.request_hash = request_hash


class StructuredOutputError(LLMError):
    """The reply still failed schema validation after one repair call."""

    def __init__(self, message: str, *, raw_text: str = ""):
        super().__init__(message)
        self.raw_text = raw_text


class BudgetExceeded(LLMError):
    """A spend cap was reached."""


class ExperimentBudgetExceeded(BudgetExceeded):
    """The experiment hit BUDGET_USD_PER_EXPERIMENT: finish in library-only mode."""

    def __init__(self, experiment_id: str, spent: float, cap: float):
        super().__init__(f"experiment {experiment_id} has spent ${spent:.2f} of its ${cap:.2f} cap")
        self.experiment_id = experiment_id
        self.spent = spent
        self.cap = cap


class DailyBudgetExceeded(BudgetExceeded):
    """BUDGET_USD_DAILY was reached: new analyses are refused, examples stay viewable."""

    def __init__(self, spent: float, cap: float):
        super().__init__(
            "Ghostline has reached today's analysis budget, so new experiments are paused until "
            "tomorrow. The example experiments are still available to explore."
        )
        self.spent = spent
        self.cap = cap
