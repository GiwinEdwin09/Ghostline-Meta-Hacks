"""The model clients a job uses, wired to Postgres (build plan Parts 21 and 23).

`llm.Runtime` keeps its ledger behind a small `SpendStore` interface. Here that interface is
backed by the repo, so every model call lands in `llm_calls` and `spend_ledger`, adds to the
experiment's `cost_usd`, and counts toward the daily and per-experiment caps even when the calls
come from different workers. `JobContext` hands stages a ready-made `meta` and `reviewer`.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from uuid import UUID

from ghostline.config import Settings
from ghostline.jobs.errors import StageError, TransientStageError
from ghostline.llm import (
    BudgetExceeded,
    CallRecord,
    DailyBudgetExceeded,
    LLMError,
    ModelNotFoundError,
    PermanentLLMError,
    ReplayMissError,
    Runtime,
    StructuredOutputError,
    TransientLLMError,
)
from ghostline.store.records import LlmCall
from ghostline.store.repo import Repo
from ghostline.store.volume import VolumePaths


def _uuid_or_none(value: str | None) -> UUID | None:
    try:
        return UUID(str(value)) if value else None
    except ValueError:
        return None


class RepoSpendStore:
    """`llm.SpendStore` over Postgres: one `llm_calls` row per call, and a `spend_ledger` row
    for each call that cost something (cache hits cost nothing and write no ledger row)."""

    def __init__(self, repo: Repo):
        self._repo = repo

    def add(self, record: CallRecord) -> None:
        experiment_id = _uuid_or_none(record.experiment_id)
        self._repo.insert_llm_call(
            LlmCall(
                experiment_id=experiment_id,
                phase=record.phase,
                provider=record.provider,
                model=record.model,
                input_tokens=record.input_tokens,
                cached_tokens=record.cached_tokens,
                output_tokens=record.output_tokens,
                frames=record.frames,
                cost_usd=record.cost_usd,
                latency_ms=record.latency_ms,
                cache_hit=record.cache_hit,
                request_hash=record.request_hash,
            )
        )
        if record.cost_usd > 0:
            category = "sam" if record.kind.startswith("sam") else "llm"
            if record.provider == "modal":
                category = "reviewer"
            self._repo.add_spend(
                amount_usd=record.cost_usd,
                provider=record.provider,
                experiment_id=experiment_id,
                category=category,
                ref=record.request_hash,
            )

    def experiment_total(self, experiment_id: str) -> float:
        parsed = _uuid_or_none(experiment_id)
        return self._repo.spend_for_experiment(parsed) if parsed else 0.0

    def day_total(self, day: date) -> float:
        return self._repo.spend_on_day(day)

    def records(self, experiment_id: str | None = None) -> list[CallRecord]:
        return []  # the guard only needs totals; the rows are in llm_calls


def cache_dir_for(paths: VolumePaths) -> str:
    """Where recorded calls live: `LLM_CACHE_DIR` if set, else /cache on the Volume (Part 22)."""
    return os.environ.get("LLM_CACHE_DIR") or str(paths.root / "cache")


def make_runtime(settings: Settings, repo: Repo, paths: VolumePaths) -> Runtime:
    """One runtime per job, so the Meta and reviewer clients feed the same ledger."""
    return Runtime.create(settings, cache_dir=cache_dir_for(paths), store=RepoSpendStore(repo))


@contextmanager
def llm_errors() -> Iterator[None]:
    """Turn model-client failures into StageErrors with messages a person can read."""
    try:
        yield
    except DailyBudgetExceeded as exc:
        raise StageError(str(exc), code="daily_budget_reached", retryable=False) from exc
    except BudgetExceeded as exc:
        raise StageError(
            "This experiment reached its model budget before it finished.",
            code="budget_exceeded",
            retryable=False,
        ) from exc
    except ModelNotFoundError as exc:
        raise StageError(
            "The model isn't available to this API key. Check the model names in the settings.",
            code="model_not_found",
            retryable=False,
        ) from exc
    except TransientLLMError as exc:
        raise TransientStageError(f"The model service is busy: {exc}", code="model_busy") from exc
    except ReplayMissError as exc:
        raise StageError(str(exc), code="replay_miss", retryable=False) from exc
    except StructuredOutputError as exc:
        raise StageError(
            "The model's answer could not be read. Press Retry.", code="bad_model_reply"
        ) from exc
    except PermanentLLMError as exc:
        raise StageError(
            f"The model service refused the request: {exc}", code="model_error", retryable=False
        ) from exc
    except LLMError as exc:
        raise StageError(f"Model call failed: {exc}", code="model_error") from exc
