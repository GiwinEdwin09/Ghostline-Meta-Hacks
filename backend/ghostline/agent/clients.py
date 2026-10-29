"""The model clients a stage uses, and where their spend goes.

Nothing else in the code base builds a `Runtime` yet, so the stages in `agent/` and `review/` get
their clients here:

- `RepoSpendStore` is the `SpendStore` over Postgres. Each call becomes an `llm_calls` row, and
  its cost goes into the spend ledger and `experiments.cost_usd` (Part 22); the budget guard
  reads the per-experiment and daily totals back from the ledger.
- `get_clients(ctx)` builds one `Runtime` (mode from `LLM_MODE`, replay cache on the Volume at
  `cache/`, or `LLM_CACHE_DIR`) and a Meta and a reviewer client that share it.
- `use_clients(meta, reviewer)` replaces both, for tests and for the recording script.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from uuid import UUID

from ghostline.jobs.context import JobContext
from ghostline.llm import CallRecord, MetaClient, ReviewerClient, Runtime
from ghostline.store.records import LlmCall
from ghostline.store.repo import Repo

_override: tuple[MetaClient, ReviewerClient] | None = None


class RepoSpendStore:
    """`SpendStore` over the Postgres repo."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo

    @staticmethod
    def _uuid(value: str | None) -> UUID | None:
        try:
            return UUID(value) if value else None
        except ValueError:
            return None

    def add(self, record: CallRecord) -> None:
        experiment_id = self._uuid(record.experiment_id)
        self.repo.insert_llm_call(
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
            category = (
                "reviewer"
                if record.provider == "modal"
                else "sam"
                if record.kind.startswith("sam")
                else "llm"
            )
            self.repo.add_spend(
                amount_usd=record.cost_usd,
                provider=record.provider,
                experiment_id=experiment_id,
                category=category,  # type: ignore[arg-type]
                ref=record.request_hash,
            )

    def experiment_total(self, experiment_id: str) -> float:
        uid = self._uuid(experiment_id)
        return self.repo.spend_for_experiment(uid) if uid else 0.0

    def day_total(self, day: date) -> float:
        return self.repo.spend_today()

    def records(self, experiment_id: str | None = None) -> list[CallRecord]:
        return []


def get_clients(ctx: JobContext) -> tuple[MetaClient, ReviewerClient]:
    if _override is not None:
        return _override
    runtime = Runtime.create(
        ctx.settings,
        cache_dir=None if os.environ.get("LLM_CACHE_DIR") else ctx.paths.root / "cache",
        store=RepoSpendStore(ctx.repo),
    )
    return MetaClient(runtime), ReviewerClient(runtime)


@contextmanager
def use_clients(meta: MetaClient, reviewer: ReviewerClient) -> Iterator[None]:
    global _override
    previous, _override = _override, (meta, reviewer)
    try:
        yield
    finally:
        _override = previous
