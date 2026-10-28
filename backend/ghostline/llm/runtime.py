"""Shared machinery behind every model call: mode, cache, retries, ledger and budget guard."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ghostline.config import LLMMode, Settings, get_settings

from .budget import BudgetGuard
from .cache import ReplayCache, normalize_request, request_hash
from .errors import ReplayMissError
from .ledger import CallRecord, InMemorySpendStore, SpendStore
from .retry import MAX_ATTEMPTS, call_with_retry

DEFAULT_CACHE_DIR = ".cache/ghostline"


@dataclass(frozen=True)
class CallSummary:
    """What the ledger records about one call."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    frames: int = 0
    cost_usd: float = 0.0


@dataclass
class Runtime:
    """One per process. Share it between the Meta and reviewer clients so they feed one ledger.

    - `live`: every call goes to the API.
    - `record`: like live, and each response is also written to the cache.
    - `replay`: no network. A call returns its recorded response or raises `ReplayMissError`.
    """

    settings: Settings
    mode: LLMMode
    cache: ReplayCache
    store: SpendStore
    guard: BudgetGuard
    attempts: int = MAX_ATTEMPTS
    sleep: Callable[[float], None] = time.sleep
    rng: Callable[[], float] = random.random
    file_hashes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        *,
        mode: LLMMode | None = None,
        cache_dir: str | Path | None = None,
        store: SpendStore | None = None,
        **kwargs: Any,
    ) -> Runtime:
        """Build a runtime from settings. The cache directory comes from `LLM_CACHE_DIR`."""
        settings = settings or get_settings()
        store = store if store is not None else InMemorySpendStore()
        return cls(
            settings=settings,
            mode=mode or settings.llm_mode,
            cache=ReplayCache(cache_dir or os.environ.get("LLM_CACHE_DIR", DEFAULT_CACHE_DIR)),
            store=store,
            guard=kwargs.pop("guard", None) or BudgetGuard.from_settings(store, settings),
            **kwargs,
        )

    def run[T](
        self,
        kind: str,
        request: dict[str, Any],
        *,
        live: Callable[[], T],
        encode: Callable[[T], dict[str, Any]],
        decode: Callable[[dict[str, Any]], T],
        summarize: Callable[[T], CallSummary],
        model: str,
        provider: str = "meta",
        experiment_id: str | None = None,
        phase: str | None = None,
    ) -> tuple[T, str, bool]:
        """Execute one call under the current mode. Returns (result, request_hash, cache_hit).

        `encode` turns a live result into the JSON-able body of a cache entry (`response`, and
        `events` for SAM video) and `decode` rebuilds the result from the whole entry.
        A cache hit is logged to the ledger at zero cost, so replays never spend or trip a cap.
        """
        normalized = normalize_request(request, self.file_hashes)
        key = request_hash(kind, normalized)
        if self.mode == "replay":
            entry = self.cache.get(kind, key)
            if entry is None:
                raise ReplayMissError(kind, key, model)
            result = decode(entry)
            self._log(kind, model, provider, key, experiment_id, phase, summarize(result), 0, True)
            return result, key, True

        self.guard.check_call(experiment_id)
        started = time.monotonic()
        result = call_with_retry(live, attempts=self.attempts, sleep=self.sleep, rng=self.rng)
        latency_ms = round((time.monotonic() - started) * 1000)
        summary = summarize(result)
        if self.mode == "record":
            self.cache.put(kind, key, {"model": model, "request": normalized, **encode(result)})
        self._log(kind, model, provider, key, experiment_id, phase, summary, latency_ms, False)
        return result, key, False

    def _log(
        self,
        kind: str,
        model: str,
        provider: str,
        key: str,
        experiment_id: str | None,
        phase: str | None,
        summary: CallSummary,
        latency_ms: int,
        cache_hit: bool,
    ) -> None:
        self.store.add(
            CallRecord(
                provider=provider,
                kind=kind,
                model=model,
                cost_usd=0.0 if cache_hit else summary.cost_usd,
                request_hash=key,
                experiment_id=experiment_id,
                phase=phase,
                input_tokens=summary.input_tokens,
                cached_tokens=summary.cached_tokens,
                output_tokens=summary.output_tokens,
                reasoning_tokens=summary.reasoning_tokens,
                frames=summary.frames,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
            )
        )
