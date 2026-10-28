"""Spend storage: the `llm_calls` rows (build plan Part 22) behind a small interface.

The budget guard only needs `add`, `experiment_total` and `day_total`. The in-memory and
JSON-file stores below are the defaults; the Postgres repo (Part 22) can implement the same
`SpendStore` protocol later and plug in unchanged.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class CallRecord:
    """One model call, shaped like the `llm_calls` table. Cache hits cost nothing."""

    provider: str  # "meta" or "modal"
    kind: str  # respond, chat, sam_video, sam_image
    model: str
    cost_usd: float
    request_hash: str
    experiment_id: str | None = None
    phase: str | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    frames: int = 0
    latency_ms: int = 0
    cache_hit: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(default_factory=lambda: utc_now().isoformat())

    @property
    def day(self) -> date:
        return datetime.fromisoformat(self.ts).astimezone(UTC).date()


@runtime_checkable
class SpendStore(Protocol):
    """Where call records persist, and the totals the budget guard reads."""

    def add(self, record: CallRecord) -> None: ...

    def experiment_total(self, experiment_id: str) -> float: ...

    def day_total(self, day: date) -> float: ...

    def records(self, experiment_id: str | None = None) -> list[CallRecord]: ...


class InMemorySpendStore:
    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._lock = threading.Lock()

    def add(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)

    def experiment_total(self, experiment_id: str) -> float:
        return sum(r.cost_usd for r in self.records(experiment_id))

    def day_total(self, day: date) -> float:
        return sum(r.cost_usd for r in self.records() if r.day == day)

    def records(self, experiment_id: str | None = None) -> list[CallRecord]:
        with self._lock:
            rows = list(self._records)
        return [r for r in rows if experiment_id is None or r.experiment_id == experiment_id]


class JsonFileSpendStore:
    """Append-only JSON lines, so spend survives restarts and concurrent workers can share a
    Volume file. Totals are summed on read, which is fine at hackathon volume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, record: CallRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":")) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def experiment_total(self, experiment_id: str) -> float:
        return sum(r.cost_usd for r in self.records(experiment_id))

    def day_total(self, day: date) -> float:
        totals: dict[date, float] = defaultdict(float)
        for record in self.records():
            totals[record.day] += record.cost_usd
        return totals[day]

    def records(self, experiment_id: str | None = None) -> list[CallRecord]:
        if not self.path.exists():
            return []
        rows = []
        with self._lock, self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(CallRecord(**json.loads(line)))
        return [r for r in rows if experiment_id is None or r.experiment_id == experiment_id]
