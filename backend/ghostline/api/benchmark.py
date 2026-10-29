"""The /benchmark table: ground truth next to what Ghostline measured (build plan Part 25).

Everything the page shows that isn't ground truth is computed here from the experiment's stored
conclusion, so the numbers on the page can't drift from the experiments behind them.
"""

import math
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from ghostline.api.views import SystemName, _system
from ghostline.store.core_schemas import Confidence


class BenchmarkRowView(BaseModel):
    clip_id: str
    experiment_id: UUID | None
    system: SystemName
    object: str
    quantity: str
    unit: str
    truth: float
    truth_sigma: float
    truth_method: str
    measured: float | None
    ci_low: float | None
    ci_high: float | None
    z: float | None  # (measured - truth) / sqrt(sigma_measured^2 + sigma_truth^2)
    covered: bool | None  # the 95% interval contains the truth
    model_selected: str | None
    model_expected: str | None
    model_correct: bool | None
    confidence: Confidence | None
    is_failure_case: bool
    failure_expected: str | None
    failure_flagged: bool | None  # a bad clip is flagged when its confidence comes out low
    note: str | None


class BenchmarkSummaryView(BaseModel):
    n_clips: int
    coverage: float
    model_selection_accuracy: float
    failure_cases_flagged: str
    generated_at: str


class BenchmarkTableView(BaseModel):
    rows: list[BenchmarkRowView]
    summary: BenchmarkSummaryView


def _measured(conclusion: dict[str, Any] | None, quantity: str) -> dict[str, Any] | None:
    for q in (conclusion or {}).get("quantities", []):
        if str(q.get("name", "")).lower() == quantity.lower():
            return q
    return None


def row_view(row: dict[str, Any]) -> BenchmarkRowView:
    failure = bool(row.get("is_failure_case"))
    q = None if failure else _measured(row.get("conclusion"), row["quantity"])
    truth, sigma_t = float(row["truth_value"]), float(row.get("truth_sigma") or 0.0)
    measured = lo = hi = z = covered = None
    if q is not None:
        measured, lo, hi = float(q["value"]), float(q["ci_low"]), float(q["ci_high"])
        sigma_m = (hi - lo) / (2 * 1.96)
        combined = math.hypot(sigma_m, sigma_t)
        z = (measured - truth) / combined if combined > 0 else None
        covered = lo <= truth <= hi
    expected = row.get("expected_model")
    selected = row.get("winner_model_id")
    done = row["status"] == "complete"
    return BenchmarkRowView(
        clip_id=row.get("clip_id") or str(row["experiment_id"])[:8],
        experiment_id=row["experiment_id"],
        system=_system(row["system"]),
        object=row.get("object") or "",
        quantity=row["quantity"],
        unit=row.get("unit") or (q or {}).get("unit", ""),
        truth=truth,
        truth_sigma=sigma_t,
        truth_method=row.get("truth_method") or "",
        measured=measured,
        ci_low=lo,
        ci_high=hi,
        z=z,
        covered=covered,
        model_selected=selected,
        model_expected=expected,
        model_correct=(selected == expected) if expected and done and selected else None,
        confidence=row.get("confidence"),
        is_failure_case=failure,
        failure_expected=row.get("failure_expected"),
        failure_flagged=(row.get("confidence") == "low") if failure and done else None,
        note=row.get("notes") or None,
    )


def benchmark_table(rows: list[dict[str, Any]]) -> BenchmarkTableView:
    views = [row_view(r) for r in rows]
    measured = [r for r in views if r.covered is not None]
    judged = [r for r in views if r.model_correct is not None]
    failures = [r for r in views if r.is_failure_case]
    flagged = [r for r in failures if r.failure_flagged]
    return BenchmarkTableView(
        rows=views,
        summary=BenchmarkSummaryView(
            n_clips=len({r.clip_id for r in views}),
            coverage=(sum(bool(r.covered) for r in measured) / len(measured)) if measured else 0.0,
            model_selection_accuracy=(
                sum(bool(r.model_correct) for r in judged) / len(judged) if judged else 0.0
            ),
            failure_cases_flagged=f"{len(flagged)}/{len(failures)}",
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )
