"""One call from numpy arrays to a selected law with uncertainty and confidence.

The agent loop (Part 13) can call the pieces separately; `analyze` strings them together:
select on held-out data, refit the winner with bootstrap and calibration refits, build the
uncertainty budget and rate the confidence.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..confidence import (
    ConfidenceAssessment,
    QualityFlag,
    ReviewStatus,
    assess_confidence,
    motion_extent_flags,
)
from .harness.schema import (
    BootstrapSpec,
    FitJob,
    FitResult,
    ModelResult,
    ModelSpec,
    SeriesData,
)
from .library import model_ids
from .runner import InProcessRunner, Runner
from .select import (
    CalibrationBudget,
    FinalQuantity,
    Selection,
    calibration_perturbations,
    final_quantities,
    select_models,
)


class Analysis(BaseModel):
    selection: Selection
    final: ModelResult | None = None
    quantities: list[FinalQuantity] = []
    confidence: ConfidenceAssessment
    select_result: FitResult


def build_series(
    t: np.ndarray,
    observed: dict[str, np.ndarray],
    sigma: float | dict[str, float],
    weights: np.ndarray | None = None,
    segment_id: np.ndarray | None = None,
) -> SeriesData:
    """Plain arrays in, harness data out. NaN values become gaps."""
    sig = {k: float(sigma) for k in observed} if np.isscalar(sigma) else dict(sigma)
    return SeriesData(
        t=np.asarray(t, dtype=float).tolist(),
        series={
            k: [None if not np.isfinite(x) else float(x) for x in np.asarray(v, dtype=float)]
            for k, v in observed.items()
        },
        sigma=sig,
        weights=None if weights is None else np.asarray(weights, dtype=float).tolist(),
        segment_id=None if segment_id is None else np.asarray(segment_id, dtype=int).tolist(),
    )


def analyze(
    t: np.ndarray,
    observed: dict[str, np.ndarray],
    sigma: float | dict[str, float],
    *,
    system: str,
    weights: np.ndarray | None = None,
    segment_id: np.ndarray | None = None,
    constants: dict[str, float] | None = None,
    library_ids: list[str] | None = None,
    custom_models: list[ModelSpec] | None = None,
    budget: CalibrationBudget | None = None,
    flags: list[QualityFlag] | None = None,
    review: ReviewStatus | None = None,
    runner: Runner | None = None,
    n_boot: int = 200,
    n_perturb: int = 50,
    seed: int = 0,
    plots: bool = False,
    job_id: str = "analysis",
) -> Analysis:
    """Select a model on held-out data, then put numbers with uncertainty on the winner."""
    runner = runner or InProcessRunner()
    constants = constants or {}
    budget = budget or CalibrationBudget()
    data = build_series(t, observed, sigma, weights, segment_id)
    specs = [ModelSpec(id=i, library_ref=i) for i in (library_ids or model_ids(system))]
    specs += custom_models or []

    select_job = FitJob(
        job_id=f"{job_id}-select",
        mode="select",
        system=system,
        data=data,
        constants=constants,
        models=specs,
        bootstrap=BootstrapSpec(n=0, seed=seed),
        plots=plots,
    )
    select_result = runner.run(select_job)
    ranges = [float(np.ptp(v[np.isfinite(v)])) for v in observed.values() if np.isfinite(v).any()]
    selection = select_models(
        select_result,
        sigma=max(data.sigma.values()),
        motion_range=max(ranges, default=None),
        seed=seed,
    )

    final = None
    quantities: list[FinalQuantity] = []
    period = None
    if selection.selected_id is not None and selection.score(selection.selected_id).status == "ok":
        winner = next(s for s in specs if s.id == selection.selected_id)
        final_job = FitJob(
            job_id=f"{job_id}-final",
            mode="final",
            system=system,
            data=data,
            constants=constants,
            models=[winner],
            bootstrap=BootstrapSpec(n=n_boot, seed=seed),
            perturbations=calibration_perturbations(budget, constants, n_perturb, seed),
            plots=plots,
            time_limit_s=90.0,
        )
        final = runner.run(final_job).models[0]
        if final.status == "ok":
            tied = [select_result.model(i) for i in selection.tied_with]
            quantities = final_quantities(final, budget, tied)
            period = next(
                (q.value for q in quantities if q.name in ("period_small_angle", "period")), None
            )
    duration = float(np.nanmax(t) - np.nanmin(t))
    all_flags = [*(flags or []), *motion_extent_flags(duration, period)]
    confidence = assess_confidence(selection, system=system, flags=all_flags, review=review)
    return Analysis(
        selection=selection,
        final=final,
        quantities=quantities,
        confidence=confidence,
        select_result=select_result,
    )
