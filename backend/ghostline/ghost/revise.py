"""Test a revised model against the first one (build plan Part 17).

The revised model is fitted on trial 1 only, with the Part 12 protocol. Then both models forecast
trial 2, which after the launch window is held-out data for both, and v2 is adopted only if its
forecast is better beyond noise. Physical parameters never come from trial 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ghostline.physics.harness.fitting import block_bootstrap_indices
from ghostline.physics.harness.schema import BootstrapSpec, FitJob, ModelSpec
from ghostline.physics.library import ParamSpec, model_ids
from ghostline.physics.runner import Runner
from ghostline.physics.select import Selection, select_models

from .diagnose import RevisedModel
from .divergence import measure_divergence
from .forecast import forecast_trial, freeze
from .model import Divergence, Forecast, FrozenFit, GhostError
from .series import TrialSeries
from .specs import model_info

N_BOOT = 1000
BLOCK = 10


@dataclass
class RevisionTest:
    spec: ModelSpec
    frozen: FrozenFit
    forecast: Forecast
    div: Divergence
    selection: Selection | None  # v1 against v2 on trial 1's held-out data
    rmse_v1_mm: float
    rmse_v2_mm: float
    diff_ci_mm: tuple[float, float]  # 95% interval of RMSE(v1) - RMSE(v2)
    adopted: bool


def spec_from_reply(
    reply: RevisedModel, system: str, observables: tuple[str, ...], taken: set[str]
) -> ModelSpec:
    """A harness spec from Wonder's revised model. Raises GhostError for an unusable reply."""
    if reply.source == "library":
        if reply.library_ref not in model_ids(system):
            raise GhostError(
                f"{reply.library_ref!r} is not a library model for a {system} system. "
                f"Choose one of {model_ids(system)} or write custom code."
            )
        ident = f"{reply.library_ref}-rev"
        while ident in taken:
            ident += "x"
        return ModelSpec(id=ident, library_ref=reply.library_ref)
    if not reply.code.strip() or not reply.params:
        raise GhostError("A custom model needs code and a parameter list.")
    ident = re.sub(r"[^A-Za-z0-9]", "", reply.id)[:20] or "Custom"
    while ident in taken:
        ident += "x"
    params = {
        p.name: ParamSpec(init=p.init, lower=p.lower, upper=p.upper, unit=p.unit, kind=p.kind)
        for p in reply.params
    }
    return ModelSpec(
        id=ident, code=reply.code, name=reply.name, mechanism=reply.mechanism,
        equations_latex=reply.equations_latex, params=params,
        observables=list(observables), state=reply.state or None,
    )  # fmt: skip


def validate_spec(
    runner: Runner, spec: ModelSpec, system: str, trial1: TrialSeries, constants: dict[str, float]
) -> list[str]:
    """Problems with a proposed model, as the harness reports them (empty when it is fine)."""
    if spec.library_ref is not None:
        return []
    info = model_info(spec, system, ("y",))
    job = FitJob(
        job_id=f"ghost-validate-{spec.id}", mode="validate", system=system,
        data=trial1.series_data(info.observables), constants=constants, models=[spec],
        plots=False,
    )  # fmt: skip
    return [e for v in runner.run(job).validation for e in v.errors]


def compare_on_trial1(
    runner: Runner,
    specs: list[ModelSpec],
    system: str,
    trial1: TrialSeries,
    constants: dict[str, float],
    observables: tuple[str, ...],
) -> Selection | None:
    """Part 12's held-out selection between the models on trial 1 alone. None if it can't run."""
    job = FitJob(
        job_id="ghost-select", mode="select", system=system,
        data=trial1.series_data(observables), constants=constants, models=specs,
        bootstrap=BootstrapSpec(n=0), plots=False, time_limit_s=60.0,
    )  # fmt: skip
    result = runner.run(job)
    ranges = [float(np.ptp(v)) for v in trial1.series_data(observables).series.values() if v]
    try:
        return select_models(
            result, sigma=max(trial1.sigma(o) for o in observables),
            motion_range=max(ranges, default=None),
        )  # fmt: skip
    except Exception:
        return None


def paired_rmse_difference(
    e1: np.ndarray, e2: np.ndarray, n: int = N_BOOT, block: int = BLOCK, seed: int = 0
) -> tuple[float, float, tuple[float, float]]:
    """RMSE of each forecast, and the 95% interval of RMSE(1) - RMSE(2) from a paired block
    bootstrap of the errors (blocks keep the autocorrelation, as in Part 12)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        idx = block_bootstrap_indices(len(e1), block, rng)
        diffs[k] = np.sqrt(np.mean(e1[idx] ** 2)) - np.sqrt(np.mean(e2[idx] ** 2))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.sqrt(np.mean(e1**2))), float(np.sqrt(np.mean(e2**2))), (float(lo), float(hi))


def try_revision(
    runner: Runner,
    v1_spec: ModelSpec,
    v2_spec: ModelSpec,
    system: str,
    constants: dict[str, float],
    trial1: TrialSeries,
    trial2: TrialSeries,
    forecast1: Forecast,
    div1: Divergence,
    *,
    n_draws: int,
) -> RevisionTest:
    """Fit v2 on trial 1, forecast trial 2 with it, and decide whether it is better beyond noise."""
    info = model_info(v2_spec, system, model_info(v1_spec, system, ("y",)).observables)
    selection = compare_on_trial1(
        runner, [v1_spec, v2_spec], system, trial1, constants, info.observables
    )
    frozen2 = freeze(runner, v2_spec, system, trial1, constants)
    forecast2 = forecast_trial(runner, v2_spec, system, frozen2, trial2, n_draws=n_draws)
    div2 = measure_divergence(forecast2, trial2)
    # Compare on the frames after the launch window that both forecasts measured.
    common = np.intersect1d(div1.frame, div2.frame)
    after = common[trial2.timestamps[common] >= forecast1.launch_end_t]
    after = after if len(after) >= 10 else common
    e1 = div1.error_mm[np.isin(div1.frame, after)]
    e2 = div2.error_mm[np.isin(div2.frame, after)]
    r1, r2, (lo, hi) = paired_rmse_difference(e1, e2)
    return RevisionTest(
        spec=v2_spec, frozen=frozen2, forecast=forecast2, div=div2, selection=selection,
        rmse_v1_mm=r1, rmse_v2_mm=r2, diff_ci_mm=(lo, hi), adopted=lo > 0.0,
    )  # fmt: skip
