"""Run a model for the What-if (build plan Part 18): `POST /experiments/{id}/simulate`.

Library models are trusted code and run in this process. A Wonder-written model goes through the
fitting harness's `simulate` mode in the Modal Sandbox, which has no network and its own time
limit; the API gives the whole call 20 s. Either way the answer is the model's observables on a
uniform time grid, plus the readouts the What-if shows where they apply: the period of a swing,
the time a spinner takes to stop, the height of the first rebound, the terminal velocity.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from ghostline.export.demo_math import mean_period, rebound_heights
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import SimulateRequest, SimulateResult
from ghostline.physics.harness.schema import FitJob, FitResult, ModelSpec, SimulateSpec
from ghostline.physics.library import ParamSpec, get_model
from ghostline.physics.sandbox import FitSandbox
from ghostline.store.records import Experiment, ModelRecord

from .context import (
    OBSERVABLE,
    WhatIfRepo,
    fitted_params,
    latest_fit,
    library_ref_of,
    measured_setup,
    motion_series,
    system_of_model,
)

STOP_TOLERANCE = 1e-6  # a spinner has stopped when its angle stays this close to its final value
SANDBOX_EXEC_TIMEOUT_S = 15
STATE_SIZES = (2, 1, 3, 4)  # state vector sizes tried for a Wonder-written model


class Runner(Protocol):
    def run(self, job: FitJob) -> FitResult: ...


class SandboxRunner:
    """Start a Sandbox, run one job in it, and shut it down."""

    def run(self, job: FitJob) -> FitResult:
        with FitSandbox() as sandbox:
            return sandbox.run(job, exec_timeout=SANDBOX_EXEC_TIMEOUT_S)


def _system(experiment: Experiment, model: ModelRecord) -> str:
    fallback = experiment.scene.system_type if experiment.scene else experiment.system_type
    ref = library_ref_of(model)
    return system_of_model(ref, fallback) if ref else fallback


def _setup(repo: WhatIfRepo, experiment: Experiment, system: str) -> dict[str, float]:
    """The constants and initial conditions of trial 1: from the stored What-if context, else
    measured again from the trajectory."""
    stored = experiment.whatif or {}
    if stored.get("system") == system and "fixed" in stored:
        return {**stored["fixed"], **stored["initial"]}
    trials = [t for t in repo.list_trials(experiment.id) if t.role == "fit"]
    rows = repo.list_trajectories([trials[-1].id]) if trials else []
    if not rows:
        return {}
    summary = rows[0].summary or {}
    setup = measured_setup(
        system, summary, trials[-1].calibration or {}, motion_series(summary, None)
    )
    return {**setup["fixed"], **setup["initial"]} if setup else {}


def _values(
    repo: WhatIfRepo, experiment: Experiment, model: ModelRecord, request: SimulateRequest
) -> dict[str, float]:
    """Every number the model needs: its defaults, trial 1's setup, its fitted parameters, then
    the request's overrides."""
    ref = library_ref_of(model)
    specs = get_model(ref, _system(experiment, model)).params if ref else (model.params_spec or {})
    values = {name: float(spec.init) for name, spec in specs.items()}
    values.update(_setup(repo, experiment, _system(experiment, model)))
    values.update(fitted_params(model, latest_fit(repo.list_fits(experiment.id), model.id)))
    values.update(request.params)
    return values


def _library_series(ref: str, system: str, tau: np.ndarray, p: dict[str, float]):
    lib = get_model(ref, system)
    missing = [n for n in lib.constants if n not in p]
    if missing:
        raise StageError(
            f"The clip didn't measure {', '.join(missing)}, which this law needs.",
            code="missing_setup",
        )
    p = lib.prepare(tau, {}, {"direction": 1.0, **p})
    out = np.asarray(lib.predict(tau, p), dtype=float)
    derived = {k: float(v) for k, v in lib.derived(p).items() if math.isfinite(v)}
    return {name: out[i].tolist() for i, name in enumerate(lib.observables)}, derived


def _custom_series(
    runner: Runner, model: ModelRecord, system: str, tau: np.ndarray, p: dict[str, float]
) -> dict[str, list[float]]:
    spec = ModelSpec(
        id=model.id,
        code=model.code,
        name=model.name,
        params={k: ParamSpec(**v.model_dump()) for k, v in (model.params_spec or {}).items()},
        observables=[OBSERVABLE.get(system, "y")],
    )
    # The state vector's size isn't stored with the model, so try the usual sizes in one job.
    specs = [
        spec.model_copy(update={"id": f"{model.id}#{n}", "state": [f"s{i}" for i in range(n)]})
        for n in STATE_SIZES
    ]
    job = FitJob(
        job_id=f"simulate-{model.id}",
        mode="simulate",
        system=system,
        constants={k: v for k, v in p.items() if k not in (spec.params or {})},
        models=specs,
        simulate=SimulateSpec(times=tau.tolist(), params={s.id: p for s in specs}),
        plots=False,
    )
    result = runner.run(job)
    for s in specs:
        if s.id in result.simulations:
            return result.simulations[s.id]
    reports = [r for r in result.validation if not r.ok]
    real = [r for r in reports if not any("returned shape" in e for e in r.errors)]
    errors = (real or reports or [None])[0]
    raise StageError(
        "The model's code didn't pass the checks: "
        + ("; ".join(errors.errors) if errors else "the Sandbox returned no simulation"),
        code="invalid_model",
    )


def stop_time(t: list[float], phi: list[float]) -> float | None:
    """When a spinner's angle stops changing, if it does before the end of the run."""
    final = phi[-1]
    scale = max(1.0, abs(final))
    for i in range(len(phi) - 1, -1, -1):
        if abs(phi[i] - final) > STOP_TOLERANCE * scale:
            return t[i + 1] if i + 1 < len(phi) - 1 else None
    return None


def readouts(system: str, t: list[float], series: dict[str, list[float]]) -> dict[str, float]:
    """Period, time to stop and bounce height, measured on the simulated observable."""
    values = next(iter(series.values()), [])
    out: dict[str, float] = {}
    if system == "pendulum":
        period = mean_period(t, values)
        if period:
            out["period"] = period
    elif system == "spin_down":
        stopped = stop_time(t, values)
        if stopped is not None:
            out["time_to_stop"] = stopped
    elif system == "bounce":
        heights = rebound_heights(values)
        if heights:
            out["bounce_height"] = heights[0]
    return out


def simulate_custom_model(
    repo: WhatIfRepo,
    experiment: Experiment,
    model: ModelRecord,
    request: SimulateRequest,
    *,
    runner: Runner | None = None,
) -> SimulateResult:
    """Simulate `model` on `request.n_points` points up to `request.t_end_s`.

    `request.params` overrides parameters by name, initial conditions included. The model starts
    from trial 1's setup and fitted values unless told otherwise.
    """
    system = _system(experiment, model)
    tau = np.linspace(0.0, request.t_end_s, request.n_points)
    p = _values(repo, experiment, model, request)
    ref = library_ref_of(model)
    if ref:
        series, derived = _library_series(ref, system, tau, p)
    else:
        series, derived = _custom_series(runner or SandboxRunner(), model, system, tau, p), {}
    if not all(math.isfinite(v) for values in series.values() for v in values):
        raise StageError(
            "The simulation produced values that are not numbers; try gentler settings.",
            code="simulation_failed",
        )
    t = tau.tolist()
    derived.update({k: v for k, v in readouts(system, t, series).items() if k not in derived})
    return SimulateResult(model_id=model.id, t=t, series=series, derived=derived)
