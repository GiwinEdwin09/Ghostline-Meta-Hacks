"""Freeze trial 1's law and forecast trial 2 with it (build plan Part 16).

    frozen = freeze(runner, spec, system, trial1, constants)
    forecast = forecast_trial(runner, spec, system, frozen, trial2)

`freeze` is the final fit of Part 12 on trial 1. `forecast_trial` holds every physical parameter
at trial 1's value and fits only the initial conditions to trial 2's launch window; everything
after the window is a pure prediction. 200 Monte Carlo draws over the parameter errors and the
initial-condition errors give the spread of the prediction at each frame.

All fitting and simulation go through a `Runner` (the Sandbox in production), so Wonder-written
model code is never run in this process.
"""

from __future__ import annotations

import numpy as np

from ghostline.physics.harness.fitting import Dataset, Problem, fit
from ghostline.physics.harness.schema import (
    BootstrapSpec,
    FitJob,
    FitResult,
    ModelResult,
    ModelSpec,
    SimulateSpec,
)
from ghostline.physics.library import ParamSpec
from ghostline.physics.library.base import clip_spec, poly_start
from ghostline.physics.runner import Runner

from .geometry import Geometry, make_geometry
from .model import N_DRAWS, Forecast, FrozenFit, GhostError, ICFit
from .series import TrialSeries
from .specs import ModelInfo, model_info, with_observables

Z_95 = 1.959964
MIN_LAUNCH_POINTS = 8
MAX_DRAW_GRID = 150  # times at which the Monte Carlo draws are evaluated; the spread is smooth


def _one(result: FitResult, what: str) -> ModelResult:
    model = result.models[0]
    if model.status != "ok" or model.full is None:
        raise GhostError(f"{what} failed: {model.error or model.status}")
    return model


def freeze(
    runner: Runner,
    spec: ModelSpec,
    system: str,
    trial1: TrialSeries,
    constants: dict[str, float],
    *,
    seed: int = 0,
) -> FrozenFit:
    """Trial 1's final fit of the model on the whole motion window.

    The standard errors are the statistical ones (the Jacobian covariance of the fit). The
    calibration part of the budget of Part 12 is a common scale on both trials, so it moves the
    ghost and the real object together and does not belong in the spread of the prediction.
    """
    info = model_info(spec, system, ("y",))
    job = FitJob(
        job_id=f"ghost-freeze-{spec.id}",
        mode="final",
        system=system,
        data=trial1.series_data(info.observables),
        constants=constants,
        models=[spec],
        bootstrap=BootstrapSpec(n=0, seed=seed),
        n_starts=8,
        plots=False,
        time_limit_s=90.0,
    )
    model = _one(runner.run(job), f"Refitting {spec.id} on the first clip")
    assert model.full is not None
    values: dict[str, float] = {}
    sigma: dict[str, float] = {}
    initial: dict[str, float] = {}
    derived: dict[str, float] = {}
    for p in model.full.params:
        if p.derived:
            derived[p.name] = p.value
        elif p.kind == "initial_condition":
            initial[p.name] = p.value
        else:
            values[p.name] = p.value
            if p.kind == "physical":
                sigma[p.name] = _standard_error(p)
    return FrozenFit(
        model_id=spec.id,
        values=values,
        sigma=sigma,
        constants=dict(constants),
        initial=initial,
        derived=derived,
        train_rmse=model.full.train.rmse,
    )


def _standard_error(p) -> float:
    if p.se is not None and np.isfinite(p.se):
        return float(p.se)
    if p.ci_low is not None and p.ci_high is not None:
        return float((p.ci_high - p.ci_low) / (2 * Z_95))
    return 0.0


def prepared_constants(
    info: ModelInfo, series: TrialSeries, rows, constants: dict[str, float]
) -> dict[str, float]:
    """The constants as the harness sees them after the model's `prepare` step, which fills in
    data-derived defaults such as a spinner's direction. The simulate mode doesn't call it."""
    if info.instance is None:
        return dict(constants)
    data = series.series_data(info.observables, rows)
    t = np.asarray(data.t, float)
    obs = {
        k: np.array([np.nan if v is None else v for v in vals]) for k, vals in data.series.items()
    }
    return info.instance.prepare(t - t[0], obs, constants)


def fit_initial_conditions(
    runner: Runner,
    spec: ModelSpec,
    system: str,
    frozen: FrozenFit,
    trial2: TrialSeries,
    info: ModelInfo,
) -> ICFit:
    """Fit only the initial conditions, to trial 2's launch window, with the law held fixed.

    Library models are fitted in this process with starting points of our own: the harness's
    data-driven guesses (a frequency from the zero crossings, say) assume a window of at least a
    swing or two, and a launch window can be much shorter. Wonder-written models go through the
    runner, which is the Sandbox.
    """
    missing = [n for n in info.frozen_names if n not in frozen.values]
    if missing:
        raise GhostError(f"The first clip's fit has no value for {missing}.")
    fixed = {n: frozen.values[n] for n in info.frozen_names}
    if info.instance is not None:
        return _fit_library(info, fixed, frozen, trial2)
    pinned = with_observables(spec, info.observables).model_copy(update={"fixed": fixed})
    job = FitJob(
        job_id=f"ghost-launch-{spec.id}",
        mode="final",
        system=system,
        data=trial2.series_data(info.observables, trial2.launch_rows()),
        constants=frozen.constants,
        models=[pinned],
        bootstrap=BootstrapSpec(n=0),
        n_starts=4,
        plots=False,
        time_limit_s=60.0,
    )
    model = _one(runner.run(job), f"Fitting {spec.id} to the launch window")
    assert model.full is not None
    free = [p for p in model.full.params if not p.derived]
    return ICFit(
        values={p.name: p.value for p in free},
        se={p.name: _standard_error(p) for p in free},
        at_bound=[p.name for p in free if p.at_bound],
        rmse=model.full.train.rmse,
        n_points=model.full.n_train,
    )


def _launch_specs(model, ds: Dataset, constants, tau: np.ndarray) -> dict[str, ParamSpec]:
    """Parameter specs for the launch fit. The model's own guesses when they work on this short
    window; otherwise its defaults, with the start and rate taken from a quadratic through the
    launch data."""
    try:
        return model.fit_setup(tau, ds.obs, constants, ds.segment)
    except Exception:
        specs = {n: s.model_copy() for n, s in model.params.items()}
        names = [n for n, s in specs.items() if s.kind == "initial_condition"]
        y = ds.obs[model.observables[0]]
        c = poly_start(tau, y, 2)
        for name, guess in zip(names[:2], c[:2], strict=False):
            s = specs[name]
            specs[name] = clip_spec(guess, s.lower, s.upper, s.unit, s.kind)
        return specs


def _fit_library(
    info: ModelInfo, fixed: dict[str, float], frozen: FrozenFit, trial2: TrialSeries
) -> ICFit:
    model = info.instance
    assert model is not None
    ds = Dataset.from_series(trial2.series_data(info.observables, trial2.launch_rows()))
    ds = ds.for_model(model)
    if ds.n < MIN_LAUNCH_POINTS:
        raise GhostError(
            f"The launch window has only {ds.n} usable points; the second clip is too short "
            "or too poorly tracked to start the ghost."
        )
    t0 = float(ds.t[0])
    tau = ds.t - t0
    constants = model.prepare(tau, ds.obs, frozen.constants)
    specs = _launch_specs(model, ds, constants, tau)
    problem = Problem(model, ds, constants, specs, fixed, t0)
    outcome = fit(problem, n_starts=4, seed=0)
    if not outcome.values:
        raise GhostError(f"{info.id} has no initial conditions to fit.")
    names = list(outcome.values)
    cov = outcome.cov_forecast if outcome.cov_forecast is not None else outcome.cov
    se = np.sqrt(np.clip(np.diag(outcome.cov), 0, None)) if outcome.cov is not None else None
    p = {**constants, **fixed, **outcome.values}
    pred = np.asarray(model.predict(tau, p), float)
    obs = np.vstack([ds.obs[o] for o in model.observables])
    return ICFit(
        values=dict(outcome.values),
        se={n: float(se[i]) if se is not None else 0.0 for i, n in enumerate(names)},
        at_bound=[n for n, b in outcome.at_bound.items() if b],
        rmse=float(np.sqrt(np.mean((obs - pred) ** 2))),
        n_points=ds.n,
        names=names,
        cov=cov,
    )


def _simulate(
    runner: Runner,
    spec: ModelSpec,
    system: str,
    constants: dict[str, float],
    times: np.ndarray,
    param_sets: list[dict[str, float]],
) -> list[np.ndarray]:
    """Predictions (n_obs, n_times) for each parameter set, in one harness job."""
    specs = [spec.model_copy(update={"id": f"s{i}", "fixed": {}}) for i in range(len(param_sets))]
    job = FitJob(
        job_id=f"ghost-simulate-{spec.id}",
        mode="simulate",
        system=system,
        constants=constants,
        models=specs,
        simulate=SimulateSpec(
            times=times.tolist(), params={s.id: p for s, p in zip(specs, param_sets, strict=True)}
        ),
        plots=False,
    )
    result = runner.run(job)
    bad = [v for v in result.validation if not v.ok]
    if bad:
        raise GhostError(f"Model {spec.id} cannot be simulated: {'; '.join(bad[0].errors)}")
    out = []
    for s in specs:
        sim = result.simulations.get(s.id)
        if sim is None:
            raise GhostError(f"Model {spec.id} returned no prediction.")
        out.append(np.array([sim[k] for k in sim], dtype=float))
    return out


def _clip(value: float, spec) -> float:
    return float(np.clip(value, spec.lower, spec.upper)) if spec is not None else float(value)


def forecast_trial(
    runner: Runner,
    spec: ModelSpec,
    system: str,
    frozen: FrozenFit,
    trial2: TrialSeries,
    *,
    n_draws: int = N_DRAWS,
    seed: int = 0,
) -> Forecast:
    """Fit the initial conditions on trial 2's launch window and predict the rest of its motion."""
    first = trial2.motion
    info = model_info(spec, system, tuple(c for c in ("y",) if c in first))
    spec = with_observables(spec, info.observables)
    geometry = make_geometry(info.observables, trial2)

    ic = fit_initial_conditions(runner, spec, system, frozen, trial2, info)
    launch = trial2.launch_rows()
    t_fit0 = float(launch["t"].iloc[0])
    f0, f1 = trial2.frame_range
    frames = np.arange(f0, f1 + 1)
    t = trial2.timestamps[frames]
    keep = t >= t_fit0 - 1e-9
    frames, t = frames[keep], t[keep]
    times = np.concatenate([[t_fit0], t[t > t_fit0 + 1e-9]])

    constants = prepared_constants(info, trial2, launch, frozen.constants)
    params = {**frozen.values, **ic.values}
    mean = _simulate(runner, spec, system, constants, times, [params])[0]
    if mean.shape[1] != len(times) or not np.all(np.isfinite(mean)):
        raise GhostError(f"The prediction of {spec.id} is not finite.")
    obs = _align(mean, times, t)

    xy = geometry.position(obs, t)
    spread = _spread(runner, spec, system, constants, info, frozen, ic, geometry, t, n_draws, seed)
    return Forecast(
        model_id=spec.id,
        frames=frames,
        t=t,
        obs=obs,
        xy=xy,
        angle=geometry.angle(obs),
        anchor_xy=geometry.anchor,
        spread_mm=spread,
        ic=ic,
        launch_end_t=float(launch["t"].iloc[-1]),
        launch_window_s=float(launch["t"].iloc[-1] - launch["t"].iloc[0]),
        params=params,
        n_draws=n_draws,
    )


def _align(obs: np.ndarray, times: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Predictions at `times` onto the frame times `t` (they agree except that `times` starts
    at the fit's first sample)."""
    if len(times) == len(t) and np.allclose(times, t):
        return obs
    return np.array([np.interp(t, times, row) for row in obs])


def _spread(
    runner: Runner,
    spec: ModelSpec,
    system: str,
    constants: dict[str, float],
    info: ModelInfo,
    frozen: FrozenFit,
    ic: ICFit,
    geometry: Geometry,
    t: np.ndarray,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """One-sigma spread in millimetres of the predicted position across Monte Carlo draws of the
    physical parameters (their statistical errors) and the initial conditions (the launch fit)."""
    if n_draws < 2 or len(t) < 2:
        return np.zeros(len(t))
    rng = np.random.default_rng(seed)
    grid = t if len(t) <= MAX_DRAW_GRID else np.linspace(t[0], t[-1], MAX_DRAW_GRID)
    sets = []
    for _ in range(n_draws):
        p = dict(frozen.values)
        for name, sigma in frozen.sigma.items():
            if name in p:
                p[name] = _clip(p[name] + sigma * rng.standard_normal(), info.params.get(name))
        for name, value in ic_draw(ic, rng).items():
            p[name] = _clip(value, info.params.get(name))
        sets.append(p)
    try:
        sims = _simulate(runner, spec, system, constants, grid, sets)
    except GhostError:
        return np.zeros(len(t))
    pos = np.array([geometry.position(s, grid) for s in sims if np.all(np.isfinite(s))])
    if len(pos) < 2:
        return np.zeros(len(t))
    var = pos.var(axis=0, ddof=1).sum(axis=1)  # variance of x plus variance of y
    spread = 1000.0 * np.sqrt(var)
    return np.interp(t, grid, spread)


def ic_draw(ic: ICFit, rng: np.random.Generator) -> dict[str, float]:
    """One draw of the initial conditions: jointly normal when the fit kept its covariance (the
    start and the rate are strongly correlated over a short window), else independent."""
    names = list(ic.values)
    mean = np.array([ic.values[n] for n in names])
    if ic.cov is not None and ic.cov.shape == (len(names), len(names)):
        w, v = np.linalg.eigh(0.5 * (ic.cov + ic.cov.T))
        draw = mean + (v * np.sqrt(np.clip(w, 0.0, None))) @ rng.standard_normal(len(names))
    else:
        draw = mean + np.array([ic.se.get(n, 0.0) for n in names]) * rng.standard_normal(len(names))
    return dict(zip(names, draw.tolist(), strict=True))
