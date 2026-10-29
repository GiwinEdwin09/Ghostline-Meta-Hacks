"""Least-squares fitting, held-out scoring and resampling (build plan Parts 11 and 12).

Everything works on plain numpy arrays so the same code runs in tests and in the Sandbox.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from ..library.base import ModelDef, ParamSpec, clip_spec, ode_tolerance
from ..library.bounce import flight_labels
from .diagnostics import diagnose
from .schema import Diagnostics, ParamEstimate, Perturbation, SeriesData, SplitFit

PENALTY = 1e3
F_SCALE = 2.0
Z_95 = 1.959964
MIN_TRAIN = 8
# Refits and forecast draws loosen the ODE tolerance; the finite-difference step grows with it.
REFIT_TOL = (1e-8, 1e-10)
REFIT_DIFF_STEP = 1e-5


class FitTimeout(Exception):
    """The per-model wall-clock cap was hit."""


@dataclass
class Dataset:
    t: np.ndarray
    obs: dict[str, np.ndarray]
    sigma: dict[str, float]
    weights: np.ndarray
    segment: np.ndarray | None = None

    @classmethod
    def from_series(cls, data: SeriesData) -> Dataset:
        t = np.asarray(data.t, dtype=float)
        obs = {
            k: np.array([np.nan if v is None else v for v in vals], dtype=float)
            for k, vals in data.series.items()
        }
        w = np.ones_like(t) if data.weights is None else np.asarray(data.weights, dtype=float)
        seg = None if data.segment_id is None else np.asarray(data.segment_id, dtype=int)
        keep = np.isfinite(t) & np.isfinite(w) & (w > 0)
        order = np.argsort(t[keep], kind="stable")

        def take(a):
            return a[keep][order]

        return cls(
            t=take(t),
            obs={k: take(v) for k, v in obs.items()},
            sigma=dict(data.sigma),
            weights=take(w),
            segment=None if seg is None else take(seg),
        )

    @property
    def n(self) -> int:
        return int(self.t.size)

    def mask(self, m: np.ndarray) -> Dataset:
        return Dataset(
            t=self.t[m],
            obs={k: v[m] for k, v in self.obs.items()},
            sigma=self.sigma,
            weights=self.weights[m],
            segment=None if self.segment is None else self.segment[m],
        )

    def for_model(self, model: ModelDef) -> Dataset:
        """Keep rows where every observable the model uses is finite."""
        missing = [o for o in model.observables if o not in self.obs]
        if missing:
            raise ValueError(f"data has no series for observables {missing}")
        good = np.all([np.isfinite(self.obs[o]) for o in model.observables], axis=0)
        return self.mask(good)


def apply_perturbation(
    model: ModelDef, ds: Dataset, constants: dict[str, float], pert: Perturbation
) -> tuple[Dataset, dict[str, float]]:
    """Scale lengths and times as a calibration or timing error would."""
    obs = dict(ds.obs)
    sigma = dict(ds.sigma)
    for name in model.observables:
        if model.observable_units.get(name) == "m":
            obs[name] = obs[name] * pert.scale
            sigma[name] = sigma[name] * pert.scale
    out = {k: v * pert.scale if k in model.length_constants else v for k, v in constants.items()}
    out.update(pert.constants)
    shifted = Dataset(ds.t * pert.time_scale, obs, sigma, ds.weights, ds.segment)
    return shifted, out


def split_masks(model: ModelDef, ds: Dataset, frac: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Chronological train/test masks. Bounces train on the first k flights and test the next."""
    if model.by_flight:
        labels = flight_labels(ds.t, ds.obs[model.observables[0]], ds.segment)
        flights = int(labels.max()) + 1
        if flights >= 3:
            k = int(np.clip(round(frac * flights), 2, flights - 1))
            train, test = labels < k, labels == k
            return train, test, float(ds.t[train][-1])
    t_split = float(ds.t[0] + frac * (ds.t[-1] - ds.t[0]))
    train = ds.t <= t_split
    return train, ~train, t_split


@dataclass
class FitOutcome:
    values: dict[str, float]
    specs: dict[str, ParamSpec]
    cov: np.ndarray | None
    condition: float | None
    cost: float
    converged: bool
    nfev: int
    at_bound: dict[str, bool] = field(default_factory=dict)
    timed_out: bool = False
    cov_forecast: np.ndarray | None = None


class Problem:
    """Residuals of one model against one dataset, in normalized parameter coordinates."""

    def __init__(
        self,
        model: ModelDef,
        ds: Dataset,
        constants: dict[str, float],
        specs: dict[str, ParamSpec],
        fixed: dict[str, float],
        t0: float,
        deadline: float | None = None,
    ):
        self.model = model
        self.constants = constants
        self.fixed = dict(fixed)
        self.names = [n for n in specs if n not in self.fixed]
        self.specs = {n: specs[n] for n in self.names}
        self.tau = ds.t - t0
        self.obs = np.vstack([ds.obs[o] for o in model.observables])
        self.inv_sigma = (1.0 / np.array([ds.sigma[o] for o in model.observables]))[:, None]
        self.sqrt_w = np.sqrt(ds.weights)[None, :]
        self.deadline = deadline
        lower = np.array([self.specs[n].lower for n in self.names])
        upper = np.array([self.specs[n].upper for n in self.names])
        self.ref = np.array([self.specs[n].init for n in self.names])
        self.scale = np.maximum(np.abs(self.ref), 0.05 * (upper - lower))
        self.scale[self.scale <= 0] = 1.0
        self.zl = (lower - self.ref) / self.scale
        self.zu = (upper - self.ref) / self.scale
        self.exact = self._closed_form_available()
        self.best_cost = np.inf
        self.best_z = np.zeros(len(self.names))

    def _closed_form_available(self) -> bool:
        probe_tau = self.tau[: min(self.tau.size, 4)]
        try:
            probe = self.model.closed_form(probe_tau, self.params(np.zeros(len(self.names))))
        except Exception:
            return False
        return probe is not None

    def point(self, z: np.ndarray) -> dict[str, float]:
        return dict(zip(self.names, (self.ref + np.asarray(z) * self.scale).tolist(), strict=True))

    def params(self, z: np.ndarray) -> dict[str, float]:
        return {**self.constants, **self.fixed, **self.point(z)}

    def normalized(self, z: np.ndarray) -> np.ndarray:
        try:
            pred = self.model.predict(self.tau, self.params(z))
            pred = np.asarray(pred, dtype=float)
            if pred.shape != self.obs.shape:
                raise ValueError("prediction has the wrong shape")
        except TimeoutError:
            raise
        except Exception:
            return np.full(self.obs.shape, PENALTY)
        r = (self.obs - pred) * self.inv_sigma * self.sqrt_w
        return np.where(np.isfinite(r), np.clip(r, -PENALTY, PENALTY), PENALTY)

    def residuals(self, z: np.ndarray) -> np.ndarray:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise FitTimeout
        r = self.normalized(z).ravel()
        cost = float(np.sum(2 * F_SCALE**2 * (np.sqrt(1 + (r / F_SCALE) ** 2) - 1)))
        if cost < self.best_cost:
            self.best_cost, self.best_z = cost, np.array(z, dtype=float)
        return r


def _starts(problem: Problem, n_starts: int, seed: int) -> list[np.ndarray]:
    starts = [np.zeros(len(problem.names))]
    if n_starts > 1 and problem.names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = qmc.Sobol(d=len(problem.names), scramble=True, seed=seed).random(n_starts - 1)
        spread = 0.15 * (problem.zu - problem.zl)
        for row in u:
            starts.append(np.clip((row - 0.5) * 2 * spread, problem.zl, problem.zu))
    return starts


def _solve(problem: Problem, z0: np.ndarray, max_nfev: int, diff_step: float = 1e-6):
    # Closed forms can be fit to machine precision; ODE fits are limited by the integrator.
    tol = 1e-13 if problem.exact else 1e-11
    return least_squares(
        problem.residuals,
        z0,
        bounds=(problem.zl, problem.zu),
        method="trf",
        loss="soft_l1",
        f_scale=F_SCALE,
        x_scale="jac",
        diff_step=diff_step,
        max_nfev=max_nfev,
        ftol=tol,
        xtol=tol,
        gtol=tol,
    )


def _covariance(problem: Problem, z: np.ndarray):
    """Gauss-Newton covariance s^2 (J'J)^-1 in parameter units, with J by central differences.

    The Jacobian that least_squares returns is stale and, with a robust loss, reweighted, so it is
    recomputed here at the solution. A weak prior (sd = a quarter of the bounds) keeps directions
    the data cannot constrain from looking precise.
    """
    n = len(z)
    none = (None, None, None)
    h = 1e-4
    jac = np.empty((problem.obs.size, n))
    for k in range(n):
        up, down = z.copy(), z.copy()
        up[k] = min(z[k] + h, problem.zu[k])
        down[k] = max(z[k] - h, problem.zl[k])
        width = up[k] - down[k]
        if width <= 0:
            return none
        jac[:, k] = (problem.normalized(up).ravel() - problem.normalized(down).ravel()) / width
    jac = jac / problem.scale
    if not np.all(np.isfinite(jac)):
        return none
    resid = problem.normalized(z).ravel()
    prior = 1.0 / ((problem.zu - problem.zl) * problem.scale / 4.0) ** 2
    try:
        cov = np.linalg.inv(jac.T @ jac + np.diag(prior))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(jac.T @ jac + np.diag(prior))
    inflate = max(1.0, float(np.sum(resid**2)) / max(resid.size - n, 1))
    # For forecasts, directions the data cannot see do not move the prediction: no prior there.
    forecast = np.linalg.pinv(jac.T @ jac, rcond=1e-10, hermitian=True) * inflate
    norms = np.linalg.norm(jac, axis=0)
    norms[norms == 0] = 1.0
    return cov * inflate, float(np.linalg.cond(jac / norms)), forecast


def fit(
    problem: Problem,
    n_starts: int = 1,
    seed: int = 0,
    max_nfev: int = 200,
    diff_step: float = 1e-6,
    want_cov: bool = True,
) -> FitOutcome:
    """Fit from the heuristic guess; with several starts, screen Sobol points, refine the best."""
    if not problem.names:
        problem.residuals(np.zeros(0))
        return FitOutcome({}, {}, None, None, problem.best_cost, True, 1)
    starts = _starts(problem, n_starts, seed)
    best = None
    timed_out = False
    try:
        screened = [(problem.residuals(z0), z0) for z0 in starts[1:]]
        screened.sort(key=lambda item: float(np.sum(item[0] ** 2)))
        chosen = [starts[0]] + [z0 for _, z0 in screened[:2]]
        for i, z0 in enumerate(chosen):
            res = _solve(problem, z0, max_nfev if i == 0 else max_nfev // 2, diff_step)
            if best is None or res.cost < best.cost:
                best = res
    except FitTimeout:
        timed_out = True
    if best is None:
        z = problem.best_z
        values = problem.point(z)
        return FitOutcome(
            values, problem.specs, None, None, problem.best_cost, False, 0, {}, timed_out
        )
    cov, cond, cov_fc = (
        _covariance(problem, best.x) if want_cov and not timed_out else (None, None, None)
    )
    z = best.x
    span = problem.zu - problem.zl
    at_bound = {
        n: bool(min(z[i] - problem.zl[i], problem.zu[i] - z[i]) < 1e-6 * span[i])
        for i, n in enumerate(problem.names)
    }
    return FitOutcome(
        values=problem.point(z),
        specs=problem.specs,
        cov=cov,
        condition=cond,
        cost=float(best.cost),
        converged=bool(best.success) and not timed_out,
        nfev=int(best.nfev),
        at_bound=at_bound,
        timed_out=timed_out,
        cov_forecast=cov_fc,
    )


def warm_specs(specs: dict[str, ParamSpec], values: dict[str, float]) -> dict[str, ParamSpec]:
    out = {}
    for name, spec in specs.items():
        init = values.get(name, spec.init)
        out[name] = clip_spec(init, spec.lower, spec.upper, spec.unit, spec.kind)
    return out


def predict_all(model: ModelDef, tau: np.ndarray, p: dict[str, float]) -> np.ndarray:
    try:
        out = np.asarray(model.predict(tau, p), dtype=float)
    except TimeoutError:
        raise
    except Exception:
        return np.full((len(model.observables), tau.size), np.nan)
    return out


def heldout_errors(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Euclidean error per point across observables; failed predictions count as a huge miss."""
    err = np.sqrt(np.sum((obs - pred) ** 2, axis=0))
    cap = 10 * max(float(np.max(np.ptp(obs, axis=1))), 1e-9)
    return np.where(np.isfinite(err), np.minimum(err, cap), cap)


def derived_values(model: ModelDef, p: dict[str, float]) -> dict[str, float]:
    try:
        return {k: float(v) for k, v in model.derived(p).items() if np.isfinite(v)}
    except Exception:
        return {}


def estimates(
    model: ModelDef,
    outcome: FitOutcome,
    constants: dict[str, float],
    fixed: dict[str, float],
    boot: list[dict[str, float]] | None = None,
) -> list[ParamEstimate]:
    """Parameter and derived-quantity estimates with Jacobian or bootstrap intervals."""
    names = list(outcome.values)
    values = outcome.values
    p_all = {**constants, **fixed, **values}
    out: list[ParamEstimate] = []
    se = {}
    if outcome.cov is not None:
        z_se = np.sqrt(np.clip(np.diag(outcome.cov), 0, None))
        se = dict(zip(names, z_se.tolist(), strict=True))
    for name in names:
        spec = outcome.specs[name]
        lo = hi = None
        if boot:
            sample = np.array([b[name] for b in boot if name in b])
            if sample.size >= 20:
                lo, hi = (float(x) for x in np.percentile(sample, [2.5, 97.5]))
        if lo is None and name in se:
            lo, hi = values[name] - Z_95 * se[name], values[name] + Z_95 * se[name]
        out.append(
            ParamEstimate(
                name=name,
                value=float(values[name]),
                se=se.get(name),
                ci_low=lo,
                ci_high=hi,
                unit=spec.unit,
                kind=spec.kind,
                at_bound=outcome.at_bound.get(name, False),
            )
        )
    units = model.derived_units()
    for name, value in derived_values(model, p_all).items():
        lo = hi = d_se = None
        if boot:
            sample = np.array([b[name] for b in boot if name in b])
            if sample.size >= 20:
                lo, hi = (float(x) for x in np.percentile(sample, [2.5, 97.5]))
        if lo is None and outcome.cov is not None:
            grad = _derived_gradient(model, outcome, p_all, name)
            if grad is not None:
                d_se = float(np.sqrt(max(grad @ outcome.cov @ grad, 0.0)))
                lo, hi = value - Z_95 * d_se, value + Z_95 * d_se
        out.append(
            ParamEstimate(
                name=name,
                value=value,
                se=d_se,
                ci_low=lo,
                ci_high=hi,
                unit=units.get(name, ""),
                kind="physical",
                derived=True,
            )
        )
    return out


def _derived_gradient(model, outcome, p_all, name) -> np.ndarray | None:
    grad = np.zeros(len(outcome.values))
    for i, key in enumerate(outcome.values):
        step = 1e-5 * max(abs(p_all[key]), 1e-3)
        hi = derived_values(model, {**p_all, key: p_all[key] + step}).get(name)
        lo = derived_values(model, {**p_all, key: p_all[key] - step}).get(name)
        if hi is None or lo is None:
            return None
        grad[i] = (hi - lo) / (2 * step)
    return grad


def fit_split(
    model: ModelDef,
    ds: Dataset,
    constants: dict[str, float],
    fixed: dict[str, float],
    frac: float | None,
    *,
    n_starts: int,
    seed: int,
    deadline: float | None,
    warm: dict[str, float] | None = None,
    boot: list[dict[str, float]] | None = None,
    n_draws: int = 0,
) -> tuple[SplitFit, FitOutcome]:
    """Fit on the training part of `ds` (all of it when frac is None) and score the rest."""
    t0 = float(ds.t[0])
    if frac is None:
        train = np.ones(ds.n, dtype=bool)
        test = np.zeros(ds.n, dtype=bool)
        t_split = float(ds.t[-1])
    else:
        train, test, t_split = split_masks(model, ds, frac)
    if train.sum() < MIN_TRAIN:
        raise ValueError(f"only {int(train.sum())} training points; need at least {MIN_TRAIN}")
    tr = ds.mask(train)
    tau_tr = tr.t - t0
    specs = model.fit_setup(tau_tr, tr.obs, constants, tr.segment)
    if warm:
        specs = warm_specs(specs, warm)
    problem = Problem(model, tr, constants, specs, fixed, t0, deadline)
    outcome = fit(problem, n_starts=n_starts, seed=seed)
    if outcome.timed_out:
        raise FitTimeout

    p_all = {**constants, **fixed, **outcome.values}
    tau_all = ds.t - t0
    pred = predict_all(model, tau_all, p_all)
    obs_all = np.vstack([ds.obs[o] for o in model.observables])
    inv_sigma = (1.0 / np.array([ds.sigma[o] for o in model.observables]))[:, None]
    norm_tr = (obs_all - pred)[:, train] * inv_sigma * np.sqrt(tr.weights)[None, :]
    norm_tr = np.where(np.isfinite(norm_tr), norm_tr, PENALTY)
    raw_rmse = float(np.sqrt(np.mean(heldout_errors(obs_all[:, train], pred[:, train]) ** 2)))
    diag: Diagnostics = diagnose(tau_tr, norm_tr, raw_rmse, len(outcome.values), outcome.condition)

    heldout = None
    errors = t_test = draws = None
    if test.any():
        errors_arr = heldout_errors(obs_all[:, test], pred[:, test])
        heldout = float(np.sqrt(np.mean(errors_arr**2)))
        errors = errors_arr.tolist()
        t_test = (ds.t[test] - t_split).tolist()
        if n_draws > 0:
            draws = forecast_draws(
                model, outcome, constants, fixed, tau_all, obs_all[:, test], test, n_draws, seed
            )
    split_fit = SplitFit(
        split=1.0 if frac is None else frac,
        t_split=t_split,
        n_train=int(train.sum()),
        n_test=int(test.sum()),
        converged=outcome.converged,
        params=estimates(model, outcome, constants, fixed, boot),
        train=diag,
        heldout_rmse=heldout,
        heldout_errors=errors,
        heldout_draws=draws,
        t_test=t_test,
    )
    return split_fit, outcome


def forecast_draws(
    model: ModelDef,
    outcome: FitOutcome,
    constants: dict[str, float],
    fixed: dict[str, float],
    tau: np.ndarray,
    obs_test: np.ndarray,
    test: np.ndarray,
    n: int,
    seed: int,
) -> list[list[float]] | None:
    """Held-out errors when the parameters are drawn from the fit's covariance.

    The realized forecast error depends on one noisy training fit. Drawing parameters around the
    fit shows how much of a gap between models could be training noise rather than physics.
    """
    if outcome.cov_forecast is None or not outcome.values:
        return None
    names = list(outcome.values)
    mean = np.array([outcome.values[k] for k in names])
    w, v = np.linalg.eigh(0.5 * (outcome.cov_forecast + outcome.cov_forecast.T))
    root = v * np.sqrt(np.clip(w, 0.0, None))
    lower = np.array([outcome.specs[k].lower for k in names])
    upper = np.array([outcome.specs[k].upper for k in names])
    rng = np.random.default_rng(seed + 17)
    rows = []
    with ode_tolerance(*REFIT_TOL):
        for _ in range(n):
            theta = np.clip(mean + root @ rng.standard_normal(len(names)), lower, upper)
            p_all = {**constants, **fixed, **dict(zip(names, theta.tolist(), strict=True))}
            pred = predict_all(model, tau, p_all)
            rows.append(np.round(heldout_errors(obs_test, pred[:, test]), 8).tolist())
    return rows


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n))
    count = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=count)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap_refits(
    model: ModelDef,
    ds: Dataset,
    constants: dict[str, float],
    fixed: dict[str, float],
    outcome: FitOutcome,
    *,
    n: int,
    block: int,
    seed: int,
    deadline: float | None,
) -> list[dict[str, float]]:
    """Block residual bootstrap: refit the fit plus resampled blocks of its residuals."""
    t0 = float(ds.t[0])
    tau = ds.t - t0
    p_all = {**constants, **fixed, **outcome.values}
    pred = predict_all(model, tau, p_all)
    obs = np.vstack([ds.obs[o] for o in model.observables])
    resid = obs - pred
    if not np.all(np.isfinite(resid)):
        return []
    rng = np.random.default_rng(seed)
    specs = warm_specs(outcome.specs, outcome.values)
    draws: list[dict[str, float]] = []
    with ode_tolerance(*REFIT_TOL):
        for _ in range(n):
            if deadline is not None and time.monotonic() > deadline:
                break
            idx = block_bootstrap_indices(ds.n, block, rng)
            fake = Dataset(
                ds.t,
                {o: pred[i] + resid[i, idx] for i, o in enumerate(model.observables)},
                ds.sigma,
                ds.weights,
                ds.segment,
            )
            problem = Problem(model, fake, constants, specs, fixed, t0, deadline)
            res = fit(problem, n_starts=1, max_nfev=40, diff_step=REFIT_DIFF_STEP, want_cov=False)
            if res.timed_out:
                break
            if res.nfev > 0:
                draws.append(
                    {**res.values, **derived_values(model, {**constants, **fixed, **res.values})}
                )
    return draws


def perturbed_refits(
    model: ModelDef,
    ds: Dataset,
    constants: dict[str, float],
    fixed: dict[str, float],
    outcome: FitOutcome,
    perturbations: list[Perturbation],
    deadline: float | None,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Refit the final model under each calibration/timing draw. Returns values and their eps."""
    specs = warm_specs(outcome.specs, outcome.values)
    reps: list[dict[str, float]] = []
    eps: list[dict[str, float]] = []
    with ode_tolerance(*REFIT_TOL):
        for pert in perturbations:
            if deadline is not None and time.monotonic() > deadline:
                break
            ds_p, const_p = apply_perturbation(model, ds, constants, pert)
            problem = Problem(model, ds_p, const_p, specs, fixed, float(ds_p.t[0]), deadline)
            res = fit(problem, n_starts=1, max_nfev=60, diff_step=REFIT_DIFF_STEP, want_cov=False)
            if res.timed_out:
                break
            reps.append({**res.values, **derived_values(model, {**const_p, **fixed, **res.values})})
            eps.append(dict(pert.eps))
    return reps, eps
