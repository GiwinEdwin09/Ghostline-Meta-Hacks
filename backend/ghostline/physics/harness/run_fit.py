"""Harness entry point: `python run_fit.py /work/job.json` writes `result.json` next to the job.

The same `run_job` runs in-process for tests and inside the Modal Sandbox (build plan Part 11).
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script (`python /harness/run_fit.py job.json`): import the package properly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ghostline.physics.harness.run_fit import main as _main

    sys.exit(_main(sys.argv))

import concurrent.futures
import multiprocessing
import platform
import time
from contextlib import nullcontext

import numpy as np
import scipy

from ..library import get_model
from ..library.base import ModelDef, ode_tolerance
from . import fitting
from .custom import build_model, dry_run, spec_issues
from .fitting import Dataset, FitTimeout
from .plots import render_plots
from .schema import (
    MAX_RESULT_BYTES,
    FitJob,
    FitResult,
    ModelResult,
    ModelSpec,
    ValidationReport,
)


def _build(spec: ModelSpec, job: FitJob) -> ModelDef:
    default = tuple(job.data.series) if job.data else ("y",)
    return build_model(spec, job.system, default)


def _validate(spec: ModelSpec, job: FitJob) -> list[str]:
    errors = spec_issues(spec)
    if errors:
        return errors
    try:
        model = _build(spec, job)
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"]
    return dry_run(model, job.constants)


def _count_physical(model: ModelDef, fixed: dict[str, float]) -> int:
    return sum(1 for n, s in model.params.items() if s.kind == "physical" and n not in fixed)


def run_model(job: FitJob, index: int, workdir: Path) -> ModelResult:
    """Fit one model according to the job's mode."""
    spec = job.models[index]
    started = time.monotonic()
    deadline = started + job.time_limit_s
    result = ModelResult(id=spec.id, status="ok", library_ref=spec.library_ref)
    errors = _validate(spec, job)
    if errors:
        result.status, result.error = "invalid", "; ".join(errors)
        return result
    try:
        model = _build(spec, job)
        assert job.data is not None
        ds = Dataset.from_series(job.data).for_model(model)
        if ds.n < fitting.MIN_TRAIN + 2:
            raise ValueError(f"only {ds.n} usable points")
        tau = ds.t - ds.t[0]
        constants = model.prepare(tau, ds.obs, job.constants)
        missing = model.missing_constants(constants)
        if missing:
            raise ValueError(f"missing constants {missing}")
        fixed = dict(spec.fixed)
        result.n_physical = _count_physical(model, fixed)
        if job.mode == "select":
            _select(model, ds, constants, fixed, job, deadline, result, workdir)
        else:
            _final(model, ds, constants, fixed, job, deadline, result, workdir)
    except FitTimeout:
        result.status, result.error = "timeout", f"exceeded {job.time_limit_s:g} s"
    except Exception as exc:
        result.status, result.error = "failed", f"{type(exc).__name__}: {exc}"
    result.elapsed_s = time.monotonic() - started
    return result


def _bootstrap(model, ds, constants, fixed, outcome, full, job, deadline, result) -> None:
    spec = job.bootstrap
    boot = fitting.bootstrap_refits(
        model, ds, constants, fixed, outcome,
        n=spec.n, block=spec.block, seed=spec.seed, deadline=deadline,
    )  # fmt: skip
    full.params = fitting.estimates(model, outcome, constants, fixed, boot)
    result.bootstrap_requested, result.bootstrap_done = spec.n, len(boot)


def _select(model, ds, constants, fixed, job, deadline, result, workdir):
    order = [job.primary_split] + [s for s in job.splits if s != job.primary_split]
    warm = None
    primary_params = primary_mask = primary_t = None
    for frac in order:
        first = warm is None
        # The primary split gets the full ODE precision; the robustness splits are warm-started
        # refits and the integrator can be looser there.
        with nullcontext() if first else ode_tolerance(*fitting.REFIT_TOL):
            split_fit, outcome = fitting.fit_split(
                model, ds, constants, fixed, frac,
                n_starts=job.n_starts if first else 1,
                seed=job.bootstrap.seed, deadline=deadline, warm=warm, n_draws=job.forecast_draws,
                diff_step=1e-6 if first else fitting.REFIT_DIFF_STEP,
            )  # fmt: skip
        result.splits.append(split_fit)
        if warm is None:
            primary_params = {**constants, **fixed, **outcome.values}
            primary_mask, _, primary_t = fitting.split_masks(model, ds, frac)
        warm = outcome.values
    with ode_tolerance(*fitting.REFIT_TOL):
        full, outcome = fitting.fit_split(
            model, ds, constants, fixed, None,
            n_starts=1, seed=job.bootstrap.seed, deadline=deadline, warm=warm,
            diff_step=fitting.REFIT_DIFF_STEP,
        )  # fmt: skip
    if job.bootstrap.n > 0:
        _bootstrap(model, ds, constants, fixed, outcome, full, job, deadline, result)
    result.full = full
    chi2 = full.train.chi2
    k = len(outcome.values)
    result.aic = chi2 + 2 * k
    result.bic = chi2 + k * float(np.log(max(ds.n, 2)))
    if job.plots and primary_params is not None:
        result.plots = render_plots(
            model, ds, primary_params, primary_t, primary_mask, workdir / "plots" / model.id
        )


def _final(model, ds, constants, fixed, job, deadline, result, workdir):
    full, outcome = fitting.fit_split(
        model, ds, constants, fixed, None,
        n_starts=job.n_starts, seed=job.bootstrap.seed, deadline=deadline,
    )  # fmt: skip
    # The calibration refits set the headline uncertainty, so they run before the bootstrap
    # can use up the time limit.
    if job.perturbations:
        result.perturbations_requested = len(job.perturbations)
        result.replicates, result.replicate_eps = fitting.perturbed_refits(
            model, ds, constants, fixed, outcome, job.perturbations, deadline
        )
    if job.bootstrap.n > 0:
        _bootstrap(model, ds, constants, fixed, outcome, full, job, deadline, result)
    result.full = full
    result.aic = full.train.chi2 + 2 * len(outcome.values)
    result.bic = full.train.chi2 + len(outcome.values) * float(np.log(max(ds.n, 2)))
    if job.plots:
        p = {**constants, **fixed, **outcome.values}
        mask = np.ones(ds.n, dtype=bool)
        result.plots = render_plots(
            model, ds, p, float(ds.t[-1]), mask, workdir / "plots" / model.id, include_heldout=False
        )


def _parameter_count(spec: ModelSpec) -> int:
    if spec.params is not None:
        return len(spec.params)
    try:
        return len(get_model(spec.library_ref or "").params)
    except KeyError:
        return 0


def _model_task(job_json: str, index: int, workdir: str) -> str:
    job = FitJob.model_validate_json(job_json)
    return run_model(job, index, Path(workdir)).model_dump_json()


def _simulate(job: FitJob, result: FitResult) -> None:
    assert job.simulate is not None
    tau = np.asarray(job.simulate.times, dtype=float)
    for spec in job.models:
        errors = _validate(spec, job)
        if errors:
            result.validation.append(ValidationReport(model_id=spec.id, ok=False, errors=errors))
            continue
        model = _build(spec, job)
        p = {**job.constants, **job.simulate.params.get(spec.id, {})}
        pred = np.asarray(model.predict(tau - tau[0], p), dtype=float)
        result.simulations[spec.id] = {o: pred[i].tolist() for i, o in enumerate(model.observables)}


def _enforce_cap(result: FitResult) -> None:
    """Keep result.json under the size cap by dropping the bulkiest optional fields."""
    if len(result.model_dump_json()) <= MAX_RESULT_BYTES:
        return
    for m in result.models:
        m.replicates, m.replicate_eps = [], []
    if len(result.model_dump_json()) <= MAX_RESULT_BYTES:
        return
    for m in result.models:
        for s in m.splits:
            s.heldout_draws = None


def run_job(job: FitJob, workdir: Path) -> FitResult:
    started = time.monotonic()
    workdir.mkdir(parents=True, exist_ok=True)
    result = FitResult(
        job_id=job.job_id,
        mode=job.mode,
        system=job.system,
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    )
    if job.mode == "validate":
        for spec in job.models:
            errors = _validate(spec, job)
            result.validation.append(
                ValidationReport(model_id=spec.id, ok=not errors, errors=errors)
            )
    elif job.mode == "simulate":
        _simulate(job, result)
    else:
        if job.data is None:
            raise ValueError(f"mode {job.mode!r} needs data")
        payload = job.model_dump_json()
        indices = range(len(job.models))
        if job.workers > 1 and len(job.models) > 1:
            ctx = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
            with concurrent.futures.ProcessPoolExecutor(job.workers, mp_context=ctx) as pool:
                # Start the models with the most parameters first so the pool finishes together.
                by_cost = sorted(indices, key=lambda i: -_parameter_count(job.models[i]))
                futures = {i: pool.submit(_model_task, payload, i, str(workdir)) for i in by_cost}
                result.models = [
                    ModelResult.model_validate_json(futures[i].result()) for i in indices
                ]
        else:
            result.models = [run_model(job, i, workdir) for i in indices]
    result.elapsed_s = time.monotonic() - started
    _enforce_cap(result)
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: run_fit.py JOB_JSON", file=sys.stderr)
        return 2
    job_path = Path(argv[1])
    job = FitJob.model_validate_json(job_path.read_text())
    result = run_job(job, job_path.parent)
    (job_path.parent / "result.json").write_text(result.model_dump_json())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
