"""Re-running one fit with an edited split and initial guesses (Parts 11-12 and 20).

The notebook's Re-run button on a fit event sends a split percentage and, optionally, initial
guesses. The model is fitted again on that split of the same motion window the analysis used, with
the same harness and the same data, and the new `FitRecord` is returned for the handler to insert.

Initial guesses have no field in a harness job, so a library model that has guesses is fitted
directly with `fitting.fit_split(..., warm=guesses)`; that is trusted library code, run in this
process. A custom model never runs in this process: it goes through the runner (a Sandbox in
production) and its guesses are ignored, with a note in the record.
"""

from __future__ import annotations

import shutil
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.physics.harness import fitting
from ghostline.physics.harness.plots import render_plots
from ghostline.physics.harness.schema import BootstrapSpec, FitJob, ModelSpec, SplitFit
from ghostline.physics.library import ParamSpec as LibraryParamSpec
from ghostline.physics.library import get_model
from ghostline.store.records import Experiment, FitParam, FitRecord, ModelRecord, Trial

from .inputs import InputsError, PhysicsInputs, build_inputs
from .runners import RunnerPool
from .session import load_session

_PLOT_FILES = {
    "fit": "fit.png",
    "resid": "residuals.png",
    "fit_train": "fit_train.png",
    "resid_train": "residuals_train.png",
}
N_BOOT = 500


def load_inputs(ctx: JobContext, experiment: Experiment, trial: Trial) -> PhysicsInputs:
    """The inputs of the analysis, rebuilt from the stored trajectory, calibration and scene."""
    if experiment.scene is None:
        raise StageError("The experiment has no scene analysis.", code="no_scene", retryable=False)
    rows = ctx.repo.list_trajectories([trial.id])
    if not rows or not rows[0].parquet_path:
        raise StageError("The clip has no trajectory yet.", code="no_trajectory", retryable=False)
    try:
        return build_inputs(
            experiment,
            trial,
            experiment.scene,
            parquet_path=rows[0].parquet_path,
            summary=rows[0].summary,
            calibration=trial.calibration,
            paths=ctx.paths,
        )
    except InputsError as exc:
        raise StageError(str(exc), code="cannot_fit", retryable=False) from exc


def model_spec(ctx: JobContext, experiment: Experiment, model: ModelRecord) -> ModelSpec:
    """The harness spec of a stored model: from the saved session when there is one (it keeps a
    custom model's observables, state and constants), else from the model row."""
    session = load_session(ctx.paths, experiment.id)
    if session is not None and model.id in session.models:
        return session.models[model.id].spec
    if model.library_ref:
        return ModelSpec(id=model.id, library_ref=model.library_ref)
    if not model.code or not model.params_spec:
        raise StageError(f"Model {model.id} has no code to run.", code="no_model", retryable=False)
    return ModelSpec(
        id=model.id,
        code=model.code,
        name=model.name,
        mechanism=model.mechanism,
        equations_latex=model.equations_latex,
        params={k: LibraryParamSpec(**v.model_dump()) for k, v in model.params_spec.items()},
    )


def _heldout_ci(errors: list[float] | None, seed: int = 0) -> list[float] | None:
    """95% interval of the held-out RMSE from a block bootstrap of the model's own errors."""
    if not errors or len(errors) < 6:
        return None
    err = np.asarray(errors, dtype=float)
    n, rng = err.size, np.random.default_rng(seed)
    block = max(1, min(10, n // 4))
    draws = np.array(
        [
            np.sqrt(np.mean(err[fitting.block_bootstrap_indices(n, block, rng)] ** 2))
            for _ in range(N_BOOT)
        ]
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def _fit_direct(inputs: PhysicsInputs, ref: str, split_pct: float, guesses: dict[str, float]):
    """Fit a library model from the given initial guesses (trusted code, this process)."""
    model = get_model(ref, inputs.system)
    ds = fitting.Dataset.from_series(inputs.data).for_model(model)
    tau = ds.t - ds.t[0]
    constants = model.prepare(tau, ds.obs, inputs.constants)
    split, outcome = fitting.fit_split(
        model, ds, constants, {}, split_pct,
        n_starts=1, seed=0, deadline=None, warm=guesses, n_draws=0,
    )  # fmt: skip
    mask, _, t_split = fitting.split_masks(model, ds, split_pct)
    params = {**constants, **outcome.values}
    return model, ds, split, params, mask, t_split


def rerun(
    ctx: JobContext,
    experiment: Experiment,
    model: ModelRecord,
    split_pct: float,
    initial_guesses: dict[str, float],
) -> FitRecord:
    trial = ctx.repo.latest_trial(experiment.id, "fit", status="ready")
    if trial is None:
        raise StageError("The first clip is missing.", code="no_trial", retryable=False)
    inputs = load_inputs(ctx, experiment, trial)
    spec = model_spec(ctx, experiment, model)
    fit_id = uuid4()
    folder = ctx.paths.fit_dir(experiment.id, fit_id)
    plots: dict[str, str] = {}
    note = None

    def save_plot(kind: str, source: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, folder / _PLOT_FILES[kind])
        plots[_PLOT_FILES[kind].removesuffix(".png")] = ctx.paths.rel(
            experiment.id, folder / _PLOT_FILES[kind]
        )

    if initial_guesses and spec.library_ref:
        import tempfile
        from pathlib import Path

        _, ds, split, params, mask, t_split = _fit_direct(
            inputs, spec.library_ref, split_pct, initial_guesses
        )
        aic = bic = None
        with tempfile.TemporaryDirectory(prefix="ghostline-rerun-") as tmp:
            library = get_model(spec.library_ref, inputs.system)
            for ref in render_plots(library, ds, params, t_split, mask, Path(tmp)):
                if ref.kind in _PLOT_FILES:
                    save_plot(ref.kind, str(Path(tmp) / ref.path))
    else:
        if initial_guesses:
            note = "Initial guesses can't be applied to a custom model; it was refit without them."
        job = FitJob(
            job_id=f"rerun-{uuid4().hex[:8]}",
            mode="select",
            system=inputs.system,
            data=inputs.data,
            constants=inputs.constants,
            models=[spec],
            splits=[split_pct],
            primary_split=split_pct,
            bootstrap=BootstrapSpec(n=0),
            plots=True,
        )
        with RunnerPool() as pool:
            result = pool.run(job).models[0]
            if result.status != "ok" or not result.splits:
                raise StageError(
                    f"The re-run of {model.id} failed: {result.error}", code="fit_failed"
                )
            split = result.splits[0]
            aic, bic = result.aic, result.bic
            for ref in result.plots:
                if ref.kind in _PLOT_FILES:
                    save_plot(ref.kind, ref.path)
    return _record(
        fit_id, experiment, trial, model, split, split_pct, plots, aic, bic, note, initial_guesses
    )


def _record(
    fit_id: UUID,
    experiment: Experiment,
    trial: Trial,
    model: ModelRecord,
    split: SplitFit,
    split_pct: float,
    plots: dict[str, str],
    aic: float | None,
    bic: float | None,
    note: str | None,
    guesses: dict[str, float],
) -> FitRecord:
    diag = split.train
    ci = _heldout_ci(split.heldout_errors)
    params = [
        FitParam(
            name=p.name,
            value=p.value,
            ci_low=p.ci_low if p.ci_low is not None else p.value,
            ci_high=p.ci_high if p.ci_high is not None else p.value,
            unit=p.unit,
        )
        for p in split.params
        if p.kind == "physical" or p.derived
    ]
    diagnostics: dict[str, Any] = {
        **diag.model_dump(),
        "reduced_chi2": diag.chi2_red,
        "lag1_autocorr": diag.lag1,
        "durbin_watson": diag.durbin_watson,
        "runs_p": diag.runs_p,
        "systematic_residual": diag.systematic,
        "converged": split.converged,
        "at_bound": [p.name for p in split.params if p.at_bound],
    }
    if note:
        diagnostics["note"] = note
    return FitRecord(
        id=fit_id,
        experiment_id=experiment.id,
        model_id=model.id,
        trial_id=trial.id,
        split={
            "pct": split_pct,
            "kind": "rerun",
            "t_split": split.t_split,
            "n_train": split.n_train,
            "n_test": split.n_test,
            "initial_guesses": guesses,
        },
        params=params,
        train_rmse=diag.rmse,
        heldout_rmse=split.heldout_rmse,
        heldout_ci=ci,
        aic=aic,
        bic=bic,
        diagnostics=diagnostics,
        plot_paths=plots,
    )
