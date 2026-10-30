"""The orchestrator's model of one experiment: harness runs, selection, records and the notebook.

`Lab` is what the tools act on. It owns the `Session` (see session.py), runs fit jobs through a
`RunnerPool`, keeps the held-out selection current, and writes every result to the database
(`ctx.repo`), the Volume and the notebook as it appears. It holds no model client: the Wonder
loop (loop.py), the library-only mode (fallback.py) and the review's respond round (review/) are
different drivers of the same Lab.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from ghostline.confidence import (
    ConfidenceAssessment,
    QualityFlag,
    ReviewStatus,
    assess_confidence,
    motion_extent_flags,
)
from ghostline.jobs.context import JobContext
from ghostline.physics.harness.schema import (
    BootstrapSpec,
    FitJob,
    FitResult,
    ModelResult,
    ModelSpec,
)
from ghostline.physics.library import catalog, get_model, model_ids
from ghostline.physics.select import (
    FinalQuantity,
    ModelScore,
    Selection,
    calibration_perturbations,
    comparison_table,
    final_quantities,
    select_models,
)
from ghostline.store.core_schemas import ParamSpec
from ghostline.store.records import Experiment, FitParam, FitRecord, ModelRecord, Trial

from . import events
from .inputs import PhysicsInputs
from .numbers import floats_in, round_tree, sorted_known
from .runners import RunnerPool
from .session import ModelEntry, Session, save_session

SPLITS = [0.65, 0.5, 0.8]
FINAL_BOOTSTRAP = 200
FINAL_PERTURBATIONS = 50
_PLOT_NAMES = {
    "fit": "fit.png",
    "resid": "residuals.png",
    "fit_train": "fit_train.png",
    "resid_train": "residuals_train.png",
}
_LEVELS = {"low": 0, "medium": 1, "high": 2}


class LabError(Exception):
    """A request the harness can't serve (an unknown model, no selection yet). The message goes
    back to Wonder as a tool error."""


class Lab:
    def __init__(
        self,
        ctx: JobContext,
        experiment: Experiment,
        trial: Trial,
        inputs: PhysicsInputs,
        session: Session,
        pool: RunnerPool,
        *,
        seed: int = 0,
    ) -> None:
        self.ctx = ctx
        self.experiment = experiment
        self.trial = trial
        self.inputs = inputs
        self.session = session
        self.pool = pool
        self.seed = seed
        self.exp_id = experiment.id

    # --- session ----------------------------------------------------------------------------
    @classmethod
    def new_session(cls, inputs: PhysicsInputs, **extra: Any) -> Session:
        return Session(
            system=inputs.system,
            trial_id=str(inputs.trial_id or ""),
            primary_split=inputs.primary_split,
            t_split=inputs.t_split,
            duration_s=inputs.duration_s,
            flags=list(inputs.flags),
            budget=inputs.budget,
            constants=dict(inputs.constants),
            sigma=inputs.sigma,
            motion_range=inputs.motion_range,
            known_numbers=sorted(set(floats_in(round_tree(inputs.context)))),
            **extra,
        )

    def save(self) -> None:
        save_session(self.ctx.paths, self.exp_id, self.session)
        self.ctx.commit()

    def show(self, payload: dict[str, Any]) -> str:
        """What a tool returns: rounded JSON text. The numbers in it become known to Wonder."""
        shown = round_tree(payload)
        self.session.known_numbers.extend(floats_in(shown))
        return json.dumps(shown, separators=(",", ":"), ensure_ascii=False)

    def remember(self, *values: Any) -> None:
        for value in values:
            self.session.known_numbers.extend(floats_in(round_tree(value)))

    def known(self) -> np.ndarray:
        splits = [*SPLITS, *(1 - s for s in SPLITS)]
        return sorted_known([*self.session.known_numbers, *splits])

    def _next_job_id(self, kind: str) -> str:
        n = self.session.counters.get(kind, 0) + 1
        self.session.counters[kind] = n
        return f"{kind}{n}"

    # --- library and models -----------------------------------------------------------------
    def library_ids(self) -> list[str]:
        return model_ids(self.inputs.system)

    def catalog(self, system: str | None = None) -> list[dict[str, Any]]:
        return catalog(system or self.inputs.system)

    def library_entry(self, ref: str, rank: int, *, round_: int = 0) -> ModelEntry:
        """A library model as a candidate, with the library's own description of it."""
        model = get_model(ref, self.inputs.system)
        return ModelEntry(
            spec=ModelSpec(
                id=ref,
                library_ref=ref,
                name=model.name,
                mechanism=model.mechanism,
                equations_latex=model.equations_latex,
            ),
            name=model.name,
            mechanism=model.mechanism,
            equations_latex=model.equations_latex,
            refuting_signature=model.signature,
            prior_rank=rank,
            round=round_,
            source="library",
        )

    def add_model(self, entry: ModelEntry) -> None:
        """Register a candidate and save it to the database."""
        self.session.models[entry.id] = entry
        spec = (
            get_model(entry.spec.library_ref, self.inputs.system).params
            if entry.spec.library_ref
            else entry.spec.params or {}
        )
        self.ctx.repo.upsert_model(
            ModelRecord(
                experiment_id=self.exp_id,
                id=entry.id,
                round=entry.round,
                source=entry.source,
                library_ref=entry.spec.library_ref,
                code=entry.spec.code,
                name=entry.name,
                mechanism=entry.mechanism,
                equations_latex=entry.equations_latex,
                params_spec={
                    name: ParamSpec.model_validate(p.model_dump()) for name, p in spec.items()
                },
                prior_rank=entry.prior_rank,
                refuting_signature=entry.refuting_signature,
                status=entry.status,
            )
        )

    def set_status(self, model_id: str, status: str, reason: str | None = None) -> None:
        entry = self.session.models[model_id]
        entry.status, entry.rejection_reason = status, reason
        self.ctx.repo.set_model_status(self.exp_id, model_id, status, reason)

    # --- fitting ----------------------------------------------------------------------------
    def _job(self, mode: str, specs: list[ModelSpec], **extra: Any) -> FitJob:
        return FitJob(
            job_id=self._next_job_id("sel" if mode == "select" else "fin"),
            mode=mode,  # type: ignore[arg-type]
            system=self.inputs.system,
            data=self.inputs.data,
            constants=self.inputs.constants,
            models=specs,
            splits=SPLITS,
            primary_split=self.inputs.primary_split,
            **extra,
        )

    def validate_custom(self, specs: list[ModelSpec]) -> dict[str, list[str]]:
        """Dry-run custom models in the runner (never in this process). Errors by model id."""
        if not specs:
            return {}
        job = FitJob(
            job_id=self._next_job_id("val"),
            mode="validate",
            system=self.inputs.system,
            data=self.inputs.data,
            constants=self.inputs.constants,
            models=specs,
        )
        result = self.pool.run(job)
        return {r.model_id: r.errors for r in result.validation if not r.ok}

    def fit_selection(self, ids: list[str]) -> list[str]:
        """Fit the models that have no result yet, keep the held-out selection current, and save
        the new fits. Returns the ids fitted by this call."""
        fresh = [i for i in ids if i in self.session.models and i not in self.session.results]
        if not fresh:
            return []
        job = self._job(
            "select",
            [self.session.models[i].spec for i in fresh],
            bootstrap=BootstrapSpec(n=0, seed=self.seed),
            plots=True,
        )
        result = self.pool.run(job)
        by_id = {m.id: m for m in result.models}
        for model_id in fresh:
            self.session.results[model_id] = by_id[model_id]
        self.reselect()
        self._save_fits(fresh)
        self.save()
        return fresh

    def reselect(self) -> Selection:
        everything = FitResult(
            job_id="selection",
            mode="select",
            system=self.inputs.system,
            models=list(self.session.results.values()),
        )
        self.session.selection = select_models(
            everything,
            primary_split=self.inputs.primary_split,
            sigma=self.inputs.sigma,
            motion_range=self.inputs.motion_range,
            seed=self.seed,
        )
        return self.session.selection

    def _plot_dir(self, fit_id: UUID) -> Path:
        return self.ctx.paths.fit_dir(self.exp_id, fit_id)

    def _copy_plots(self, result: ModelResult, fit_id: UUID) -> dict[str, str]:
        """Copy a model's plot files from the runner's scratch folder to the Volume."""
        folder = self._plot_dir(fit_id)
        saved: dict[str, str] = {}
        for ref in result.plots:
            source = Path(ref.path)
            name = _PLOT_NAMES.get(ref.kind)
            if name is None or not source.is_file():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, folder / name)
            relative = self.ctx.paths.rel(self.exp_id, folder / name)
            ref.path = relative
            self.session.plot_files[ref.id] = relative
            saved[name.removesuffix(".png")] = relative
        return saved

    def _fit_params(self, score: ModelScore) -> list[FitParam]:
        out = []
        for p in score.params:
            if p.kind != "physical" and not p.derived:
                continue
            lo = p.ci_low if p.ci_low is not None else p.value
            hi = p.ci_high if p.ci_high is not None else p.value
            out.append(FitParam(name=p.name, value=p.value, ci_low=lo, ci_high=hi, unit=p.unit))
        return out

    def _fit_record(self, entry: ModelEntry, score: ModelScore, plots: dict[str, str]) -> FitRecord:
        sel = self.session.selection
        assert sel is not None
        prim = f"{sel.primary_split:.2f}"
        diag = score.train_diagnostics
        result = self.session.results[entry.id]
        first = result.splits[0] if result.splits else None
        diagnostics: dict[str, Any] = diag.model_dump() if diag else {}
        diagnostics.update(
            reduced_chi2=diag.chi2_red if diag else None,
            lag1_autocorr=diag.lag1 if diag else None,
            durbin_watson=diag.durbin_watson if diag else None,
            runs_p=diag.runs_p if diag else None,
            systematic_residual=diag.systematic if diag else None,
            nrmse=score.nrmse,
            time_to_divergence_s=score.divergence_time,
            heldout_rmse_by_split=score.heldout_rmse,
            heldout_ci_by_split={k: list(v) for k, v in score.heldout_ci.items()},
            at_bound=[p.name for p in score.params if p.at_bound],
            status=score.status,
            error=score.error,
        )
        ci = score.heldout_ci.get(prim)
        return FitRecord(
            id=UUID(self.session.fit_ids[entry.id]),
            experiment_id=self.exp_id,
            model_id=entry.id,
            trial_id=self.trial.id,
            split={
                "pct": sel.primary_split,
                "kind": "selection",
                "splits": SPLITS,
                "t_split": first.t_split if first else None,
                "n_train": first.n_train if first else None,
                "n_test": first.n_test if first else None,
            },
            params=self._fit_params(score),
            train_rmse=score.train_rmse,
            heldout_rmse=score.heldout_rmse.get(prim),
            heldout_ci=list(ci) if ci else None,
            aic=score.aic,
            bic=score.bic,
            diagnostics=diagnostics,
            plot_paths=plots,
        )

    def _save_fits(self, ids: list[str]) -> None:
        sel = self.session.selection
        assert sel is not None
        unit = _unit(self.inputs)
        for model_id in ids:
            entry = self.session.models[model_id]
            score = sel.score(model_id)
            fit_id = uuid4()
            self.session.fit_ids[model_id] = str(fit_id)
            plots = self._copy_plots(self.session.results[model_id], fit_id)
            record = self._fit_record(entry, score, plots)
            self.ctx.repo.insert_fit(record)
            if score.status == "ok":
                self.set_status(model_id, "fitted")
            else:
                self.set_status(model_id, "invalid", score.error)
            events.fit(self.ctx.notebook, entry, score, sel, unit, str(fit_id))
        names = {m.id: m.name for m in self.session.models.values()}
        events.verdict(self.ctx.notebook, sel, names)

    # --- the table Wonder reads -------------------------------------------------------------
    def table_payload(self, new_ids: list[str] | None = None) -> dict[str, Any]:
        sel = self.session.selection
        if sel is None:
            raise LabError("no model has been fitted yet; call run_fits first")
        rows = comparison_table(sel)
        for row in rows:
            entry = self.session.models[row["model_id"]]
            score = sel.score(row["model_id"])
            row["name"] = entry.name
            row["n_physical_parameters"] = score.n_physical
            row["heldout_rmse_by_split"] = score.heldout_rmse
            row["fitted_now"] = row["model_id"] in (new_ids or [])
        return {
            "training_window": {
                "fraction": sel.primary_split,
                "ends_at_s": self.session.t_split,
                "unit": _unit(self.inputs),
            },
            "verdict": {
                "status": sel.status,
                "winner": sel.selected_id,
                "splits_won": sel.splits_won,
                "splits": sel.n_splits,
                "tied_with": sel.tied_with,
                "message": sel.message,
            },
            "models": rows,
        }

    def agent_plot_ids(self) -> set[str]:
        sel = self.session.selection
        if sel is None:
            return set()
        return {p.id for m in sel.models for p in m.plots if p.audience == "agent"}

    def plot_bytes(self, plot_id: str) -> bytes:
        relative = self.session.plot_files.get(plot_id)
        if relative is None:
            raise LabError(f"unknown plot id {plot_id!r}")
        return self.ctx.paths.resolve_media(self.exp_id, relative).read_bytes()

    def reviewer_plots(self, model_id: str) -> dict[str, bytes]:
        """The fit and residual plots that show the held-out points: for the reviewer."""
        out = {}
        for kind in ("fit", "resid"):
            relative = self.session.plot_files.get(f"{model_id}.{kind}")
            if relative:
                out[kind] = self.ctx.paths.resolve_media(self.exp_id, relative).read_bytes()
        return out

    # --- the final fit ----------------------------------------------------------------------
    def ensure_final(self, model_id: str | None = None) -> list[FinalQuantity]:
        """Refit a model (the selected winner by default) on all the data with the bootstrap and
        the calibration refits, and keep its quantities. A repeat call returns the stored ones."""
        sel = self.session.selection
        model_id = model_id or (sel.selected_id if sel else None)
        if model_id is None or model_id not in self.session.models:
            raise LabError("there is no winner to refit yet; fit the models first")
        if model_id in self.session.quantities:
            return self.session.quantities[model_id]
        spec = self.session.models[model_id].spec
        score = sel.score(model_id) if sel else None
        if score is None or score.status != "ok":
            raise LabError(f"{model_id} could not be fitted, so it has no final numbers")
        tied = [
            self.session.results[i]
            for i in (sel.tied_with if sel else [])
            if i in self.session.results and self.session.results[i].status == "ok"
        ]
        perturbations = calibration_perturbations(
            self.session.budget, self.inputs.constants, FINAL_PERTURBATIONS, self.seed
        )
        quantities = None
        for limit in (90.0, 240.0):
            job = self._job(
                "final",
                [spec],
                bootstrap=BootstrapSpec(n=FINAL_BOOTSTRAP, seed=self.seed),
                perturbations=perturbations,
                plots=False,
                time_limit_s=limit,
            )
            final = self.pool.run(job).models[0]
            if final.status != "ok":
                raise LabError(f"the final fit of {model_id} failed: {final.error}")
            try:
                quantities = final_quantities(final, self.session.budget, tied)
            except ValueError as error:
                if limit >= 240.0:
                    raise LabError(str(error)) from error
                continue
            break
        assert quantities is not None
        self.session.finals[model_id] = final
        self.session.quantities[model_id] = quantities
        period = next(
            (q.value for q in quantities if q.name in ("period_small_angle", "period")), None
        )
        self._add_flags(motion_extent_flags(self.session.duration_s, period))
        self._save_final_fit(model_id, quantities)
        self.save()
        return quantities

    def _add_flags(self, flags: list[QualityFlag]) -> None:
        have = {f.code for f in self.session.flags}
        self.session.flags.extend(f for f in flags if f.code not in have)

    def _save_final_fit(self, model_id: str, quantities: list[FinalQuantity]) -> None:
        """The winner's last fit record: the all-data parameters with their full uncertainty,
        next to the held-out score of its selection fit."""
        sel = self.session.selection
        assert sel is not None
        score = sel.score(model_id)
        prim = f"{sel.primary_split:.2f}"
        base = self._fit_record(self.session.models[model_id], score, {})
        old = UUID(self.session.fit_ids[model_id])
        plots = {
            name.removesuffix(".png"): self.session.plot_files[f"{model_id}.{kind}"]
            for kind, name in _PLOT_NAMES.items()
            if f"{model_id}.{kind}" in self.session.plot_files
        }
        ci = score.heldout_ci.get(prim)
        record = base.model_copy(
            update={
                "id": uuid4(),
                "split": {
                    **base.split,
                    "pct": 1.0,
                    "kind": "final",
                    "heldout_from_pct": sel.primary_split,
                },
                "params": [
                    FitParam(
                        name=q.name,
                        value=q.value,
                        ci_low=q.ci_low,
                        ci_high=q.ci_high,
                        unit=q.unit,
                    )
                    for q in quantities
                ],
                "heldout_rmse": score.heldout_rmse.get(prim),
                "heldout_ci": list(ci) if ci else None,
                "plot_paths": plots,
                "diagnostics": {
                    **base.diagnostics,
                    "budget": {q.name: q.budget for q in quantities},
                    "selection_fit_id": str(old),
                },
            }
        )
        self.ctx.repo.insert_fit(record)
        self.session.fit_ids[model_id] = str(record.id)

    # --- confidence -------------------------------------------------------------------------
    def assess(self, review: ReviewStatus | None = None) -> ConfidenceAssessment:
        sel = self.session.selection
        if sel is None:
            raise LabError("there is no selection to assess")
        return assess_confidence(
            sel, system=self.inputs.system, flags=self.session.flags, review=review
        )

    @staticmethod
    def lower_of(a: str, b: str) -> str:
        return a if _LEVELS[a] <= _LEVELS[b] else b


def _unit(inputs: PhysicsInputs) -> str:
    units = set(inputs.units.values())
    return units.pop() if len(units) == 1 else "/".join(sorted(units))
