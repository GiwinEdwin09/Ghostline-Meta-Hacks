"""Part 17 as a stage: diagnose the drift with Wonder, fit the revision on trial 1 only, adopt it
if it forecasts trial 2 better beyond noise, and render ghost v2 then.

`ghost.stage.diagnose_revision` calls into this module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import RevisionResult
from ghostline.physics.harness.schema import ModelSpec
from ghostline.physics.library import get_model
from ghostline.physics.runner import Runner
from ghostline.store.core_schemas import Review
from ghostline.store.records import Attachment, Experiment, GhostRender, ModelRecord, Ref

from .diagnose import (
    Ask,
    DiagnosisInputs,
    DiagnosisReply,
    ask_wonder,
    build_brief,
    error_plot_png,
    frame_indices,
    frame_jpegs,
)
from .divergence import measure_divergence
from .forecast import forecast_trial, freeze
from .loading import constants_for, load_series, make_sprite, masked_track, trial_file
from .model import N_DRAWS, Divergence, Forecast, GhostError
from .plan import build_plan
from .revise import RevisionTest, spec_from_reply, try_revision, validate_spec
from .specs import library_spec, model_info, spec_from_record
from .stage import check_system, render_version, runner_for


@dataclass
class RevisionDossier:
    """What a reviewer sees of a revision: the claim and the evidence for it."""

    diagnosis: str
    v1_model_id: str
    v2_model_id: str
    frozen_v2: dict[str, float]
    rmse_v1_mm: float
    rmse_v2_mm: float
    diff_ci_mm: tuple[float, float]
    onset_v1_s: float | None
    onset_v2_s: float | None
    adopted: bool
    trial1_selection: str | None
    images: dict[str, Path] = field(default_factory=dict)


ReviewHook = Callable[[RevisionDossier], Review | None]


def diagnose(
    ctx: JobContext,
    experiment: Experiment,
    ghost: GhostRender,
    *,
    runner: Runner | None = None,
    ask: Ask | None = None,
    review: ReviewHook | None = None,
    n_draws: int = N_DRAWS,
) -> RevisionResult:
    scene = experiment.scene
    if scene is None:
        raise StageError("The experiment has no scene analysis.", code="no_scene", retryable=False)
    system = check_system(scene)
    prove = ctx.repo.get_trial(ghost.trial_id)
    fit_trial = ctx.repo.latest_trial(experiment.id, "fit", status="ready")
    if prove is None or fit_trial is None:
        raise StageError("The clips behind this ghost are gone.", code="no_trial", retryable=False)
    v1_id = ghost.model_id or experiment.winner_model_id or ""
    record = ctx.repo.get_model(experiment.id, v1_id)
    v1_spec = spec_from_record(record) if record else library_spec(v1_id)
    v1_name = record.name if record else v1_id
    try:
        trial1 = load_series(ctx, fit_trial, scene)
        trial2 = load_series(ctx, prove, scene)
        constants = constants_for(system, trial1, experiment)
        with runner_for(runner, v1_spec) as run1:
            frozen1 = freeze(run1, v1_spec, system, trial1, constants)
            forecast1 = forecast_trial(run1, v1_spec, system, frozen1, trial2, n_draws=20)
        div1 = measure_divergence(forecast1, trial2)
    except GhostError as error:
        raise StageError(str(error), code="ghost_failed", retryable=False) from error

    # Wonder looks at the drift.
    master = trial_file(ctx, prove, "master")
    if master is None:
        raise StageError("The second clip's master video is missing.", code="no_master")
    sprite = make_sprite(master, masked_track(ctx, prove, scene, master), trial2)
    plan, track = build_plan(
        forecast1, div1, trial2, sprite, version=ghost.version,
        label=f"Ghost v{ghost.version} · {v1_name}", source_angle=sprite.source_angle,
    )  # fmt: skip
    inputs, images = _inputs(
        ctx, experiment, master, plan, track.onset_s, system, v1_spec, v1_name, frozen1,
        forecast1, div1, trial1, trial2, ghost.version,
    )  # fmt: skip
    ask = ask or (lambda i: ask_wonder(ctx.meta, i, experiment_id=str(experiment.id)))
    try:
        reply = ask(inputs)
        taken = {m.id for m in ctx.repo.list_models(experiment.id)} | {v1_id}
        v2_spec = _spec(reply, system, v1_spec, taken)
        with runner_for(runner, v1_spec, v2_spec) as run2:
            problems = validate_spec(run2, v2_spec, system, trial1, constants)
            if problems:  # one repair attempt, with the harness's own errors
                reply = ask(_with_errors(inputs, problems))
                v2_spec = _spec(reply, system, v1_spec, taken)
                problems = validate_spec(run2, v2_spec, system, trial1, constants)
                if problems:
                    raise GhostError("The revised model code is not valid: " + "; ".join(problems))
            ctx.phase("fitting")
            test = try_revision(
                run2, v1_spec, v2_spec, system, constants, trial1, trial2, forecast1, div1,
                n_draws=n_draws,
            )  # fmt: skip
    except GhostError as error:
        raise StageError(str(error), code="revision_failed", retryable=False) from error
    _notebook(ctx, ghost, reply, images, test, v1_id)

    # v2, only when it is better beyond noise.
    ghost_v2 = None
    if test.adopted:
        ctx.phase("rendering")
        try:
            ghost_v2, _ = render_version(
                ctx, experiment, prove, scene, trial2, test.forecast, test.div,
                version=ctx.repo.next_ghost_version(experiment.id),
                model_name=reply.revised_model.name or test.spec.id,
            )  # fmt: skip
        except GhostError as error:
            raise StageError(str(error), code="ghost_failed", retryable=False) from error
    text = _summary(reply, test, div1, forecast1, v1_id)
    text += _review_round(ctx, review, reply, test, div1, v1_id, forecast1, images)
    return RevisionResult(
        diagnosis_md=text,
        adopted=test.adopted,
        revised_model=_record(ctx, experiment, reply, test),
        ghost=ghost_v2,
    )


def _with_errors(inputs: DiagnosisInputs, problems: list[str]) -> DiagnosisInputs:
    text = (
        inputs.text
        + "\n\nYour custom model failed the harness checks: "
        + "; ".join(problems)
        + "\nAnswer again with corrected code."
    )
    return DiagnosisInputs(text, inputs.images, inputs.labels)


def _spec(reply: DiagnosisReply, system: str, v1_spec: ModelSpec, taken: set[str]) -> ModelSpec:
    observables = model_info(v1_spec, system, ("y",)).observables
    spec = spec_from_reply(reply.revised_model, system, observables, taken)
    if spec.library_ref is not None and spec.library_ref == v1_spec.library_ref:
        raise GhostError(
            f"Wonder proposed {spec.library_ref} again, which is the model that drifted."
        )
    return spec


def _inputs(
    ctx, experiment, master, plan, onset_rel, system, spec, name, frozen, forecast, div,
    trial1, trial2, version,
):  # fmt: skip
    """The brief, the error plot and four annotated frames, saved next to the ghost video."""
    t0 = float(forecast.t[0])
    plot = error_plot_png(div, t0, plan.launch_end_rel, f"ghost v{version} ({spec.id})")
    picks = frame_indices(plan, onset_rel, plan.error_mm)
    jpegs = frame_jpegs(master, plan, picks)
    if spec.library_ref:
        info = get_model(spec.library_ref, system).catalog_entry()
    else:
        info = {
            "id": spec.id, "name": name, "mechanism": spec.mechanism,
            "equations_latex": spec.equations_latex, "signature": "",
        }  # fmt: skip
    brief = build_brief(
        system=system, model=info, frozen=frozen, forecast=forecast, div=div, trial1=trial1,
        trial2=trial2, t0=t0, frame_times=[float(plan.t_rel[i]) for i in picks],
    )  # fmt: skip
    labels = ["Image 1: error against time."] + [
        f"Image {k + 2}: frame at {plan.t_rel[i]:.2f} s." for k, i in enumerate(picks)
    ]
    folder = ctx.paths.ghost_video(experiment.id, version).parent
    folder.mkdir(parents=True, exist_ok=True)
    saved = {"error_plot": folder / f"v{version}_diag_error.png"}
    saved["error_plot"].write_bytes(plot)
    for k, i in enumerate(picks):
        saved[f"frame_{k}"] = folder / f"v{version}_diag_{k}.jpg"
        saved[f"frame_{k}"].write_bytes(jpegs[i])
    return DiagnosisInputs(brief, [plot, *[jpegs[i] for i in picks]], labels), saved


def _notebook(ctx, ghost, reply: DiagnosisReply, images, test: RevisionTest, v1_id: str) -> None:
    rel = lambda p: ctx.paths.rel(ctx.experiment_id, p)  # noqa: E731
    ctx.notebook.append(
        actor="wonder",
        kind="revision",
        title=f"Why the ghost drifted: {reply.missing_physics}",
        body_md=f"{reply.diagnosis}\n\nIf that is right: {reply.predicted_signature}",
        data={
            "missing_physics": reply.missing_physics,
            "revised_model": reply.revised_model.model_dump(),
            "predicted_signature": reply.predicted_signature,
        },
        attachments=[Attachment(path=rel(p), kind="image") for p in images.values()],
        refs=[Ref(kind="ghost", id=str(ghost.id))],
    )
    selection = test.selection
    low, high = test.diff_ci_mm
    ctx.notebook.append(
        actor="harness",
        kind="fit",
        title=f"{test.spec.id} fitted on trial 1 only, tested on trial 2",
        body_md=(
            f"After the launch window both models forecast trial 2, which neither was fitted to. "
            f"RMSE {v1_id}: {test.rmse_v1_mm:.2f} mm, {test.spec.id}: {test.rmse_v2_mm:.2f} mm; "
            f"95% interval of the difference {low:.2f} to {high:.2f} mm. "
            + ("Better beyond noise." if test.adopted else "Not better beyond noise.")
        ),
        data={
            "adopted": test.adopted,
            "rmse_v1_mm": test.rmse_v1_mm,
            "rmse_v2_mm": test.rmse_v2_mm,
            "rmse_difference_ci95_mm": list(test.diff_ci_mm),
            "frozen_v2": test.frozen.values,
            "frozen_v2_sigma": test.frozen.sigma,
            "trial1_selection": None if selection is None else selection.message,
        },
    )


def _onset(div: Divergence, forecast: Forecast) -> float | None:
    return None if div.onset_t is None else float(div.onset_t - forecast.t[0])


def _summary(reply, test: RevisionTest, div1, forecast1, v1_id: str) -> str:
    def when(o):
        return "never drifted" if o is None else f"drifted at {o:.2f} s"

    o1, o2 = _onset(div1, forecast1), _onset(test.div, test.forecast)
    verdict = (
        f"Adopted: {test.spec.id} {when(o2)}, where {v1_id} {when(o1)}."
        if test.adopted
        else f"The revision didn't help: {test.spec.id} did not forecast trial 2 better beyond "
        f"the noise (RMSE {test.rmse_v2_mm:.2f} mm against {test.rmse_v1_mm:.2f} mm)."
    )
    return (
        f"{reply.diagnosis}\n\nRevised model: **{reply.revised_model.name}** "
        f"({test.spec.id}), fitted on trial 1 only. {verdict}"
    )


def _review_round(ctx, review, reply, test, div1, v1_id, forecast1, images) -> str:
    if review is None:
        return ""
    ctx.phase("peer_review")
    dossier = RevisionDossier(
        diagnosis=reply.diagnosis, v1_model_id=v1_id, v2_model_id=test.spec.id,
        frozen_v2=test.frozen.values, rmse_v1_mm=test.rmse_v1_mm, rmse_v2_mm=test.rmse_v2_mm,
        diff_ci_mm=test.diff_ci_mm, onset_v1_s=_onset(div1, forecast1),
        onset_v2_s=_onset(test.div, test.forecast), adopted=test.adopted,
        trial1_selection=None if test.selection is None else test.selection.message,
        images=images,
    )  # fmt: skip
    outcome = review(dossier)
    if outcome is None:
        return ""
    ctx.notebook.append(
        actor="spark_open",
        kind="review",
        title=f"Review of the revision: {outcome.verdict}",
        body_md=outcome.summary,
        data=outcome.model_dump(),
    )
    return f"\n\nReviewer ({outcome.verdict}): {outcome.summary}"


def _record(ctx, experiment, reply: DiagnosisReply, test: RevisionTest) -> ModelRecord:
    library = test.spec.library_ref is not None
    ref = test.spec.library_ref
    round_ = max((m.round for m in ctx.repo.list_models(experiment.id)), default=0) + 1
    note = (
        f"Did not forecast trial 2 better beyond noise (RMSE {test.rmse_v2_mm:.2f} mm "
        f"against {test.rmse_v1_mm:.2f} mm)."
    )
    return ModelRecord(
        experiment_id=experiment.id,
        id=test.spec.id,
        round=round_,
        source="library" if library else "wonder",
        library_ref=ref,
        code=None if library else test.spec.code,
        name=reply.revised_model.name or test.spec.id,
        mechanism=reply.revised_model.mechanism,
        equations_latex=reply.revised_model.equations_latex,
        params_spec=None
        if library
        else {k: v.model_dump() for k, v in (test.spec.params or {}).items()},
        refuting_signature=reply.predicted_signature,
        status="proposed" if test.adopted else "rejected",
        rejection_reason=None if test.adopted else note,
    )
