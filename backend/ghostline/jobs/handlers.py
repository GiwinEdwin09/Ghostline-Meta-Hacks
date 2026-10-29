"""The body of each job kind (build plan Part 21): call the stages in order, persist what they
return, and keep the notebook and the experiment row up to date. Stage calls go through
`jobs.stages` so tests and wave 2 can replace them."""

from collections.abc import Callable
from typing import Any
from uuid import UUID

from ghostline.jobs import stages
from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.jobs.retry import call_with_retries
from ghostline.jobs.stage_types import GhostResult
from ghostline.store.core_schemas import SceneAnalysis
from ghostline.store.records import (
    Experiment,
    GhostRender,
    JobKind,
    Ref,
    TrajectoryRecord,
    Trial,
)

Handler = Callable[[JobContext], None]

RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 1.0


def _stage[T](fn: Callable[..., T], *args: Any) -> T:
    """Run a stage, retrying on TransientStageError."""
    return call_with_retries(
        lambda: fn(*args), attempts=RETRY_ATTEMPTS, base_delay_s=RETRY_BASE_DELAY_S
    )


def _experiment(ctx: JobContext) -> Experiment:
    experiment = ctx.repo.get_experiment(ctx.experiment_id)
    if experiment is None:
        raise StageError("The experiment no longer exists.", code="not_found", retryable=False)
    return experiment


def _trial(ctx: JobContext, role: str, *, key: str = "trial_id") -> Trial:
    """The trial named in the job's params, else the newest ready trial with this role."""
    raw = ctx.job.params.get(key)
    trial = (
        ctx.repo.get_trial(UUID(str(raw)))
        if raw
        else ctx.repo.latest_trial(ctx.experiment_id, role, status="ready")
    )
    if trial is None or trial.experiment_id != ctx.experiment_id:
        raise StageError(f"There is no {role} clip to work on.", code="no_trial", retryable=False)
    return trial


def _scene(experiment: Experiment) -> SceneAnalysis:
    if experiment.scene is None:
        raise StageError(
            "The experiment has no scene analysis yet. Run the analysis first.",
            code="no_scene",
            retryable=False,
        )
    return experiment.scene


def _measure(ctx: JobContext, experiment: Experiment, trial: Trial, scene: SceneAnalysis):
    """Tracking, calibration and trajectory for one trial: the MEASURE steps shared by analysis
    and prove-it. Persists tracks, the calibration and the trajectory."""
    ctx.phase("tracking")
    tracking = _stage(stages.track_objects, ctx, trial, scene)
    for track in tracking.tracks:
        ctx.repo.upsert_track(track)

    ctx.phase("calibrating")
    calibration = _stage(stages.calibrate, ctx, experiment, trial, scene, tracking)
    ctx.repo.update_trial(trial.id, calibration=calibration.calibration)
    if calibration.paths:
        ctx.repo.merge_trial_paths(trial.id, calibration.paths)
    trial = ctx.repo.get_trial(trial.id) or trial

    trajectory = _stage(
        stages.extract_trajectory, ctx, experiment, trial, scene, tracking, calibration
    )
    ctx.repo.upsert_trajectory(
        TrajectoryRecord(
            trial_id=trial.id, parquet_path=trajectory.parquet_path, summary=trajectory.summary
        )
    )
    return trial, calibration, trajectory


def _save_ghost(ctx: JobContext, trial: Trial, result: GhostResult) -> GhostRender:
    ghost = ctx.repo.add_ghost_render(
        GhostRender(
            experiment_id=ctx.experiment_id,
            trial_id=trial.id,
            version=result.version or 0,
            model_id=result.model_id,
            video_path=result.video_path,
            track_path=result.track_path,
            slow_video_path=result.slow_video_path,
            divergence_onset_s=result.divergence_onset_s,
            max_error_mm=result.max_error_mm,
            threshold_mm=result.threshold_mm,
            predicted_ahead_s=result.predicted_ahead_s,
        )
    )
    ctx.commit()
    return ghost


def ingest_trial(ctx: JobContext) -> None:
    trial = ctx.repo.get_trial(UUID(str(ctx.job.params["trial_id"])))
    if trial is None or trial.experiment_id != ctx.experiment_id:
        raise StageError("The uploaded clip no longer exists.", code="no_trial", retryable=False)
    ctx.phase("normalizing")
    result = _stage(stages.ingest, ctx, trial)
    ctx.repo.update_trial(
        trial.id,
        status="ready",
        error=None,
        probe=result.probe,
        timing=result.timing,
        n_frames=result.n_frames,
        duration_s=result.duration_s,
        paths={**trial.paths, **result.paths},
        meta_file_ids=result.meta_file_ids,
        trim=result.trim,
    )
    for warning in result.warnings:
        ctx.notebook.warning("Clip warning", warning)


def analyze_experiment(ctx: JobContext) -> None:
    experiment = _experiment(ctx)
    trial = _trial(ctx, "fit")

    ctx.phase("watching")
    scene = _stage(stages.analyze_scene, ctx, experiment, trial)
    updates: dict[str, Any] = {"scene": scene}
    if experiment.system_type == "auto":
        updates["system_type"] = scene.system_type
    experiment = ctx.repo.update_experiment(experiment.id, **updates) or experiment
    if not scene.proceed:
        raise StageError(
            scene.reason_if_not or "Ghostline can't measure this clip.",
            code="cannot_measure",
            retryable=False,
        )

    trial, calibration, trajectory = _measure(ctx, experiment, trial, scene)

    # The physicist stage enters 'hypothesizing' and 'fitting' itself and saves models and fits.
    conclusion = _stage(
        stages.run_physicist, ctx, experiment, trial, scene, trajectory, calibration
    )
    experiment = (
        ctx.repo.update_experiment(
            experiment.id,
            conclusion=conclusion,
            winner_model_id=conclusion.winner_id,
            confidence=conclusion.confidence,
        )
        or experiment
    )

    ctx.phase("peer_review")
    outcome = _stage(stages.run_review, ctx, experiment, conclusion)
    ctx.repo.update_experiment(
        experiment.id,
        review_status=outcome.review_status,
        conclusion=outcome.conclusion,
        winner_model_id=outcome.conclusion.winner_id,
        confidence=outcome.conclusion.confidence,
    )


def prove_trial(ctx: JobContext) -> None:
    experiment = _experiment(ctx)
    scene = _scene(experiment)
    prove = _trial(ctx, "prove")
    fit_trial = ctx.repo.latest_trial(experiment.id, "fit", status="ready")
    if fit_trial is None:
        raise StageError("The first clip is missing.", code="no_trial", retryable=False)

    prove, _calibration, trajectory = _measure(ctx, experiment, prove, scene)
    # The ghost stage enters 'predicting' and 'rendering' itself.
    result = _stage(stages.predict_ghost, ctx, experiment, fit_trial, prove, scene, trajectory)
    ghost = _save_ghost(ctx, prove, result)

    onset = (
        f"The real object drifted from the ghost after {ghost.divergence_onset_s:.2f} s."
        if ghost.divergence_onset_s is not None
        else "The ghost stayed with the real object for the whole clip."
    )
    ctx.notebook.append(
        actor="harness",
        kind="ghost",
        title="Ghost verdict",
        body_md=onset,
        data={
            "version": ghost.version,
            "divergence_onset_s": ghost.divergence_onset_s,
            "max_error_mm": ghost.max_error_mm,
        },
        refs=[Ref(kind="ghost", id=str(ghost.id)), Ref(kind="trial", id=str(prove.id))],
    )


def revise(ctx: JobContext) -> None:
    experiment = _experiment(ctx)
    raw = ctx.job.params.get("ghost_id")
    ghosts = ctx.repo.list_ghosts(experiment.id)
    ghost = next((g for g in ghosts if str(g.id) == str(raw)), None) if raw else None
    ghost = ghost or (ghosts[-1] if ghosts else None)
    if ghost is None:
        raise StageError("There is no ghost to diagnose yet.", code="no_ghost", retryable=False)

    ctx.phase("diagnosing")
    result = _stage(stages.diagnose_revision, ctx, experiment, ghost)
    refs = [Ref(kind="ghost", id=str(ghost.id))]
    if result.revised_model is not None:
        ctx.repo.upsert_model(result.revised_model)
        refs.append(Ref(kind="model", id=result.revised_model.id))
    revision = {"diagnosis": result.diagnosis_md, "adopted": result.adopted, "note": ""}
    if result.adopted and result.ghost is not None:
        new_ghost = _save_ghost(ctx, _trial_of(ctx, ghost), result.ghost)
        revision["note"] = f"Ghost v{new_ghost.version} forecasts better than v{ghost.version}."
        ctx.repo.set_ghost_revision(new_ghost.id, revision)
        refs.append(Ref(kind="ghost", id=str(new_ghost.id)))
    else:
        revision["note"] = "The revision didn't forecast better beyond the noise, so v1 stays."
        ctx.repo.set_ghost_revision(ghost.id, revision)
    ctx.notebook.append(
        actor="wonder",
        kind="revision",
        title="Revision adopted" if result.adopted else "The revision didn't help",
        body_md=result.diagnosis_md,
        data={"adopted": result.adopted},
        refs=refs,
    )


def _trial_of(ctx: JobContext, ghost: GhostRender) -> Trial:
    trial = ctx.repo.get_trial(ghost.trial_id)
    if trial is None:
        raise StageError("The clip behind this ghost is gone.", code="no_trial", retryable=False)
    return trial


def rerun(ctx: JobContext) -> None:
    experiment = _experiment(ctx)
    model_id = str(ctx.job.params["model_id"])
    model = ctx.repo.get_model(experiment.id, model_id)
    if model is None:
        raise StageError(f"Model {model_id} no longer exists.", code="no_model", retryable=False)
    split_pct = float(ctx.job.params.get("split_pct", 0.65))
    guesses = {k: float(v) for k, v in (ctx.job.params.get("initial_guesses") or {}).items()}

    ctx.phase("fitting")
    fit = _stage(stages.rerun_model, ctx, experiment, model, split_pct, guesses)
    fit = ctx.repo.insert_fit(fit)
    ctx.commit()
    ctx.notebook.append(
        actor="harness",
        kind="fit",
        title=f"Re-ran {model.name} with a {split_pct * 100:.0f}% split",
        data={
            "fit_id": str(fit.id),
            "split_pct": split_pct,
            "initial_guesses": guesses,
            "train_rmse": fit.train_rmse,
            "heldout_rmse": fit.heldout_rmse,
            "params": [p.model_dump() for p in fit.params],
        },
        refs=[Ref(kind="model", id=model.id), Ref(kind="fit", id=str(fit.id))],
    )


def export_verify(ctx: JobContext) -> None:
    experiment = _experiment(ctx)
    ctx.phase("verifying")
    check = _stage(stages.verify_export, ctx, experiment)
    ctx.repo.update_experiment(experiment.id, replay_check=check)
    passed = bool(check.get("passed"))
    ctx.notebook.append(
        actor="harness",
        kind="observation",
        title="Reproduced in PyBullet" if passed else "Did not reproduce in PyBullet",
        data=check,
    )


HANDLERS: dict[JobKind, Handler] = {
    "ingest_trial": ingest_trial,
    "analyze_experiment": analyze_experiment,
    "prove_trial": prove_trial,
    "revise": revise,
    "rerun": rerun,
    "export_verify": export_verify,
}
