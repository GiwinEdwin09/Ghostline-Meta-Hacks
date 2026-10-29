"""The two stage entry points for jobs/stages.py (build plan Parts 16 and 17).

    predict_ghost(ctx, experiment, fit_trial, prove_trial, scene, trajectory) -> GhostResult
    diagnose_revision(ctx, experiment, ghost) -> RevisionResult

Both have the signatures of the stubs in jobs/stages.py, plus keyword-only hooks with defaults for
tests and for the lead to wire (the fitting runner, the Wonder client, the reviewer round).
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import GhostResult, TrajectoryResult
from ghostline.physics.harness.schema import ModelSpec
from ghostline.physics.library import model_ids
from ghostline.physics.runner import InProcessRunner, Runner
from ghostline.store.core_schemas import SceneAnalysis
from ghostline.store.records import Experiment, Ref, Trial

from .divergence import measure_divergence
from .forecast import forecast_trial, freeze
from .loading import (
    SUPPORTED_SYSTEMS,
    constants_for,
    load_series,
    make_sprite,
    masked_track,
    trial_file,
    write_sprite_png,
)
from .model import N_DRAWS, Divergence, Forecast, GhostError, GhostTrack
from .plan import build_plan, is_slow_motion
from .render import render_ghost_video
from .series import TrialSeries
from .specs import library_spec, spec_from_record

log = logging.getLogger("ghostline.ghost")


@contextlib.contextmanager
def runner_for(runner: Runner | None, *specs: ModelSpec) -> Iterator[Runner]:
    """The fitting runner. Library models run in this process; Wonder-written code only ever runs
    in the no-network Modal Sandbox, so a job with custom code starts one."""
    if runner is not None:
        yield runner
    elif all(s.library_ref is not None for s in specs):
        yield InProcessRunner()
    else:
        from ghostline.physics.sandbox import FitSandbox

        with FitSandbox() as sandbox:
            yield sandbox  # type: ignore[misc]


def winner_spec(ctx: JobContext, experiment: Experiment, system: str) -> tuple[ModelSpec, str]:
    """The experiment's winning model as a harness spec, and its display name."""
    winner = experiment.winner_model_id
    if not winner:
        raise StageError(
            "The experiment has no winning model to predict with.",
            code="no_winner",
            retryable=False,
        )
    record = ctx.repo.get_model(experiment.id, winner)
    if record is not None:
        return spec_from_record(record), record.name
    if winner in model_ids(system):
        return library_spec(winner), winner
    raise StageError(f"Model {winner} is missing.", code="no_model", retryable=False)


def check_system(scene: SceneAnalysis) -> str:
    if scene.system_type not in SUPPORTED_SYSTEMS:
        raise StageError(
            f"Ghostline can't draw a ghost for a {scene.system_type!r} system yet.",
            code="unsupported_system",
            retryable=False,
        )
    return scene.system_type


def render_version(
    ctx: JobContext,
    experiment: Experiment,
    prove_trial: Trial,
    scene: SceneAnalysis,
    series2: TrialSeries,
    forecast: Forecast,
    div: Divergence,
    *,
    version: int,
    model_name: str,
) -> tuple[GhostResult, GhostTrack]:
    """Draw one ghost version on trial 2's master and write v{n}.mp4 (plus the slow-motion copy)
    and v{n}_track.json."""
    master = trial_file(ctx, prove_trial, "master")
    if master is None:
        raise StageError(
            "The second clip's master video is missing.", code="no_master", retryable=False
        )
    sprite = make_sprite(master, masked_track(ctx, prove_trial, scene, master), series2)
    write_sprite_png(ctx.paths.trial_file(experiment.id, prove_trial.id, "sprite.png"), sprite)
    slow = is_slow_motion(series2, experiment.declared_capture.mode == "slow_mo")
    plan, track = build_plan(
        forecast,
        div,
        series2,
        sprite,
        version=version,
        label=f"Ghost v{version} · {model_name}",
        slow_motion=slow,
        source_angle=sprite.source_angle,
    )
    video = ctx.paths.ghost_video(experiment.id, version)
    track_path = ctx.paths.ghost_track(experiment.id, version)
    slow_path = ctx.paths.ghost_slow(experiment.id, version) if slow else None
    stats = render_ghost_video(master, plan, video, slow_path, progress=ctx.progress)
    track_path.write_text(track.model_dump_json())
    log.info(
        "rendered ghost v%d: %d frames at %dx%d in %.1f s",
        version, stats.n_frames, stats.width, stats.height, stats.seconds,
    )  # fmt: skip
    rel = lambda p: ctx.paths.rel(experiment.id, p)  # noqa: E731
    return (
        GhostResult(
            version=version,
            model_id=forecast.model_id,
            video_path=rel(video),
            track_path=rel(track_path),
            slow_video_path=rel(slow_path) if slow_path else None,
            divergence_onset_s=track.onset_s,
            max_error_mm=div.max_error_mm,
        ),
        track,
    )


def predict_ghost(
    ctx: JobContext,
    experiment: Experiment,
    fit_trial: Trial,
    prove_trial: Trial,
    scene: SceneAnalysis,
    trajectory: TrajectoryResult,
    *,
    runner: Runner | None = None,
    n_draws: int = N_DRAWS,
) -> GhostResult:
    """Part 16: freeze trial 1's final fit, fit only the initial conditions to trial 2's launch
    window, predict the rest, and render the ghost video and its track."""
    system = check_system(scene)
    ctx.phase("predicting")
    spec, name = winner_spec(ctx, experiment, system)
    try:
        trial1 = load_series(ctx, fit_trial, scene)
        trial2 = load_series(
            ctx, prove_trial, scene,
            trajectory_path=trajectory.parquet_path, summary=trajectory.summary,
        )  # fmt: skip
        constants = constants_for(system, trial1, experiment)
        with runner_for(runner, spec) as run:
            frozen = freeze(run, spec, system, trial1, constants)
            ctx.progress(0.4)
            forecast = forecast_trial(run, spec, system, frozen, trial2, n_draws=n_draws)
        div = measure_divergence(forecast, trial2)
    except GhostError as error:
        raise StageError(str(error), code="ghost_failed", retryable=False) from error
    ctx.progress(1.0)
    _note_prediction(ctx, prove_trial, spec, name, frozen, forecast, div)

    ctx.phase("rendering")
    try:
        version = ctx.repo.next_ghost_version(experiment.id)
        result, _ = render_version(
            ctx, experiment, prove_trial, scene, trial2, forecast, div,
            version=version, model_name=name,
        )  # fmt: skip
    except GhostError as error:
        raise StageError(str(error), code="ghost_failed", retryable=False) from error
    return result


def _note_prediction(ctx, prove_trial, spec, name, frozen, forecast, div) -> None:
    laws = ", ".join(
        f"{k} = {v:.4g} ± {frozen.sigma[k]:.2g}"
        for k, v in frozen.values.items()
        if k in frozen.sigma
    )
    initial = ", ".join(f"{k} = {v:.4g}" for k, v in forecast.ic.values.items())
    ctx.notebook.append(
        actor="harness",
        kind="prediction",
        title=f"Ghost: {name} frozen from trial 1, launch window fitted",
        body_md=(
            f"The law is held fixed at trial 1's values ({laws or 'no free physical parameters'}). "
            f"Only the starting state ({initial}) was fitted, to the first "
            f"{forecast.launch_window_s:.2f} s of trial 2. Everything after that is a pure "
            f"prediction, {forecast.n_draws} Monte Carlo draws wide."
        ),
        data={
            "model_id": spec.id,
            "frozen": frozen.values,
            "frozen_sigma": frozen.sigma,
            "constants": frozen.constants,
            "initial_conditions": forecast.ic.values,
            "initial_condition_se": forecast.ic.se,
            "launch_window_s": forecast.launch_window_s,
            "threshold_mm": div.tau_mm,
            "n_draws": forecast.n_draws,
        },
        refs=[Ref(kind="trial", id=str(prove_trial.id))],
    )
    if forecast.ic.at_bound:
        ctx.notebook.warning(
            "Launch fit hit a limit",
            f"The fitted start {forecast.ic.at_bound} sits at the edge of its allowed range, so "
            "the launch may be outside what the model expects.",
        )
