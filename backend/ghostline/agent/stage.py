"""Stage entry points for the physicist (the signatures of the stubs in jobs/stages.py).

    run_physicist(ctx, experiment, trial, scene, trajectory, calibration) -> Conclusion
    rerun_model(ctx, experiment, model, split_pct, initial_guesses) -> FitRecord

`run_physicist` is Parts 10-13 and 15: it builds the harness inputs from the trajectory, runs the
Wonder loop (loop.py) and, if Wonder can't be used, library-only mode (fallback.py). It enters
the `hypothesizing` and `fitting` phases itself and saves models, fits and notebook events as they
appear. Its last act is to write the session to `experiments/{id}/agent/session.json`: that file
is how `run_review` (a separate phase, possibly another container) continues the conversation and
reads the harness results (see session.py). Wiring: call these from jobs/stages.py.
"""

from __future__ import annotations

import logging

from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError, TransientStageError
from ghostline.jobs.stage_types import CalibrationResult, TrajectoryResult
from ghostline.llm import ReplayMissError
from ghostline.store.core_schemas import Conclusion, SceneAnalysis
from ghostline.store.records import Experiment, FitRecord, ModelRecord, Trial

from . import rerun as rerun_mod
from .clients import get_clients
from .fallback import run_library_only
from .inputs import InputsError, build_inputs
from .lab import Lab
from .loop import PhysicistLoop, WonderUnavailable, load_prompt, wonder_disabled
from .runners import RunnerPool

log = logging.getLogger("ghostline.agent")


def run_physicist(
    ctx: JobContext,
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    trajectory: TrajectoryResult,
    calibration: CalibrationResult,
) -> Conclusion:
    ctx.phase("hypothesizing")
    try:
        inputs = build_inputs(
            experiment,
            trial,
            scene,
            parquet_path=trajectory.parquet_path,
            summary=trajectory.summary,
            calibration=calibration.calibration,
            paths=ctx.paths,
        )
    except InputsError as exc:
        raise StageError(str(exc), code="cannot_fit", retryable=False) from exc

    session = Lab.new_session(inputs)
    with RunnerPool() as pool:
        lab = Lab(ctx, experiment, trial, inputs, session, pool)
        view = inputs.training()
        ctx.notebook.append(
            actor="harness",
            kind="observation",
            title="The training window",
            body_md=(
                f"The motion window has {len(inputs.data.t)} measured points. Wonder works from "
                f"the first {round(inputs.primary_split * 100)}% ({len(view)} points, up to "
                f"t = {inputs.t_split:.2f} s). The harness scores every model on the rest, which "
                "Wonder never sees."
            ),
            data={
                "t_split": inputs.t_split,
                "n_train": len(view),
                "n_total": len(inputs.data.t),
                "fraction": inputs.primary_split,
                "flags": [f.model_dump() for f in inputs.flags],
            },
        )
        conclusion = _conclude(lab, scene, experiment)
        lab.save()
    return conclusion


def _conclude(lab: Lab, scene: SceneAnalysis, experiment: Experiment) -> Conclusion:
    model = lab.ctx.settings.physicist_model
    if wonder_disabled(model):
        return _library_only(lab, "Wonder is switched off for this deployment.")
    meta, _ = get_clients(lab.ctx)
    hint = experiment.description.strip() or experiment.title
    if experiment.system_type not in ("auto", ""):
        hint = f"{hint} (the user says it is a {experiment.system_type.replace('_', ' ')})"
    loop = PhysicistLoop(lab, meta, model=model, scene=scene, hint=hint)
    try:
        return loop.run()
    except WonderUnavailable as exc:
        log.warning("Wonder unavailable, finishing library-only: %s", exc)
        return _library_only(lab, str(exc))
    except ReplayMissError:
        raise
    except TransientStageError:
        raise


def _library_only(lab: Lab, reason: str) -> Conclusion:
    if not lab.library_ids():
        raise StageError(
            f"Wonder is unavailable and the library has no models for a "
            f"{lab.inputs.system!r} system. {reason}",
            code="no_library",
            retryable=False,
        )
    return run_library_only(lab, reason)


def rerun_model(
    ctx: JobContext,
    experiment: Experiment,
    model: ModelRecord,
    split_pct: float,
    initial_guesses: dict[str, float],
) -> FitRecord:
    return rerun_mod.rerun(ctx, experiment, model, split_pct, initial_guesses)


def prompt_version() -> str:
    """Short hash of the physicist prompt, to store with eval results."""
    return load_prompt()[1]
