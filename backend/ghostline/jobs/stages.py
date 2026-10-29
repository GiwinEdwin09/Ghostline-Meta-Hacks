"""Every pipeline stage the jobs call lives here.

The job handlers (jobs/handlers.py) and two API routes (export and simulate) call only these
functions. The five measurement stages (ingest, analyze_scene, track_objects, calibrate and
extract_trajectory) are wired to their modules in jobs/measure.py. The rest are stubs that raise
NotImplementedError until the owning module is merged and wired: replace the body with a call
into it and keep the signature, or change the signature and the one place in handlers.py that
calls it. Tests replace stages with fakes (monkeypatch.setattr).

Rules for implementations:
- Report with ctx.phase(...) and ctx.progress(...). Write notebook events with ctx.notebook.
- Return the typed results from jobs/stage_types.py; the handler persists them. The physicist and
  review stages also write models, fits and reviews incrementally through ctx.repo
  (upsert_model, insert_fit, insert_review), so the UI shows them as they appear.
- Raise StageError(message, retryable=False) for problems only a new clip can fix, and
  TransientStageError for failures worth retrying. Anything else fails the job with its message.
- All paths are relative to the experiment folder; use ctx.paths and ctx.paths.rel(...).
"""

from typing import Literal

from ghostline.jobs import measure
from ghostline.jobs.context import JobContext
from ghostline.jobs.stage_types import (
    CalibrationResult,
    GhostResult,
    IngestResult,
    ReviewOutcome,
    RevisionResult,
    SimulateRequest,
    SimulateResult,
    TrackingResult,
    TrajectoryResult,
)
from ghostline.store.core_schemas import Conclusion, SceneAnalysis, SimExport
from ghostline.store.records import Experiment, FitRecord, GhostRender, ModelRecord, Trial
from ghostline.store.repo import Repo

_WIRED_LATER = "wired in wave 2"


# --- ingest_trial -------------------------------------------------------------------------
def ingest(ctx: JobContext, trial: Trial) -> IngestResult:
    """Part 5 (ingest/): probe, timing model, master.mp4 and derived files, auto-trim, Meta Files
    upload. The original is already assembled at paths.original(...); trial.original_filename,
    trial.sha256 and the experiment's declared capture are the inputs. Part 28's upload checks
    (ffprobe, video streams only, frame limit) run first."""
    return measure.ingest(ctx, trial)


# --- analyze_experiment -------------------------------------------------------------------
def analyze_scene(ctx: JobContext, experiment: Experiment, trial: Trial) -> SceneAnalysis:
    """Part 6 (scene/): Wonder watches wonder_480.mp4 and returns the SceneAnalysis."""
    return measure.analyze_scene(ctx, experiment, trial)


def track_objects(ctx: JobContext, trial: Trial, scene: SceneAnalysis) -> TrackingResult:
    """Part 7 (vision/): SAM 3.1 tracks every object in scene.tracked_objects plus the sheet."""
    return measure.track_objects(ctx, trial, scene)


def calibrate(
    ctx: JobContext,
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    tracking: TrackingResult,
) -> CalibrationResult:
    """Part 8 (calib/): sheet corners, homography, plane-offset correction, gravity direction."""
    return measure.calibrate(ctx, experiment, trial, scene, tracking)


def extract_trajectory(
    ctx: JobContext,
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    tracking: TrackingResult,
    calibration: CalibrationResult,
) -> TrajectoryResult:
    """Part 9 (traj/): clean physical time series, noise estimate, events, quality flags."""
    return measure.extract_trajectory(ctx, experiment, trial, scene, tracking, calibration)


def run_physicist(
    ctx: JobContext,
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    trajectory: TrajectoryResult,
    calibration: CalibrationResult,
) -> Conclusion:
    """Parts 10-13 and 15 (agent/, physics/): the HYPOTHESIZE -> FIT -> ANALYZE -> CONCLUDE loop,
    or library-only mode. Calls ctx.phase('hypothesizing') and ctx.phase('fitting') itself and
    persists models and fits as they appear. The returned conclusion's confidence comes from
    confidence.py and its numbers from the harness."""
    raise NotImplementedError(_WIRED_LATER)


def run_review(ctx: JobContext, experiment: Experiment, conclusion: Conclusion) -> ReviewOutcome:
    """Part 14 (review/): Spark Open challenges the claim, Wonder responds, up to two rounds.
    Persists reviews with ctx.repo.insert_review. If the reviewer endpoint is down, return
    review_status 'not_reviewed' with the conclusion unchanged."""
    raise NotImplementedError(_WIRED_LATER)


# --- prove_trial and revise ---------------------------------------------------------------
def predict_ghost(
    ctx: JobContext,
    experiment: Experiment,
    fit_trial: Trial,
    prove_trial: Trial,
    scene: SceneAnalysis,
    trajectory: TrajectoryResult,
) -> GhostResult:
    """Part 16 (ghost/): freeze trial 1's parameters, fit initial conditions on trial 2's launch
    window, predict the rest, render the ghost video and track json. Calls ctx.phase('predicting')
    then ctx.phase('rendering')."""
    raise NotImplementedError(_WIRED_LATER)


def diagnose_revision(
    ctx: JobContext, experiment: Experiment, ghost: GhostRender
) -> RevisionResult:
    """Part 17 (ghost/, agent/): Wonder diagnoses the drift, a revised model is fitted on trial 1
    only, and ghost v2 is rendered when it forecasts better beyond noise."""
    raise NotImplementedError(_WIRED_LATER)


# --- rerun and export_verify --------------------------------------------------------------
def rerun_model(
    ctx: JobContext,
    experiment: Experiment,
    model: ModelRecord,
    split_pct: float,
    initial_guesses: dict[str, float],
) -> FitRecord:
    """Parts 11-12 and 20: refit one model with an edited split and initial guesses. Returns the
    new FitRecord; the handler inserts it and writes the notebook event."""
    raise NotImplementedError(_WIRED_LATER)


def verify_export(ctx: JobContext, experiment: Experiment) -> dict:
    """Part 19 (export/verify_pybullet.py): the PyBullet replay check. Returns the replay_check
    dict (simulator, quantity, simulated, measured, passed)."""
    raise NotImplementedError(_WIRED_LATER)


# --- called straight from API routes (no job) ---------------------------------------------
def build_export(
    repo: Repo, experiment: Experiment, fmt: Literal["json", "pybullet"], mass_g: float | None
) -> SimExport | str:
    """Part 19 (export/): the SimExport for fmt='json', or the PyBullet snippet text."""
    raise NotImplementedError(_WIRED_LATER)


def simulate_custom_model(
    repo: Repo, experiment: Experiment, model: ModelRecord, request: SimulateRequest
) -> SimulateResult:
    """Part 18: run a model in the sandbox and return its trajectory. The API gives it
    SIMULATE_TIMEOUT_S (20 s) and answers 504 if it takes longer."""
    raise NotImplementedError(_WIRED_LATER)
