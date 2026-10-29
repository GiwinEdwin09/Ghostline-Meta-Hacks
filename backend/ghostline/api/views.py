"""What the web app reads: experiments, trials, models and the rest in the shapes the UI draws
(build plan Parts 20, 21 and 24).

The tables hold what the pipeline produced (paths, JSON dumps, ids). These models join them and
turn them into what a page needs: media URLs, quality flags, one fit per model, a conclusion with
its confidence reasons and uncertainty budget. The generated TypeScript types (web/src/lib/
api-types.ts) come from these classes, so a field renamed here fails the web build.

Media fields are paths relative to GET /api/media/ ("{experiment}/trials/{trial}/poster.jpg"),
or "api/..." paths for the one computed file (the trajectory JSON). `resolveMedia()` in the web
client turns either into a URL.

Contract with the stages that write the rows (listed in the report that came with this change):
- FitRecord.diagnostics may hold reduced_chi2, lag1_autocorr, durbin_watson, runs_p,
  systematic_residual, nrmse, time_to_divergence_s, at_bound (parameter names),
  heldout_rmse_by_split ({"0.50": rmse, ...}), budget ({quantity: {component: sigma}}) and series
  ({t, observed, predicted, split_t, unit, label}). FitRecord.split.kind is "selection", "final"
  or "rerun". plot_paths maps "fit" and "residuals" to paths relative to the experiment folder.
- The notebook event titled "Confidence: <level>" carries data {cant_tell, reasons[{code, level,
  text}], fixes[]}.
- ReviewRecord.response may hold {"response_md": "..."}.
- The ghost track file is GhostTrack JSON (ghost/v{n}_track.json).
"""

import logging
import math
import re
from typing import Any, Literal
from uuid import UUID

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ghostline.api.errors import ApiError
from ghostline.calib.model import Calibration
from ghostline.store.core_schemas import Conclusion, Confidence, ParamSpec
from ghostline.store.records import (
    Event,
    Experiment,
    FitRecord,
    GhostRender,
    Job,
    ModelRecord,
    ReviewRecord,
    TrackRecord,
    TrajectoryRecord,
    Trial,
)
from ghostline.store.repo import Repo

log = logging.getLogger("ghostline.api")

SystemName = Literal[
    "free_fall", "projectile", "pendulum", "ramp", "bounce", "spin_down", "oscillator", "other"
]
ViewStatus = Literal["draft", "ingesting", "ready", "analyzing", "complete", "failed", "blocked"]
ReviewStatusName = Literal[
    "accepted", "accepted_with_caveats", "contested", "not_reviewed", "pending"
]

# Library models whose parameters a simulator can take (Part 19's table).
EXPORTABLE_MODELS = {
    "R1", "R3", "B0", "B1", "B2", "FF1", "FF2", "P2", "P3", "P4", "S0", "S1", "S2", "S3",
}  # fmt: skip
# Notices that undercut a claim (Part 15's major flags).
MAJOR_CODES = {"out_of_plane", "looks_retimed", "no_motion", "retime_unconfirmed"}


def media(experiment_id: UUID | str, rel: str | None) -> str | None:
    return f"{experiment_id}/{rel}" if rel else None


def pretty(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


# --- quality flags ----------------------------------------------------
class QualityFlagView(BaseModel):
    id: str
    severity: Literal["info", "minor", "major"]
    title: str
    detail: str


def _flag(code: str, severity: str, message: str) -> QualityFlagView:
    level = {"info": "info", "warn": "minor", "error": "major"}.get(severity, severity)
    if level not in ("info", "minor", "major"):
        level = "minor"
    if code in MAJOR_CODES:
        level = "major"
    return QualityFlagView(id=code, severity=level, title=pretty(code), detail=message)  # type: ignore[arg-type]


def notice_flags(notices: list[dict[str, Any]] | None) -> list[QualityFlagView]:
    return [
        _flag(str(n.get("code", "notice")), str(n.get("severity", "warn")), str(n.get("message")))
        for n in notices or []
    ]


# --- trials ----------------------------------------------------
class TimingView(BaseModel):
    classification: Literal["cfr", "vfr_jitter", "mixed", "retimed"]
    detected_fps: float
    declared_mode: Literal["normal", "slow_mo"]
    declared_fps: float
    time_factor: float
    jitter_pct: float
    warnings: list[str]


class CalibrationView(BaseModel):
    paper: Literal["a4", "letter"]
    mm_per_px: float
    sigma_scale_rel: float
    plane_offset_cm: float
    gravity_source: Literal["data", "plumb", "camera_level"]
    ramp_angle_deg: float | None
    camera_motion_px: float
    # Row-major 3x3 from physics-frame metres (x right, y up) to image pixels, with the
    # plane-offset correction folded in, for placing a simulated object on the user's frame.
    world_to_px: list[float] | None


class TrimView(BaseModel):
    start_s: float
    end_s: float
    suggested_start_s: float
    suggested_end_s: float


class TrialMediaView(BaseModel):
    playback_url: str
    poster_url: str | None
    calib_overlay_url: str | None
    trajectory_url: str | None  # JSON, TrajectoryView
    sam_events_url: str | None  # JSONL of raw SAM events


class TrialView(BaseModel):
    id: UUID
    role: Literal["fit", "prove"]
    status: Literal["uploading", "ingesting", "ready", "tracking", "tracked", "failed"]
    width: int
    height: int
    n_frames: int
    duration_s: float
    timing: TimingView | None
    trim: TrimView
    calibration: CalibrationView | None
    quality_flags: list[QualityFlagView]
    media: TrialMediaView
    error: str | None


def timing_view(trial: Trial, declared_mode: str, declared_fps: float | None) -> TimingView | None:
    t = trial.timing
    if not t:
        return None
    classification = "retimed" if t.get("retimed") else t.get("classification", "cfr")
    fps = _num(t.get("real_fps"), declared_fps or 0)
    return TimingView(
        classification=classification,
        detected_fps=round(fps, 3),
        declared_mode="slow_mo" if declared_mode == "slow_mo" else "normal",
        declared_fps=declared_fps or round(fps, 3),
        time_factor=_num(t.get("time_factor"), 1.0),
        jitter_pct=round(_num(t.get("jitter_rel")) * 100, 3),
        warnings=[str(w.get("message", "")) for w in t.get("warnings", [])],
    )


def world_to_px(calibration: Calibration) -> list[float] | None:
    """The projective map from physics-frame metres to pixels, from where four points land."""
    try:
        world = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        px = calibration.world_to_px(world).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(world.astype(np.float32), px)
        return [float(v) for v in (matrix / matrix[2, 2]).ravel()]
    except Exception:
        log.warning("could not build the world-to-pixel map", exc_info=True)
        return None


def calibration_view(trial: Trial) -> CalibrationView | None:
    if not trial.calibration:
        return None
    try:
        cal = Calibration.model_validate(trial.calibration)
    except Exception:
        log.warning("stored calibration does not validate")
        return None
    return CalibrationView(
        paper="letter" if cal.paper == "Letter" else "a4",
        mm_per_px=cal.m_per_px * 1000,
        sigma_scale_rel=cal.uncertainty.sigma_scale_rel,
        plane_offset_cm=cal.offset.d_m * 100,
        gravity_source=cal.gravity.source,
        ramp_angle_deg=cal.ramp.angle_deg if cal.ramp else None,
        camera_motion_px=cal.camera_motion.max_shift_px if cal.camera_motion else 0.0,
        world_to_px=world_to_px(cal),
    )


def trial_flags(trial: Trial, trajectory: TrajectoryRecord | None) -> list[QualityFlagView]:
    timing = trial.timing or {}
    flags = notice_flags(timing.get("warnings"))
    if timing.get("needs_confirmation"):
        flags.append(
            _flag("retime_unconfirmed", "warn", "The time base of this clip needs confirming.")
        )
    flags += notice_flags((trial.calibration or {}).get("notices"))
    if trajectory:
        flags += notice_flags(trajectory.summary.get("notices"))
    seen: set[str] = set()
    return [f for f in flags if not (f.id in seen or seen.add(f.id))]


def sam_events_path(tracks: list[TrackRecord], trial_id: UUID) -> str | None:
    """The raw SAM stream of the object the trajectory follows: the first body that moved."""
    mine = [t for t in tracks if t.trial_id == trial_id and t.stats.get("events_path")]
    bodies = [t for t in mine if t.stats.get("role") == "moving_body"] or mine
    return bodies[0].stats["events_path"] if bodies else None


def trial_view(
    experiment: Experiment,
    trial: Trial,
    tracks: list[TrackRecord],
    trajectories: list[TrajectoryRecord],
) -> TrialView:
    trajectory = next((t for t in trajectories if t.trial_id == trial.id), None)
    paths = trial.paths
    probe = trial.probe or {}
    trim = trial.trim
    duration = trial.duration_s or 0.0
    status = {"uploaded": "uploading"}.get(trial.status, trial.status)
    if trial.status == "ready" and trajectory is not None:
        status = "tracked"
    start = trim.start_s if trim else 0.0
    end = trim.end_s if trim else duration
    exp_id = experiment.id
    return TrialView(
        id=trial.id,
        role=trial.role,
        status=status,  # type: ignore[arg-type]
        width=int(probe.get("display_width", 0)),
        height=int(probe.get("display_height", 0)),
        n_frames=trial.n_frames or 0,
        duration_s=duration,
        timing=timing_view(
            trial, experiment.declared_capture.mode, experiment.declared_capture.fps
        ),
        trim=TrimView(
            start_s=start,
            end_s=end,
            suggested_start_s=trim.suggested_start_s
            if trim and trim.suggested_start_s is not None
            else start,
            suggested_end_s=trim.suggested_end_s
            if trim and trim.suggested_end_s is not None
            else end,
        ),
        calibration=calibration_view(trial),
        quality_flags=trial_flags(trial, trajectory),
        media=TrialMediaView(
            playback_url=media(exp_id, paths.get("playback_720")) or "",
            poster_url=media(exp_id, paths.get("poster")),
            calib_overlay_url=media(exp_id, paths.get("calib_overlay")),
            trajectory_url=(
                f"api/experiments/{exp_id}/trials/{trial.id}/trajectory" if trajectory else None
            ),
            sam_events_url=media(exp_id, sam_events_path(tracks, trial.id)),
        ),
        error=trial.error,
    )


# --- the trajectory JSON ------------------------------------------------------------------------
class TrajectoryEvent(BaseModel):
    kind: Literal["release", "impact", "stop"]
    t: float


class TrajectoryView(BaseModel):
    """Positions in metres, angles in radians, time in real seconds. Arrays line up by index;
    a missing value (a dropped frame) is null."""

    t: list[float]
    x: list[float | None] | None
    y: list[float | None] | None
    theta: list[float | None] | None
    phi: list[float | None] | None
    omega: list[float | None] | None
    px: list[list[float] | None]  # pixel centroid per row, for the trail on the video
    split_t: float  # end of the training window: the first 65% of the motion
    noise_sigma: float
    main: Literal["x", "y", "theta", "phi"]
    main_unit: str
    events: list[TrajectoryEvent]


def _column(df: pd.DataFrame, name: str) -> list[float | None] | None:
    if name not in df:
        return None
    return [float(v) if np.isfinite(v) else None for v in df[name].to_numpy(dtype=float)]


def trajectory_view(
    frame: pd.DataFrame, summary: dict[str, Any], tracks: pd.DataFrame | None, key: str | None
) -> TrajectoryView:
    angle = summary.get("angle_column")
    if angle in ("theta", "phi"):
        main, unit = angle, "rad"
        noise = _num(summary.get("noise", {}).get("sigma_angle_rad"))
    else:
        main = "y" if np.ptp(frame["y"]) >= np.ptp(frame["x"]) else "x"
        unit = "m"
        noise = _num(summary.get("noise", {}).get("sigma_pos_m"))
    window = summary.get("motion_window") or {}
    t0 = _num(window.get("t_start"), float(frame["t"].iloc[0]))
    t1 = _num(window.get("t_end"), float(frame["t"].iloc[-1]))
    px: list[list[float] | None] = [None] * len(frame)
    if tracks is not None and key is not None and "frame" in frame:
        mine = tracks[tracks["object_key"] == key].set_index("frame")
        for i, f in enumerate(frame["frame"].to_numpy()):
            if f in mine.index:
                px[i] = [float(mine.at[f, "cx"]), float(mine.at[f, "cy"])]
    events = summary.get("events") or {}
    marks = []
    if events.get("release_t") is not None:
        marks.append(TrajectoryEvent(kind="release", t=events["release_t"]))
    marks += [TrajectoryEvent(kind="impact", t=t) for t in events.get("impacts_t") or []]
    if events.get("stop_t") is not None:
        marks.append(TrajectoryEvent(kind="stop", t=events["stop_t"]))
    return TrajectoryView(
        t=[float(v) for v in frame["t"].to_numpy()],
        x=_column(frame, "x"),
        y=_column(frame, "y"),
        theta=_column(frame, "theta"),
        phi=_column(frame, "phi"),
        omega=_column(frame, "omega"),
        px=px,
        split_t=t0 + 0.65 * (t1 - t0),
        noise_sigma=noise,
        main=main,
        main_unit=unit,
        events=marks,
    )


# --- models and fits ----------------------------------------------------
class ParamEstimateView(BaseModel):
    name: str
    label: str
    value: float
    ci_low: float
    ci_high: float
    unit: str
    kind: Literal["physical", "initial_condition", "nuisance"]
    at_bound: bool


class FitDiagnosticsView(BaseModel):
    reduced_chi2: float
    lag1_autocorr: float
    durbin_watson: float
    runs_p: float
    systematic_residual: bool


class FitSeriesView(BaseModel):
    t: list[float]
    observed: list[float]
    predicted: list[float]
    split_t: float
    unit: str
    label: str


class PlotUrlsView(BaseModel):
    fit: str | None
    residuals: str | None


class FitView(BaseModel):
    id: UUID
    params: list[ParamEstimateView]
    train_rmse: float
    heldout_rmse: float
    heldout_ci: list[float]
    rmse_unit: str
    nrmse: float
    time_to_divergence_s: float | None
    aic: float
    bic: float
    diagnostics: FitDiagnosticsView
    plot_urls: PlotUrlsView
    series: FitSeriesView | None
    splits_won: int


ModelStatusName = Literal["proposed", "winner", "tied", "rejected", "invalid"]


class ModelView(BaseModel):
    id: str
    name: str
    round: int
    source: Literal["library", "wonder"]
    library_ref: str | None
    mechanism: str
    equations_latex: str
    prior_rank: int
    refuting_signature: str
    status: ModelStatusName
    rejection_reason: str | None
    fit: FitView | None


def _at_bound(name: str, value: float, spec: ParamSpec | None, listed: list[str]) -> bool:
    if name in listed:
        return True
    if spec is None or spec.upper <= spec.lower:
        return False
    edge = 0.01 * (spec.upper - spec.lower)
    return value <= spec.lower + edge or value >= spec.upper - edge


def fit_view(
    fit: FitRecord,
    model: ModelRecord,
    experiment_id: UUID,
    unit: str,
    splits_won: int,
) -> FitView:
    d = fit.diagnostics or {}
    specs = model.params_spec or {}
    listed = [str(n) for n in d.get("at_bound") or []]
    params = [
        ParamEstimateView(
            name=p.name,
            label=p.name,
            value=p.value,
            ci_low=p.ci_low,
            ci_high=p.ci_high,
            unit=p.unit,
            kind=specs[p.name].kind if p.name in specs else "physical",
            at_bound=_at_bound(p.name, p.value, specs.get(p.name), listed),
        )
        for p in fit.params
    ]
    ci = fit.heldout_ci or [fit.heldout_rmse or 0.0, fit.heldout_rmse or 0.0]
    series = None
    if isinstance(d.get("series"), dict):
        try:
            series = FitSeriesView.model_validate(d["series"])
        except Exception:
            series = None
    plots = fit.plot_paths or {}
    return FitView(
        id=fit.id,
        params=params,
        train_rmse=_num(fit.train_rmse),
        heldout_rmse=_num(fit.heldout_rmse),
        heldout_ci=[_num(ci[0]), _num(ci[-1])],
        rmse_unit=str(d.get("rmse_unit") or unit),
        nrmse=_num(d.get("nrmse")),
        time_to_divergence_s=d.get("time_to_divergence_s"),
        aic=_num(fit.aic),
        bic=_num(fit.bic),
        diagnostics=FitDiagnosticsView(
            reduced_chi2=_num(d.get("reduced_chi2")),
            lag1_autocorr=_num(d.get("lag1_autocorr")),
            durbin_watson=_num(d.get("durbin_watson")),
            runs_p=_num(d.get("runs_p")),
            systematic_residual=bool(d.get("systematic_residual")),
        ),
        plot_urls=PlotUrlsView(
            fit=media(experiment_id, plots.get("fit")),
            residuals=media(experiment_id, plots.get("residuals")),
        ),
        series=series,
        splits_won=splits_won,
    )


def comparison_fits(fits: list[FitRecord]) -> dict[str, FitRecord]:
    """The fit each model is judged by: its newest held-out comparison (a selection or a
    re-run), else its newest fit of any kind."""
    best: dict[str, FitRecord] = {}
    for fit in sorted(fits, key=lambda f: f.created_at or 0):  # type: ignore[arg-type,return-value]
        if (fit.split or {}).get("kind", "selection") != "final" or fit.model_id not in best:
            best[fit.model_id] = fit
    return best


def splits_won_by(fits: dict[str, FitRecord]) -> dict[str, int]:
    """How many of the 50/65/80% splits each model predicted best."""
    per_split: dict[str, dict[str, float]] = {}
    for model_id, fit in fits.items():
        for split, rmse in (fit.diagnostics.get("heldout_rmse_by_split") or {}).items():
            if rmse is not None:
                per_split.setdefault(str(split), {})[model_id] = float(rmse)
    wins = dict.fromkeys(fits, 0)
    for scores in per_split.values():
        wins[min(scores, key=lambda k: scores[k])] += 1
    return wins


def model_view(
    model: ModelRecord, fit: FitRecord | None, experiment_id: UUID, unit: str, won: int
) -> ModelView:
    status = model.status if model.status in ModelStatusName.__args__ else "proposed"  # type: ignore[attr-defined]
    return ModelView(
        id=model.id,
        name=model.name,
        round=model.round,
        source=model.source,
        library_ref=model.library_ref,
        mechanism=model.mechanism,
        equations_latex=model.equations_latex,
        prior_rank=model.prior_rank or 0,
        refuting_signature=model.refuting_signature,
        status=status,  # type: ignore[arg-type]
        rejection_reason=model.rejection_reason,
        fit=fit_view(fit, model, experiment_id, unit, won) if fit else None,
    )


# --- the conclusion ----------------------------------------------------
class BudgetTermView(BaseModel):
    label: str
    sigma: float


class QuantityView(BaseModel):
    name: str
    symbol_latex: str
    value: float
    ci_low: float
    ci_high: float
    unit: str
    budget: list[BudgetTermView] | None
    note: str | None


class RejectedView(BaseModel):
    id: str
    reason: str
    evidence: str


class CantTellView(BaseModel):
    reason: str
    fixes: list[str]


class ConclusionView(BaseModel):
    winner_id: str
    quantities: list[QuantityView]
    rejected: list[RejectedView]
    confidence: Confidence
    confidence_reasons: list[str]
    caveats: list[str]
    narrative_md: str
    cant_tell: CantTellView | None


def conclusion_view(
    conclusion: Conclusion, winner_fit: FitRecord | None, confidence_event: Event | None
) -> ConclusionView:
    budgets = (winner_fit.diagnostics.get("budget") if winner_fit else None) or {}
    data = confidence_event.data if confidence_event else {}
    reasons = [str(r.get("text", "")) for r in data.get("reasons") or [] if r.get("text")]
    fixes = [str(f) for f in data.get("fixes") or []]
    low_reasons = [str(r["text"]) for r in data.get("reasons") or [] if r.get("level") == "low"]
    cant_tell = None
    if data.get("cant_tell") or (conclusion.confidence == "low" and fixes):
        reason = " ".join(low_reasons or reasons[:1]) or "The data can't settle which law is right."
        cant_tell = CantTellView(reason=reason, fixes=fixes)
    quantities = []
    for q in conclusion.quantities:
        terms = budgets.get(q.name)
        quantities.append(
            QuantityView(
                name=q.name,
                symbol_latex=q.name,
                value=q.value,
                ci_low=q.ci_low,
                ci_high=q.ci_high,
                unit=q.unit,
                budget=(
                    [BudgetTermView(label=pretty(k), sigma=_num(v)) for k, v in terms.items()]
                    if isinstance(terms, dict)
                    else None
                ),
                note=None,
            )
        )
    return ConclusionView(
        winner_id=conclusion.winner_id,
        quantities=quantities,
        rejected=[
            RejectedView(
                id=str(r.get("id", "")),
                reason=str(r.get("reason", "")),
                evidence=str(r.get("evidence", "")),
            )
            for r in conclusion.rejected
        ],
        confidence=conclusion.confidence,
        confidence_reasons=reasons,
        caveats=conclusion.caveats,
        narrative_md=conclusion.narrative_md,
        cant_tell=cant_tell,
    )


def headline(conclusion: Conclusion | None) -> str | None:
    if conclusion is None or not conclusion.quantities:
        return None
    q = conclusion.quantities[0]
    half = (q.ci_high - q.ci_low) / 2
    return f"{q.name} = {q.value:.3g} ± {half:.2g} {q.unit}".strip()


# --- review, ghosts ----------------------------------------------------
class ReviewIssueView(BaseModel):
    id: str
    severity: Literal["minor", "major"]
    claim: str
    evidence: str
    requested_action: Literal["run_model", "refit", "lower_confidence", "add_caveat", "none"]
    detail: str


class ReviewRoundView(BaseModel):
    round: int
    verdict: Literal["accept", "accept_with_caveats", "challenge"]
    issues: list[ReviewIssueView]
    summary: str
    response_md: str | None


class ReviewThreadView(BaseModel):
    status: ReviewStatusName
    rounds: list[ReviewRoundView]


def review_view(experiment: Experiment, reviews: list[ReviewRecord]) -> ReviewThreadView | None:
    if not reviews and not experiment.review_status:
        return None
    status = experiment.review_status or "pending"
    if status not in ReviewStatusName.__args__:  # type: ignore[attr-defined]
        status = "pending"
    return ReviewThreadView(
        status=status,  # type: ignore[arg-type]
        rounds=[
            ReviewRoundView(
                round=r.round,
                verdict=r.verdict,
                issues=[ReviewIssueView.model_validate(i.model_dump()) for i in r.issues],
                summary=r.summary,
                response_md=(r.response or {}).get("response_md"),
            )
            for r in reviews
        ],
    )


class RevisionView(BaseModel):
    diagnosis: str
    adopted: bool
    note: str


class GhostView(BaseModel):
    id: UUID
    version: int
    trial_id: UUID
    model_id: str
    model_name: str
    video_url: str
    slow_video_url: str | None
    track_url: str  # JSON, GhostTrack
    poster_url: str | None
    divergence_onset_s: float | None
    max_error_mm: float
    threshold_mm: float
    predicted_ahead_s: float
    revision: RevisionView | None


def ghost_view(
    ghost: GhostRender,
    experiment_id: UUID,
    model_names: dict[str, str],
    posters: dict[UUID, str],
    read_track,
) -> GhostView:
    threshold, ahead = ghost.threshold_mm, ghost.predicted_ahead_s
    if (threshold is None or ahead is None) and ghost.track_path:
        track = read_track(ghost.track_path) or {}
        t = track.get("t") or [0.0]
        threshold = _num(track.get("tau_mm")) if threshold is None else threshold
        ahead = (
            max(_num(t[-1]) - _num(track.get("launch_window_s")), 0.0) if ahead is None else ahead
        )
    return GhostView(
        id=ghost.id,
        version=ghost.version,
        trial_id=ghost.trial_id,
        model_id=ghost.model_id or "",
        model_name=model_names.get(ghost.model_id or "", ghost.model_id or ""),
        video_url=media(experiment_id, ghost.video_path) or "",
        slow_video_url=media(experiment_id, ghost.slow_video_path),
        track_url=media(experiment_id, ghost.track_path) or "",
        poster_url=media(experiment_id, posters.get(ghost.trial_id)),
        divergence_onset_s=ghost.divergence_onset_s,
        max_error_mm=_num(ghost.max_error_mm),
        threshold_mm=_num(threshold),
        predicted_ahead_s=_num(ahead),
        revision=RevisionView.model_validate(ghost.revision) if ghost.revision else None,
    )


# --- what-if ----------------------------------------------------
class ObjectView(BaseModel):
    name: str
    radius_m: float
    mass_kg: float | None


class WhatIfView(BaseModel):
    system: Literal["free_fall", "pendulum", "ramp", "bounce", "spin_down"]
    model_ref: str
    params: dict[str, float]
    fixed: dict[str, float]
    initial: dict[str, float]
    measured_range: dict[str, list[float]]
    object: ObjectView
    still_url: str | None
    sim_supported: bool
    anchor_m: list[float]


# --- notebook events ----------------------------------------------------
class AttachmentView(BaseModel):
    kind: Literal["image", "video", "file"]
    url: str
    caption: str | None


class RefView(BaseModel):
    kind: str
    id: str


class EventView(BaseModel):
    id: int
    experiment_id: UUID
    seq: int
    ts: str
    phase: str
    actor: Literal["wonder", "spark_open", "harness", "sam", "system", "user"]
    kind: str
    title: str
    body_md: str
    data: dict[str, Any]
    attachments: list[AttachmentView]
    refs: list[RefView]


def event_view(event: Event) -> EventView:
    return EventView(
        id=event.id,
        experiment_id=event.experiment_id,
        seq=event.seq,
        ts=event.ts.isoformat(),
        phase=event.phase or "",
        actor=event.actor,
        kind=event.kind,
        title=event.title,
        body_md=event.body_md,
        data=event.data,
        attachments=[
            AttachmentView(
                kind="file" if a.kind == "json" else a.kind,
                url=f"{event.experiment_id}/{a.path}",
                caption=a.caption,
            )
            for a in event.attachments
        ],
        refs=[RefView(kind=r.kind, id=r.id) for r in event.refs],
    )


# --- experiments ----------------------------------------------------
class SetupView(BaseModel):
    plane_offset_cm: float
    paper: Literal["a4", "letter"]
    pendulum_length_cm: float | None


class DeclaredCaptureView(BaseModel):
    mode: Literal["normal", "slow_mo"]
    fps: float | None


class ExperimentSummaryView(BaseModel):
    id: UUID
    title: str
    system_type: SystemName
    status: ViewStatus
    created_at: str
    is_example: bool
    confidence: Confidence | None
    review_status: ReviewStatusName | None
    headline: str | None
    winner_name: str | None
    thumbnail_url: str | None
    blurb: str | None


class ExperimentView(ExperimentSummaryView):
    description: str
    declared_capture: DeclaredCaptureView
    setup: SetupView
    scene: "SceneView | None"
    trials: list[TrialView]
    models: list[ModelView]
    conclusion: ConclusionView | None
    review: ReviewThreadView | None
    ghosts: list[GhostView]
    whatif: WhatIfView | None
    export_formats: list[Literal["json", "pybullet"]]
    quality_flags: list[QualityFlagView]
    job: Job | None
    error: str | None
    last_seq: int
    cost_usd: float | None = None  # only for requests that carry the passcode


class TrackedObjectView(BaseModel):
    key: str
    sam_prompt: str
    role: Literal["moving_body", "marker", "pivot", "surface"]
    why: str


class SceneView(BaseModel):
    system_type: SystemName
    description: str
    tracked_objects: list[TrackedObjectView]
    object_properties: dict[str, Any]
    initial_hypotheses: list[dict[str, Any]]
    quality_issues: list[dict[str, Any]]
    proceed: bool
    reason_if_not: str | None


def _system(name: str) -> SystemName:
    return name if name in SystemName.__args__ else "other"  # type: ignore[attr-defined,return-value]


def _status(experiment: Experiment, jobs: list[Job]) -> ViewStatus:
    if experiment.status == "failed":
        latest = next((j for j in jobs if j.kind == "analyze_experiment"), None)
        if latest and latest.error_code == "cannot_measure":
            return "blocked"
    return experiment.status


def blurb(conclusion: Conclusion | None) -> str | None:
    if conclusion is None:
        return None
    first = re.split(r"(?<=[.!?])\s", conclusion.narrative_md.strip(), maxsplit=1)[0]
    return first or None


def summary_view(experiment: Experiment, repo: Repo, admin: bool = False) -> ExperimentSummaryView:
    trials = repo.list_trials(experiment.id)
    first = next((t for t in trials if t.role == "fit"), None)
    winner = (
        repo.get_model(experiment.id, experiment.winner_model_id)
        if experiment.winner_model_id
        else None
    )
    return ExperimentSummaryView(
        id=experiment.id,
        title=experiment.title,
        system_type=_system(experiment.system_type),
        status=experiment.status,
        created_at=experiment.created_at.isoformat(),
        is_example=experiment.is_example,
        confidence=experiment.confidence,
        review_status=_review_status(experiment),
        headline=headline(experiment.conclusion),
        winner_name=winner.name if winner else None,
        thumbnail_url=media(experiment.id, first.paths.get("poster")) if first else None,
        blurb=blurb(experiment.conclusion),
    )


def _review_status(experiment: Experiment) -> ReviewStatusName | None:
    s = experiment.review_status
    return s if s in ReviewStatusName.__args__ else None  # type: ignore[attr-defined,return-value]


def experiment_view(experiment: Experiment, repo: Repo, read_track, admin: bool) -> ExperimentView:
    eid = experiment.id
    trials = repo.list_trials(eid)
    ids = [t.id for t in trials]
    tracks, trajectories = repo.list_tracks(ids), repo.list_trajectories(ids)
    models, fits = repo.list_models(eid), repo.list_fits(eid)
    jobs = repo.list_jobs(eid, limit=20)
    trial_views = [trial_view(experiment, t, tracks, trajectories) for t in trials]
    fit_trial = next((t for t in trial_views if t.role == "fit"), None)
    fit_trajectory = next(
        (tr for tr in trajectories if fit_trial and tr.trial_id == fit_trial.id), None
    )
    unit = "rad" if (fit_trajectory and fit_trajectory.summary.get("angle_column")) else "m"

    chosen = comparison_fits(fits)
    won = splits_won_by(chosen)
    model_views = [model_view(m, chosen.get(m.id), eid, unit, won.get(m.id, 0)) for m in models]
    winner = next((m for m in models if m.id == experiment.winner_model_id), None)
    final = next(
        (
            f
            for f in reversed(fits)
            if f.model_id == experiment.winner_model_id and (f.split or {}).get("kind") == "final"
        ),
        None,
    )
    conclusion = (
        conclusion_view(experiment.conclusion, final, repo.find_event(eid, "Confidence"))
        if experiment.conclusion
        else None
    )
    flags = list(fit_trial.quality_flags) if fit_trial else []
    for issue in experiment.scene.quality_issues if experiment.scene else []:
        flags.append(
            _flag(
                f"scene_{len(flags)}",
                "error" if issue.get("severity") == "major" else "warn",
                str(issue.get("note") or issue.get("issue", "")),
            ).model_copy(update={"title": str(issue.get("issue", "Quality issue"))})
        )
    posters = {t.id: t.paths["poster"] for t in trials if t.paths.get("poster")}
    names = {m.id: m.name for m in models}
    scene = experiment.scene
    exportable = bool(
        winner and winner.library_ref in EXPORTABLE_MODELS and experiment.status == "complete"
    )
    whatif = None
    if experiment.whatif:
        try:
            whatif = WhatIfView.model_validate(experiment.whatif)
        except Exception:
            log.warning("stored what-if context does not validate", exc_info=True)
    base = summary_view(experiment, repo)
    return ExperimentView(
        **{**base.model_dump(), "status": _status(experiment, jobs)},
        description=experiment.description,
        declared_capture=DeclaredCaptureView(
            mode=experiment.declared_capture.mode, fps=experiment.declared_capture.fps
        ),
        setup=SetupView(**experiment.setup.model_dump()),
        scene=SceneView.model_validate(scene.model_dump()) if scene else None,
        trials=trial_views,
        models=model_views,
        conclusion=conclusion,
        review=review_view(experiment, repo.list_reviews(eid)),
        ghosts=[ghost_view(g, eid, names, posters, read_track) for g in repo.list_ghosts(eid)],
        whatif=whatif,
        export_formats=["json", "pybullet"] if exportable else [],
        quality_flags=flags,
        job=next((j for j in jobs if j.status in ("queued", "running")), jobs[0] if jobs else None),
        error=experiment.error,
        last_seq=repo.last_event_seq(eid),
        cost_usd=experiment.cost_usd if admin else None,
    )


ExperimentView.model_rebuild()


__all__ = [
    "ApiError",
    "Field",
    "experiment_view",
    "summary_view",
    "event_view",
    "trajectory_view",
]
