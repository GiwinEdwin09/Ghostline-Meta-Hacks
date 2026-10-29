"""From a forecast and its divergence to the render plan and ghost_track.json (build plan Part 16).

World coordinates go back to pixels the way the plan says: remove the plane-offset correction,
then trial 2's H^-1 (`Calibration.world_to_px`).
"""

from __future__ import annotations

import math

import numpy as np

from .divergence import error_at_frames, status_of
from .model import Divergence, Forecast, GhostTrack
from .render import RenderPlan
from .series import TrialSeries
from .sprite import Sprite

SLOW_MOTION_FPS = 100.0  # at or above this frame rate the clip also gets a slow-playback copy


def is_slow_motion(trial2: TrialSeries, declared_slow_mo: bool = False) -> bool:
    return declared_slow_mo or trial2.real_fps() >= SLOW_MOTION_FPS


def build_plan(
    forecast: Forecast,
    div: Divergence,
    trial2: TrialSeries,
    sprite: Sprite,
    *,
    version: int,
    label: str,
    slow_motion: bool = False,
    source_angle: float | None = None,
) -> tuple[RenderPlan, GhostTrack]:
    """The per-frame drawing instructions and the JSON the web reads, from the same numbers."""
    frames = forecast.frames
    t0 = float(forecast.t[0])
    t_rel = forecast.t - t0
    predicted = trial2.calib.world_to_px(forecast.xy, trial2.down)
    actual, measured = trial2.actual_px(frames)
    error = error_at_frames(div, frames)
    status = [status_of(float(e), div.tau_mm) for e in error]
    launch_end_rel = float(forecast.launch_end_t - t0)
    anchor_px = None
    if forecast.anchor_xy is not None:
        anchor_px = trial2.calib.world_to_px(forecast.anchor_xy[None], trial2.down)[0]
    angle_deg = None
    if forecast.angle is not None:
        ref = source_angle if source_angle is not None else float(forecast.angle[0])
        angle_deg = np.degrees(forecast.angle - ref)
    plan = RenderPlan(
        frames=frames,
        t_rel=t_rel,
        predicted_px=predicted,
        actual_px=actual,
        error_mm=error,
        status=status,
        sprite=sprite,
        launch_end_rel=launch_end_rel,
        label=label,
        anchor_px=anchor_px,
        angle_deg=angle_deg,
        slow_motion=slow_motion,
    )
    onset = None if div.onset_t is None else float(div.onset_t - t0)
    track = GhostTrack(
        t=_round(t_rel, 5),
        error_mm=_round(error, 3),
        predicted_px=_pairs(predicted),
        actual_px=_pairs(actual),
        tau_mm=round(float(div.tau_mm), 4),
        onset_s=None if onset is None else round(onset, 5),
        launch_window_s=round(launch_end_rel, 5),
        version=version,
        model_id=forecast.model_id,
        frames=[int(f) for f in frames],
        t0_s=round(t0, 6),
        measured=[bool(m) for m in measured],
        status=status,
        spread_mm=_round(forecast.spread_mm, 3),
        max_error_mm=round(float(div.max_error_mm), 4),
        predicted_ahead_s=round(float(t_rel[-1] - launch_end_rel), 4),
        sigma_noise_mm=round(float(div.sigma_noise_mm), 4),
        object_radius_mm=round(float(div.radius_mm), 3),
        slow_motion=slow_motion,
        real_fps=round(float(trial2.real_fps()), 3),
    )
    return plan, track


def _round(a: np.ndarray, digits: int) -> list[float]:
    return [round(float(v), digits) if math.isfinite(float(v)) else 0.0 for v in a]


def _pairs(a: np.ndarray) -> list[tuple[float, float]]:
    return [(round(float(x), 2), round(float(y), 2)) for x, y in a]
