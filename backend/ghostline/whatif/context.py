"""The What-if tab's data, built from an analysis (build plan Part 18).

`build_whatif` returns the dict stored in `experiments.whatif` and read back as `WhatIfView`
(api/views.py): the winner's final parameters under their library names, the constants the fit
held fixed, trial 1's initial conditions, the ranges the clip measured, and where the motion is
anchored in the world. The sliders and the TypeScript twins in `web/src/lib/sim` run from it.

The initial conditions are not saved with a fit, so they are read off the start of the measured
motion with a short quadratic fit. Pass the full-rate trajectory frame for that; without it the
60 samples in the trajectory summary are used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import numpy as np
import pandas as pd

from ghostline.export.measured import amplitude_range_deg, describe_objects
from ghostline.physics.library import get_model
from ghostline.physics.library.ramp import STANDARD_GRAVITY
from ghostline.store.records import Experiment, FitRecord, ModelRecord, TrajectoryRecord, Trial
from ghostline.traj.model import FLAG_AT_REST

SYSTEM_OF_PREFIX = {
    "FF": "free_fall",
    "P": "pendulum",
    "R": "ramp",
    "B": "bounce",
    "S": "spin_down",
}
# Library models that have a TypeScript twin in web/src/lib/sim/models.
TWIN_MODELS = {
    "FF0", "FF1", "FF2", "P0", "P1", "P2", "P3", "P4", "R0", "R1", "R2", "R3",
    "B0", "B1", "B2", "S0", "S1", "S2", "S3",
}  # fmt: skip
OBSERVABLE = {
    "free_fall": "y", "pendulum": "theta", "ramp": "s", "bounce": "y", "spin_down": "phi",
}  # fmt: skip
DEFAULT_RADIUS_M = 0.02
SLIDER_MARGIN = 0.25  # relative margin around a single measured drop height or spin
RAMP_ANGLE_MARGIN_DEG = 5.0  # friction is angle-independent, so a few degrees is not a stretch
START_POINTS = 6  # points of the motion's start used for the initial conditions


class WhatIfRepo(Protocol):
    def list_trials(self, experiment_id: UUID) -> list[Trial]: ...
    def list_trajectories(self, trial_ids: list[UUID]) -> list[TrajectoryRecord]: ...
    def list_models(self, experiment_id: UUID) -> list[ModelRecord]: ...
    def list_fits(self, experiment_id: UUID) -> list[FitRecord]: ...


def system_of_model(model_id: str, fallback: str) -> str:
    """The What-if system of a library model id, else the experiment's own system type."""
    prefix = model_id.rstrip("0123456789")
    return SYSTEM_OF_PREFIX.get(prefix, fallback)


def library_ref_of(model: ModelRecord) -> str | None:
    """The library model a row stands for: its library_ref, else its id for a library row."""
    if model.library_ref:
        return model.library_ref
    return model.id if model.source == "library" and model.code is None else None


def latest_fit(fits: list[FitRecord], model_id: str) -> FitRecord | None:
    """The model's last 'final' fit, else its last fit of any kind."""
    mine = [f for f in fits if f.model_id == model_id]
    final = [f for f in mine if (f.split or {}).get("kind") == "final"]
    return (final or mine or [None])[-1]


def physical_names(model: ModelRecord) -> list[str]:
    """Parameter names of the model that are physical, library or Wonder-written."""
    ref = library_ref_of(model)
    if ref:
        return [n for n, s in get_model(ref).params.items() if s.kind == "physical"]
    return [n for n, s in (model.params_spec or {}).items() if s.kind == "physical"]


def fitted_params(model: ModelRecord, fit: FitRecord | None) -> dict[str, float]:
    if fit is None:
        return {}
    keep = set(physical_names(model))
    return {p.name: float(p.value) for p in fit.params if p.name in keep}


# --- the measured motion ---------------------------------------------------------------------
def motion_series(summary: dict[str, Any], frame: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """The motion window as arrays (t and the trajectory columns), from the full-rate frame if
    given, else from the 60 samples in the summary."""
    if frame is not None and len(frame):
        if "flags" in frame.columns:
            moving = (frame["flags"].to_numpy(dtype=int) & FLAG_AT_REST) == 0
            frame = frame[moving] if moving.sum() >= 10 else frame
        return {c: frame[c].to_numpy(dtype=float) for c in frame.columns if c != "flags"}
    return {k: np.asarray(v, dtype=float) for k, v in (summary.get("samples") or {}).items()}


def _start(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Value and slope at the first sample, from a quadratic through the first few points."""
    n = min(START_POINTS, t.size)
    if n < 3:
        return float(y[0]), 0.0
    c2, c1, c0 = np.polyfit(t[:n] - t[0], y[:n], 2)
    return float(c0), float(c1)


def _along_path(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance travelled along the line from the first to the last point, and its direction."""
    origin = np.array([x[0], y[0]])
    delta = np.array([x[-1], y[-1]]) - origin
    axis = delta / (np.hypot(*delta) or 1.0)
    return (np.stack([x, y], axis=1) - origin) @ axis, axis


def _around(value: float, margin: float = SLIDER_MARGIN) -> list[float]:
    return [value * (1 - margin), value * (1 + margin)]


def measured_setup(
    system: str, summary: dict[str, Any], calibration: dict[str, Any], series: dict[str, np.ndarray]
) -> dict[str, Any] | None:
    """Constants, initial conditions, measured ranges and the anchor for one system, or None if
    the clip lacks what the system needs."""
    t = series.get("t")
    if t is None or t.size < 3:
        return None
    fixed: dict[str, float] = {}
    initial: dict[str, float] = {}
    ranges: dict[str, list[float]] = {}
    if system == "pendulum":
        info = summary.get("pendulum") or {}
        length = info.get("length_eff_m") or info.get("length_m")
        if not length or "theta" not in series:
            return None
        fixed["L"] = float(length)
        initial["theta0"], initial["omega0"] = _start(t, series["theta"])
        amplitude = amplitude_range_deg(summary) or [0.0, abs(math.degrees(initial["theta0"]))]
        ranges["amplitude_deg"] = amplitude
        anchor = info.get("pivot_xy_m") or [float(series["x"][0]), float(series["y"][0]) + length]
    elif system == "ramp":
        ramp = calibration.get("ramp") or {}
        incline = ramp.get("incline_deg", (summary.get("ramp") or {}).get("board_angle_deg"))
        if incline is None or "x" not in series:
            return None
        s, axis = _along_path(series["x"], series["y"])
        fixed.update(
            alpha=math.radians(abs(float(incline))),
            g=STANDARD_GRAVITY,
            s_max=max(float(s.max()) * 1.2, 0.3),
            dir_x=1.0 if axis[0] >= 0 else -1.0,
        )
        initial["s0"] = 0.0
        initial["v0"] = max(_start(t, s)[1], 0.0)
        ranges["angle_deg"] = [
            abs(incline) - RAMP_ANGLE_MARGIN_DEG,
            abs(incline) + RAMP_ANGLE_MARGIN_DEG,
        ]
        anchor = [float(series["x"][0]), float(series["y"][0])]
    elif system in ("free_fall", "bounce"):
        if "y" not in series:
            return None
        y = series["y"]
        floor = float(y.min())
        height, speed = _start(t, y)
        initial["y0"] = max(height - floor, 0.0)
        initial["vy0" if system == "free_fall" else "v0"] = speed
        ranges["height_m"] = _around(max(initial["y0"], 0.05))
        anchor = [float(np.mean(series["x"])) if "x" in series else 0.0, floor]
    elif system == "spin_down":
        info = summary.get("spinner") or {}
        if "phi" not in series or not info.get("hub_xy_m"):
            return None
        _, rate = _start(t, series["phi"])
        direction = -1.0 if (rate < 0 or info.get("direction") == "cw") else 1.0
        fixed.update(
            direction=direction,
            g=STANDARD_GRAVITY,
            marker_radius_m=float(info.get("marker_radius_m") or 0.03),
        )
        initial["phi0"] = 0.0
        initial["omega0"] = abs(rate)
        end = abs(_start(t[::-1] * -1, series["phi"][::-1])[1])
        low, high = sorted([abs(rate), end])
        ranges["spin_rev_s"] = [low / (2 * math.pi), high * (1 + 0.1) / (2 * math.pi)]
        anchor = info["hub_xy_m"]
    else:
        return None
    return {
        "fixed": fixed,
        "initial": initial,
        "measured_range": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
        "anchor_m": [float(anchor[0]), float(anchor[1])],
    }


def trajectory_frame(root: Path, repo: WhatIfRepo, experiment: Experiment) -> pd.DataFrame | None:
    """The fit trial's full-rate trajectory from the experiment folder, if it can be read."""
    trials = [t for t in repo.list_trials(experiment.id) if t.role == "fit"]
    rows = repo.list_trajectories([trials[-1].id]) if trials else []
    if not rows or not rows[0].parquet_path:
        return None
    try:
        return pd.read_parquet(root / rows[0].parquet_path)
    except (OSError, ValueError):
        return None


def build_whatif(
    repo: WhatIfRepo, experiment: Experiment, frame: pd.DataFrame | None = None
) -> dict[str, Any] | None:
    """The `WhatIfView` dict for a finished analysis, or None if the winner can't be played
    (no result, an oscillator or projectile, or a clip without the geometry)."""
    winner_id = experiment.winner_model_id
    if experiment.conclusion is None or winner_id is None:
        return None
    models = {m.id: m for m in repo.list_models(experiment.id)}
    model = models.get(winner_id)
    if model is None:
        return None
    library_ref = library_ref_of(model)
    ref = library_ref or winner_id
    fallback = experiment.scene.system_type if experiment.scene else experiment.system_type
    system = system_of_model(ref, fallback) if library_ref else fallback
    if system not in OBSERVABLE:
        return None
    trials = [t for t in repo.list_trials(experiment.id) if t.role == "fit"]
    trial = trials[-1] if trials else None
    rows = repo.list_trajectories([trial.id]) if trial else []
    if trial is None or not rows:
        return None
    summary = rows[0].summary or {}
    setup = measured_setup(system, summary, trial.calibration or {}, motion_series(summary, frame))
    if setup is None:
        return None
    name, _surface, size_cm = describe_objects(experiment)
    radius = (summary.get("pendulum") or {}).get("bob_radius_m") or (
        size_cm / 200 if size_cm else DEFAULT_RADIUS_M
    )
    poster = trial.paths.get("poster")
    return {
        "system": system,
        "model_ref": ref,
        "params": fitted_params(model, latest_fit(repo.list_fits(experiment.id), winner_id)),
        "fixed": setup["fixed"],
        "initial": setup["initial"],
        "measured_range": setup["measured_range"],
        "object": {"name": name, "radius_m": float(radius), "mass_kg": None},
        "still_url": f"{experiment.id}/{poster}" if poster else None,
        "sim_supported": bool(library_ref and ref in TWIN_MODELS),
        "anchor_m": setup["anchor_m"],
    }
