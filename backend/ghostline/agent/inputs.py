"""Harness inputs for one analysis, built from the measurement stages' outputs (Parts 9 and 12-13).

`build_inputs` reads trajectory.parquet and the trajectory summary, the calibration, the scene
analysis and the clip's timing, and returns everything the fitting harness and Wonder need:

- the motion window as harness `SeriesData` (the part of the clip between the release and the
  stop; rows flagged `FLAG_AT_REST` are left out),
- the job constants (pendulum length, ramp angle) and the calibration/timing uncertainty budget,
- the data-quality flags the confidence rules and the reviewer read,
- the **training window**: the first 65% of the motion (the first k flights for a bounce). It is
  the only part of the data Wonder is ever shown; `TrainingView` is the one place that cuts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ghostline.confidence import MAJOR_FLAGS, QualityFlag
from ghostline.physics.harness.fitting import Dataset, split_masks
from ghostline.physics.harness.schema import SeriesData
from ghostline.physics.library import get_model, model_ids
from ghostline.physics.select import CalibrationBudget
from ghostline.store.core_schemas import SceneAnalysis
from ghostline.store.records import Experiment, Trial
from ghostline.store.volume import VolumePaths
from ghostline.traj.model import FLAG_AT_REST

PRIMARY_SPLIT = 0.65
SAMPLE_POINTS = 60  # points of the training window in the first prompt (Part 13)
MAX_TOOL_POINTS = 400

# The trajectory columns each system is fitted on (traj/model.py), and their units.
SYSTEM_COLUMNS: dict[str, tuple[str, ...]] = {
    "free_fall": ("y",),
    "projectile": ("x", "y"),
    "pendulum": ("theta",),
    "ramp": ("s",),
    "bounce": ("y",),
    "spin_down": ("phi",),
    "oscillator": ("y",),
}
UNITS = {"x": "m", "y": "m", "s": "m", "theta": "rad", "phi": "rad"}
# Which entry of the trajectory summary's `noise` block is each column's one-sigma.
_NOISE_KEY = {
    "x": "sigma_x_m",
    "y": "sigma_y_m",
    "s": "sigma_pos_m",
    "theta": "sigma_angle_rad",
    "phi": "sigma_angle_rad",
}
# Notices from the trajectory stage that are major flags, with the code the rules know them by.
_MAJOR_NOTICES = {"no_motion": "no_motion", "looks_retimed": "time_base_uncertain"}


class InputsError(ValueError):
    """The trajectory can't be fitted (no usable motion, an unsupported system)."""


@dataclass
class TrainingView:
    """The training window of the motion, as numpy arrays: what Wonder may see."""

    t: np.ndarray
    obs: dict[str, np.ndarray]
    t_split: float

    def __len__(self) -> int:
        return int(self.t.size)

    def sample(self, n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """About n evenly spaced points (the first and last included)."""
        idx = _spread(len(self), n)
        return self.t[idx], {k: v[idx] for k, v in self.obs.items()}

    def window(
        self, t_start: float | None, t_end: float | None, max_points: int
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Points with t_start <= t <= t_end, clamped to the training window and thinned."""
        keep = np.ones(len(self), dtype=bool)
        if t_start is not None:
            keep &= self.t >= t_start
        if t_end is not None:
            keep &= self.t <= t_end
        t, obs = self.t[keep], {k: v[keep] for k, v in self.obs.items()}
        idx = _spread(t.size, max_points)
        return t[idx], {k: v[idx] for k, v in obs.items()}


def _spread(n: int, count: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    if n <= count:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, count).round().astype(int))


@dataclass
class PhysicsInputs:
    system: str
    data: SeriesData
    constants: dict[str, float]
    budget: CalibrationBudget
    flags: list[QualityFlag]
    units: dict[str, str]
    t_split: float
    train_mask: np.ndarray
    primary_split: float = PRIMARY_SPLIT
    context: dict[str, Any] = field(default_factory=dict)  # measurement summary for the prompt
    trial_id: Any = None

    @property
    def observables(self) -> tuple[str, ...]:
        return tuple(self.data.series)

    @property
    def sigma(self) -> float:
        return max(self.data.sigma.values())

    @property
    def duration_s(self) -> float:
        return float(self.data.t[-1] - self.data.t[0]) if self.data.t else 0.0

    @property
    def motion_range(self) -> float | None:
        ranges = [_ptp(v) for v in self.data.series.values()]
        return max(ranges, default=None)

    def arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        t = np.asarray(self.data.t, dtype=float)
        obs = {
            k: np.array([np.nan if x is None else x for x in v], dtype=float)
            for k, v in self.data.series.items()
        }
        return t, obs

    def training(self) -> TrainingView:
        t, obs = self.arrays()
        mask = self.train_mask
        return TrainingView(t[mask], {k: v[mask] for k, v in obs.items()}, self.t_split)


def _ptp(values: list[float | None]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(max(finite) - min(finite)) if finite else 0.0


def effective_system(experiment: Experiment, scene: SceneAnalysis) -> str:
    """The system the library is indexed by: the user's choice, else what Wonder saw."""
    system = experiment.system_type
    return scene.system_type if system in ("auto", "", None) else system


def build_inputs(
    experiment: Experiment,
    trial: Trial,
    scene: SceneAnalysis,
    *,
    parquet_path: str,
    summary: dict[str, Any],
    calibration: dict[str, Any] | None,
    paths: VolumePaths,
) -> PhysicsInputs:
    """Read the trajectory and assemble the inputs. `parquet_path` is relative to the experiment
    folder; `summary` is the trajectory stage's summary dict; `calibration` is calibration.json."""
    system = effective_system(experiment, scene)
    columns = SYSTEM_COLUMNS.get(system)
    if columns is None:
        raise InputsError(f"Ghostline has no measurement columns for a {system!r} system.")
    calibration = calibration or trial.calibration or {}
    frame = pd.read_parquet(paths.experiment_dir(experiment.id) / parquet_path)
    return inputs_from_frame(
        frame,
        system=system,
        summary=summary,
        calibration=calibration,
        timing=trial.timing,
        scene=scene,
        pendulum_length_cm=experiment.setup.pendulum_length_cm,
        trial_id=trial.id,
    )


def inputs_from_frame(
    frame: pd.DataFrame,
    *,
    system: str,
    summary: dict[str, Any],
    calibration: dict[str, Any],
    timing: dict[str, Any] | None,
    scene: SceneAnalysis | None = None,
    pendulum_length_cm: float | None = None,
    trial_id: Any = None,
) -> PhysicsInputs:
    columns = SYSTEM_COLUMNS[system]
    missing = [c for c in ("t", *columns) if c not in frame.columns]
    if missing:
        raise InputsError(f"trajectory.parquet has no column(s) {missing} for a {system} clip.")
    if "flags" in frame.columns:
        moving = (frame["flags"].to_numpy(dtype=int) & FLAG_AT_REST) == 0
        window = frame[moving] if moving.sum() >= 10 else frame
    else:
        window = frame
    window = window.sort_values("t", kind="stable")
    t = window["t"].to_numpy(dtype=float)
    if t.size < 10:
        raise InputsError(f"only {t.size} usable trajectory points; need at least 10.")

    noise = (summary.get("noise") or {}) if summary else {}
    sigma = {c: _sigma_for(c, noise, window[c].to_numpy(dtype=float)) for c in columns}
    series = {
        c: [None if not np.isfinite(x) else float(x) for x in window[c].to_numpy(dtype=float)]
        for c in columns
    }
    weights = window["weight"].to_numpy(dtype=float).tolist() if "weight" in window else None
    segment = window["segment_id"].to_numpy(dtype=int).tolist() if "segment_id" in window else None
    data = SeriesData(t=t.tolist(), series=series, sigma=sigma, weights=weights, segment_id=segment)

    constants = _constants(system, summary, calibration, pendulum_length_cm)
    budget = _budget(summary, calibration, timing)
    flags = quality_flags(summary, calibration, timing, scene)
    t_split, train_mask = _training_split(system, data, PRIMARY_SPLIT)
    inputs = PhysicsInputs(
        system=system,
        data=data,
        constants=constants,
        budget=budget,
        flags=flags,
        units={c: UNITS[c] for c in columns},
        t_split=t_split,
        train_mask=train_mask,
        trial_id=trial_id,
    )
    inputs.context = measurement_context(inputs, summary, calibration, timing)
    return inputs


def _sigma_for(column: str, noise: dict[str, Any], values: np.ndarray) -> float:
    """The stage's noise estimate, else a robust one from second differences of the data."""
    given = noise.get(_NOISE_KEY[column])
    if not given and UNITS[column] == "m":
        given = noise.get("sigma_pos_m")
    if given and math.isfinite(given) and given > 0:
        return float(given)
    finite = values[np.isfinite(values)]
    if finite.size >= 5:
        d2 = np.diff(finite, n=2)
        mad = float(np.median(np.abs(d2 - np.median(d2)))) * 1.4826
        if mad > 0:
            return mad / math.sqrt(6.0)
    return max(1e-6, 1e-3 * (float(np.ptp(finite)) if finite.size else 1.0))


def _constants(
    system: str, summary: dict[str, Any], calibration: dict[str, Any], tape_cm: float | None
) -> dict[str, float]:
    out: dict[str, float] = {}
    if system == "pendulum":
        pend = summary.get("pendulum") or {}
        length = pend.get("length_eff_m") or pend.get("length_m")
        if not length and tape_cm:
            length = tape_cm / 100.0
        if not length:
            raise InputsError(
                "The pendulum's length couldn't be measured from the clip. Track the pivot or "
                "enter the string length."
            )
        out["L"] = float(length)
        if tape_cm:
            out["L_tape"] = float(tape_cm) / 100.0
    elif system == "ramp":
        ramp = calibration.get("ramp") or (summary.get("ramp") or {})
        incline = ramp.get("incline_deg", ramp.get("board_angle_deg", ramp.get("path_angle_deg")))
        if incline is None:
            raise InputsError("The ramp's angle couldn't be measured from the clip.")
        out["alpha"] = math.radians(abs(float(incline)))
    return out


def _budget(
    summary: dict[str, Any], calibration: dict[str, Any], timing: dict[str, Any] | None
) -> CalibrationBudget:
    unc = (summary or {}).get("uncertainty") or {}
    scale = unc.get("scale_rel_at_object") or unc.get("scale_rel")
    if scale is None:
        scale = (calibration.get("uncertainty") or {}).get("sigma_scale_rel")
    timing_rel = unc.get("timing_rel")
    if timing_rel is None and timing:
        timing_rel = timing.get("sigma_time_rel")
    ramp = calibration.get("ramp") or {}
    angle = math.radians(float(ramp["sigma_deg"])) if ramp.get("sigma_deg") is not None else 0.0
    return CalibrationBudget(
        sigma_scale=float(scale or 0.0),
        sigma_offset=float(unc.get("out_of_plane_rel") or 0.0),
        sigma_angle=angle,
        sigma_timing=float(timing_rel or 0.0),
    )


def quality_flags(
    summary: dict[str, Any],
    calibration: dict[str, Any],
    timing: dict[str, Any] | None,
    scene: SceneAnalysis | None,
) -> list[QualityFlag]:
    """Data-quality flags for the confidence rules and the reviewer (Parts 14-15)."""
    flags: dict[str, QualityFlag] = {}

    def add(flag: QualityFlag) -> None:
        old = flags.get(flag.code)
        if old is None or (flag.severity == "major" and old.severity != "major"):
            flags[flag.code] = flag

    oop = (summary or {}).get("out_of_plane") or {}
    if oop.get("flag"):
        share = oop.get("dz_over_z")
        note = MAJOR_FLAGS["out_of_plane"]
        if share:
            note += f" Its apparent size changes by about {share:.1%} of the distance."
        add(QualityFlag.major("out_of_plane", note))
    motion = calibration.get("camera_motion") or {}
    if motion.get("status") == "stop":
        add(
            QualityFlag.major(
                "camera_motion",
                f"The camera moved by up to {motion.get('max_shift_px', 0):.1f} px during the "
                "clip.",
            )
        )
    elif motion.get("status") == "warn":
        add(
            QualityFlag(
                code="camera_motion_minor",
                note=f"The camera shifted by up to {motion.get('max_shift_px', 0):.1f} px.",
            )
        )
    retime = (summary or {}).get("retime_check") or {}
    if retime.get("suspected_factor"):
        add(
            QualityFlag.major(
                "time_base_uncertain",
                f"The fitted gravity is about 9.81/{retime['suspected_factor'] ** 2:g}, as if the "
                f"clip were slowed {retime['suspected_factor']:g}x.",
            )
        )
    if timing and (
        timing.get("retimed")
        or timing.get("needs_confirmation")
        or timing.get("classification") == "mixed"
    ):
        why = "The clip looks re-timed (slow motion or sped up)"
        if timing.get("needs_confirmation"):
            why = "The capture frame rate could not be confirmed"
        note = MAJOR_FLAGS["time_base_uncertain"] + f" {why}."
        add(QualityFlag.major("time_base_uncertain", note))
    for notice in (summary or {}).get("notices") or []:
        code = notice.get("code", "")
        if code in _MAJOR_NOTICES:
            add(QualityFlag.major(_MAJOR_NOTICES[code], notice.get("message", "")))
        elif code and code != "out_of_plane":
            add(QualityFlag(code=code, note=notice.get("message", "")))
    if scene is not None:
        for issue in scene.quality_issues:
            if issue.get("severity") == "major":
                text = f"{issue.get('issue', 'A problem')}: {issue.get('note', '')}".strip()
                add(QualityFlag.major("scene_issue", text))
    return list(flags.values())


def _training_split(system: str, data: SeriesData, frac: float) -> tuple[float, np.ndarray]:
    """The same chronological split the harness uses (it depends on the model only for bounces,
    which train on whole flights), taken with the first library model of the system."""
    ds = Dataset.from_series(data)
    ids = model_ids(system)
    if ids:
        model = get_model(ids[0], system)
        t_split = split_masks(model, ds.for_model(model), frac)[2]
    else:
        t_split = float(ds.t[0] + frac * (ds.t[-1] - ds.t[0]))
    return float(t_split), np.asarray(data.t, dtype=float) <= t_split


def measurement_context(
    inputs: PhysicsInputs,
    summary: dict[str, Any],
    calibration: dict[str, Any],
    timing: dict[str, Any] | None,
) -> dict[str, Any]:
    """The calibration and measurement facts Wonder is shown. Nothing here reaches into the
    held-out part of the motion: ranges and events are cut at the end of the training window."""
    out: dict[str, Any] = {"system": inputs.system}
    scale = calibration.get("mm_per_px_center", calibration.get("mm_per_px"))
    unc = (summary or {}).get("uncertainty") or {}
    out["scale"] = {
        "mm_per_px": scale,
        "relative_uncertainty": unc.get("scale_rel_at_object") or unc.get("scale_rel"),
    }
    offset = calibration.get("offset") or {}
    if offset.get("d_m") is not None:
        out["plane_offset_cm"] = round(float(offset["d_m"]) * 100, 2)
    motion = calibration.get("camera_motion") or {}
    if motion:
        out["camera_motion_px"] = motion.get("max_shift_px")
    if timing:
        out["timing"] = {
            k: timing.get(k)
            for k in ("classification", "real_fps", "sigma_time_rel", "retimed")
            if timing.get(k) is not None
        }
    gravity = (summary or {}).get("gravity") or {}
    if gravity:
        out["gravity_direction"] = {
            "source": gravity.get("source"),
            "sigma_deg": gravity.get("sigma_deg"),
        }
    noise = (summary or {}).get("noise") or {}
    out["noise"] = {k: v for k, v in inputs.data.sigma.items()}
    out["noise_note"] = noise.get("method")
    pend = (summary or {}).get("pendulum") or {}
    if pend:
        out["pendulum"] = {
            k: pend.get(k) for k in ("length_m", "length_eff_m", "bob_radius_m") if k in pend
        }
    ramp = calibration.get("ramp") or {}
    if ramp:
        out["ramp"] = {k: ramp.get(k) for k in ("incline_deg", "sigma_deg") if k in ramp}
    spinner = (summary or {}).get("spinner") or {}
    if spinner:
        out["spinner"] = {
            k: spinner.get(k)
            for k in ("direction", "marker_radius_m", "max_step_deg_per_frame")
            if k in spinner
        }
    events = (summary or {}).get("events") or {}
    t0 = inputs.data.t[0]
    out["events"] = {
        "impacts_t": [round(x, 4) for x in events.get("impacts_t") or [] if x <= inputs.t_split],
        "release_t": events.get("release_t"),
    }
    view = inputs.training()
    out["training_window"] = {
        "t_start": round(float(view.t[0]), 4),
        "t_end": round(float(view.t[-1]), 4),
        "n_points": len(view),
        "fraction": inputs.primary_split,
        "ranges": {
            k: [round(float(np.nanmin(v)), 5), round(float(np.nanmax(v)), 5)]
            for k, v in view.obs.items()
        },
    }
    out["total_n_points"] = len(inputs.data.t)
    out["motion_start_t"] = round(float(t0), 4)
    out["quality_flags"] = [
        {"code": f.code, "severity": f.severity, "note": f.note} for f in inputs.flags
    ]
    return out
