"""Types shared by the ghost package (build plan Parts 16 and 17).

`GhostTrack` is the JSON written to `ghost/v{n}_track.json`. Its first seven fields are the
contract with `web/src/lib/types.ts` (GhostTrack); the rest are extras the web ignores.

Time in a track is real seconds since the start of the ghost video, which starts at the first
frame of trial 2's motion window. The web draws the launch-window band from 0 to
`launch_window_s` and scales a slow-motion file's clock against the last `t`, so both depend on
that origin. `t0_s` is where it sits on the master's clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

LAUNCH_FRACTION = 0.10  # of the motion window
LAUNCH_MIN_S = 0.25  # whichever is longer
LAUNCH_MIN_POINTS = 12  # the harness needs 10 usable points; a slow frame rate stretches the window
MIN_RUN_FRAMES = 5  # consecutive frames over tau that make an onset
N_DRAWS = 200  # Monte Carlo draws for the spread (Part 16)
AMBER_FACTOR = 2.0  # amber from tau up to 2 tau, red beyond

Status = Literal["green", "amber", "red"]


class GhostTrack(BaseModel):
    """ghost_track.json: one entry per frame of the ghost video."""

    t: list[float]
    error_mm: list[float]
    predicted_px: list[tuple[float, float]]
    actual_px: list[tuple[float, float]]
    tau_mm: float
    onset_s: float | None
    launch_window_s: float
    # --- extras ---
    version: int = 1
    model_id: str = ""
    frames: list[int] = Field(default_factory=list)  # master frame index of each entry
    t0_s: float = 0.0  # master time of the first entry
    measured: list[bool] = Field(default_factory=list)  # False where `actual` is interpolated
    status: list[Status] = Field(default_factory=list)  # the outline colour of each frame
    spread_mm: list[float] = Field(default_factory=list)  # Monte Carlo spread of the prediction
    max_error_mm: float = 0.0
    predicted_ahead_s: float = 0.0  # from the end of the launch window to the last frame
    sigma_noise_mm: float = 0.0
    object_radius_mm: float = 0.0
    slow_motion: bool = False
    real_fps: float = 0.0


@dataclass
class ICFit:
    """Initial conditions fitted on the launch window, with their standard errors."""

    values: dict[str, float]
    se: dict[str, float]
    at_bound: list[str]
    rmse: float | None = None
    n_points: int = 0


@dataclass
class FrozenFit:
    """Trial 1's final fit: everything except the initial conditions, held fixed for trial 2."""

    model_id: str
    values: dict[str, float]  # physical and nuisance parameters
    sigma: dict[str, float]  # standard errors of the physical ones (statistical)
    constants: dict[str, float]
    initial: dict[str, float] = field(default_factory=dict)  # trial 1's own ICs, for reference
    derived: dict[str, float] = field(default_factory=dict)
    train_rmse: float | None = None


@dataclass
class Forecast:
    """One model's prediction for trial 2, at every frame of its motion window."""

    model_id: str
    frames: np.ndarray  # master frame indices
    t: np.ndarray  # real seconds on the master's clock
    obs: np.ndarray  # (n_obs, n) predicted observables
    xy: np.ndarray  # (n, 2) predicted world position of the tracked point, metres
    angle: np.ndarray | None  # (n,) predicted rotation angle of the object, spinners only
    spread_mm: np.ndarray  # (n,) one-sigma spread across the Monte Carlo draws
    ic: ICFit
    launch_end_t: float  # master time where the launch window ends
    launch_window_s: float
    params: dict[str, float]
    n_draws: int


@dataclass
class Divergence:
    """How far reality drifted from a forecast."""

    t: np.ndarray  # per measured row, master clock
    frame: np.ndarray
    error_mm: np.ndarray
    tau_mm: float
    sigma_noise_mm: float
    radius_mm: float
    onset_t: float | None  # master clock
    max_error_mm: float
    status: list[Status]


class GhostError(RuntimeError):
    """The ghost cannot be built from this data. The message is safe to show the user."""
