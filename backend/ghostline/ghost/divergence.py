"""How far reality drifts from the ghost (build plan Part 16).

e_t   = |actual - predicted| in millimetres, at every measured frame
tau   = max(3 sigma_noise, half the object's radius)
onset = the first time e > tau for five or more frames in a row, after the launch window
green below tau, amber up to 2 tau, red beyond that
"""

from __future__ import annotations

import numpy as np

from .model import AMBER_FACTOR, MIN_RUN_FRAMES, Divergence, Forecast, GhostError, Status
from .series import TrialSeries


def threshold_mm(sigma_noise_mm: float, radius_mm: float) -> float:
    return max(3.0 * sigma_noise_mm, 0.5 * radius_mm)


def status_of(error_mm: float, tau_mm: float) -> Status:
    if error_mm < tau_mm:
        return "green"
    return "amber" if error_mm < AMBER_FACTOR * tau_mm else "red"


def find_onset(
    t: np.ndarray, error_mm: np.ndarray, tau_mm: float, after_t: float, run: int = MIN_RUN_FRAMES
) -> float | None:
    """Time of the first sample of the first run of `run` samples over tau, after `after_t`."""
    count = 0
    for i in range(len(t)):
        if t[i] > after_t and error_mm[i] > tau_mm:
            count += 1
            if count >= run:
                return float(t[i - run + 1])
        else:
            count = 0
    return None


def measure_divergence(forecast: Forecast, trial2: TrialSeries) -> Divergence:
    """Compare a forecast with the positions trial 2's trajectory measured."""
    rows = trial2.motion
    rows = rows[rows["frame"].isin(forecast.frames)]
    if len(rows) < MIN_RUN_FRAMES:
        raise GhostError("Too few measured frames in the second clip to compare with the ghost.")
    index = np.searchsorted(forecast.frames, rows["frame"].to_numpy(int))
    actual = rows[["x", "y"]].to_numpy(float)
    error = 1000.0 * np.hypot(*(actual - forecast.xy[index]).T)
    sigma_mm = 1000.0 * trial2.sigma("x")
    radius_mm = trial2.object_radius_mm()
    tau = threshold_mm(sigma_mm, radius_mm)
    t = rows["t"].to_numpy(float)
    after = t >= forecast.launch_end_t
    onset = find_onset(t, error, tau, forecast.launch_end_t)
    return Divergence(
        t=t,
        frame=rows["frame"].to_numpy(int),
        error_mm=error,
        tau_mm=tau,
        sigma_noise_mm=sigma_mm,
        radius_mm=radius_mm,
        onset_t=onset,
        max_error_mm=float(error[after].max()) if after.any() else float(error.max()),
        status=[status_of(e, tau) for e in error],
    )


def error_at_frames(div: Divergence, frames: np.ndarray) -> np.ndarray:
    """The error at every frame of the video: measured where there is a measurement,
    interpolated across gaps (for drawing only)."""
    return np.interp(np.asarray(frames, float), div.frame.astype(float), div.error_mm)
