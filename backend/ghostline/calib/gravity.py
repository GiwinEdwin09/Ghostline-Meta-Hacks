"""Which way is down, in the sheet frame (millimetres, y down), with its source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import find_peaks

from .homography import apply_h

CAMERA_LEVEL_SIGMA_DEG = 1.0  # how well a hand-levelled phone matches true vertical


@dataclass
class Gravity:
    down: np.ndarray  # unit vector in the sheet frame
    source: Literal["data", "plumb", "camera_level"]
    sigma_deg: float

    @property
    def angle_deg(self) -> float:
        """Angle of 'down' from the sheet's +y axis, positive toward +x."""
        return float(np.degrees(np.arctan2(self.down[0], self.down[1])))


def physics_axes(down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(right, up) unit vectors in the sheet frame for a given 'down'."""
    up = -np.asarray(down, float)
    return np.array([-up[1], up[0]]), up


def from_camera_level(h: np.ndarray, image_size: tuple[int, int]) -> Gravity:
    """Image vertical at the image centre mapped through the sheet homography."""
    centre = np.array([(image_size[0] - 1) / 2, (image_size[1] - 1) / 2])
    a, b = apply_h(h, np.array([centre, centre + [0, 50.0]]))
    d = b - a
    return Gravity(d / np.linalg.norm(d), "camera_level", CAMERA_LEVEL_SIGMA_DEG)


def from_plumb(pivot_mm: np.ndarray, bob_at_rest_mm: np.ndarray) -> Gravity:
    d = np.asarray(bob_at_rest_mm, float) - np.asarray(pivot_mm, float)
    return Gravity(d / np.linalg.norm(d), "plumb", 0.5)


def from_free_fall(t: np.ndarray, xy_mm: np.ndarray) -> Gravity | None:
    """Direction of the mean acceleration of a free flight, from a quadratic fit."""
    t = np.asarray(t, float)
    xy = np.asarray(xy_mm, float)
    if len(t) < 8:
        return None
    tc = t - t.mean()
    design = np.vander(tc, 3)
    coef, *_ = np.linalg.lstsq(design, xy, rcond=None)
    accel = 2 * coef[0]
    norm = np.linalg.norm(accel)
    if norm < 1e-9:
        return None
    resid = xy - design @ coef
    dof = max(len(t) - 3, 1)
    sigma = np.sqrt((resid**2).sum(axis=0) / dof)
    var_c2 = np.linalg.inv(design.T @ design)[0, 0]
    perp = np.array([-accel[1], accel[0]]) / norm
    sigma_a = 2 * np.sqrt(var_c2) * np.sqrt(np.sum((sigma * perp) ** 2))
    return Gravity(accel / norm, "data", float(np.degrees(sigma_a / norm)))


def plumb_from_oscillation(
    t: np.ndarray, xy_mm: np.ndarray, pivot_mm: np.ndarray
) -> Gravity | None:
    """Equilibrium direction of a swinging pendulum from the midpoints of its turning points.

    Averaging (e_k + 2 e_k+1 + e_k+2) / 4 over consecutive extremes cancels a slow decay, so the
    result is the rest direction even if the swing started from one side.
    """
    d = np.asarray(xy_mm, float) - np.asarray(pivot_mm, float)
    alpha = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    span = float(alpha.max() - alpha.min())
    if len(alpha) < 10 or span < np.radians(2):
        return None
    prominence = 0.2 * span
    hi, _ = find_peaks(alpha, prominence=prominence)
    lo, _ = find_peaks(-alpha, prominence=prominence)
    ext = np.sort(np.concatenate([hi, lo]))
    if len(ext) < 4:
        return None
    values = alpha[ext]
    mids = (values[:-2] + 2 * values[1:-1] + values[2:]) / 4
    centre = float(np.mean(mids))
    sigma = float(np.degrees(np.std(mids) / np.sqrt(len(mids)))) if len(mids) > 1 else 1.0
    return Gravity(np.array([np.cos(centre), np.sin(centre)]), "plumb", max(sigma, 0.3))
