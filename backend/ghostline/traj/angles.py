"""Angles: pendulum swing angle and spinner rotation angle, plus the circle fit behind both."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

ALIAS_STEP_DEG = 120.0  # a step above this per frame risks aliasing
MIN_ARC_DEG = 20.0  # below this swing the circle fit can't pin down the pivot


@dataclass
class CircleFit:
    centre: np.ndarray
    radius: float
    rms: float  # of the radial residuals
    sigma_radius: float
    sigma_centre: float
    arc_deg: float  # angular span of the points seen from the centre


def kasa_fit(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic least-squares circle (Kasa): fast, a good start for the geometric fit."""
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    sol, *_ = np.linalg.lstsq(a, x**2 + y**2, rcond=None)
    centre = sol[:2]
    return centre, float(np.sqrt(max(sol[2] + centre @ centre, 1e-12)))


def fit_circle(xy: np.ndarray) -> CircleFit:
    """Kasa fit refined by geometric least squares; uncertainties from the Jacobian."""
    xy = np.asarray(xy, float)
    c0, r0 = kasa_fit(xy)

    def resid(p):
        return np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1]) - p[2]

    sol = least_squares(resid, [c0[0], c0[1], r0], method="lm")
    centre, radius = sol.x[:2], float(sol.x[2])
    r = sol.fun
    dof = max(len(xy) - 3, 1)
    rms = float(np.sqrt(np.mean(r**2)))
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * (r @ r) / dof
        sig_c = float(np.sqrt(np.trace(cov[:2, :2]) / 2))
        sig_r = float(np.sqrt(cov[2, 2]))
    except np.linalg.LinAlgError:
        sig_c = sig_r = float("inf")
    ang = np.unwrap(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))
    return CircleFit(centre, radius, rms, sig_r, sig_c, float(np.degrees(np.ptp(ang))))


def pendulum_theta(xy: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """theta = atan2(x_bob - x_pivot, -(y_bob - y_pivot)), zero hanging straight down."""
    d = np.asarray(xy, float) - np.asarray(pivot, float)
    return np.arctan2(d[:, 0], -d[:, 1])


def effective_length(length: float, bob_radius: float) -> float:
    """L_eff = L + (2/5) r^2 / L for a solid spherical bob on a light string."""
    return length + 0.4 * bob_radius**2 / length


@dataclass
class SpinnerAngle:
    phi: np.ndarray  # unwrapped, radians, counter-clockwise
    alias_risk: np.ndarray  # per sample: the step into it exceeds 120 degrees per frame
    max_step_deg: float
    direction: int  # +1 counter-clockwise, -1 clockwise, 0 not moving
    direction_consistent: bool
    n_reversals: int


def spinner_phi(
    xy_marker: np.ndarray, hub: np.ndarray, frames: np.ndarray, sigma_phi: float = 0.0
) -> SpinnerAngle:
    """phi = atan2(marker - hub), unwrapped over the kept samples.

    The step into each sample is divided by the number of frames it spans, so a dropped frame
    isn't mistaken for a faster spin. Direction consistency is judged on steps clearly above noise.
    """
    d = np.asarray(xy_marker, float) - np.asarray(hub, float)
    phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    step = np.diff(phi)
    gap = np.maximum(np.diff(np.asarray(frames)), 1)
    per_frame = np.degrees(np.abs(step)) / gap
    alias = np.concatenate([[False], per_frame > ALIAS_STEP_DEG])
    moving = np.abs(step) > max(3 * np.sqrt(2) * sigma_phi, np.radians(0.5))
    signs = np.sign(step[moving])
    if len(signs) == 0:
        return SpinnerAngle(phi, alias, float(per_frame.max(initial=0)), 0, True, 0)
    direction = 1 if signs.sum() >= 0 else -1
    wrong = int(np.sum(signs != direction))
    return SpinnerAngle(
        phi, alias, float(per_frame.max(initial=0)), direction, wrong <= 0.1 * len(signs), wrong
    )
