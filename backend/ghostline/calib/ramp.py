"""Ramp angle from the board's top edge, cross-checked against the object's path."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ghostline.ingest.models import Notice

from .gravity import physics_axes
from .homography import apply_h

END_TRIM = 0.05
MIN_SIGMA_DEG = 0.1
CROSS_CHECK_DEG = 1.0


@dataclass
class RampAngle:
    angle_deg: float  # signed: positive when the board rises to the right
    incline_deg: float  # absolute angle from horizontal
    sigma_deg: float  # line-fit part only; gravity direction is added by the caller
    n_points: int


def top_edge_points(mask: np.ndarray) -> np.ndarray:
    """Top boundary of the board in pixels: one point per column, half a pixel above the mask."""
    m = np.asarray(mask) > 0
    cols = np.nonzero(m.any(axis=0))[0]
    if len(cols) < 10:
        raise ValueError("the ramp mask is too small")
    lo, hi = cols[0] + END_TRIM * len(cols), cols[-1] - END_TRIM * len(cols)
    cols = cols[(cols >= lo) & (cols <= hi)]
    tops = m[:, cols].argmax(axis=0)
    return np.column_stack([cols, tops - 0.5]).astype(float)


def direction_angle_deg(u_sheet: np.ndarray, down: np.ndarray) -> float:
    """Angle of a direction above horizontal in the physics frame, folded to (-90, 90]."""
    right, up = physics_axes(down)
    ur, uu = float(u_sheet @ right), float(u_sheet @ up)
    if ur < 0 or (ur == 0 and uu < 0):
        ur, uu = -ur, -uu
    return float(np.degrees(np.arctan2(uu, ur)))


def ramp_angle_from_mask(mask: np.ndarray, h: np.ndarray, down: np.ndarray) -> RampAngle:
    """Fit a line to the board's top edge in the image (lines stay lines under H), map its
    direction to the sheet frame and measure it against the horizontal."""
    pts = top_edge_points(mask)
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 1e-2, 1e-2).ravel()
    p0 = np.array([x0, y0], float)
    d = np.array([vx, vy], float)
    a, b = apply_h(h, np.array([p0, p0 + 100 * d]))
    u = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-d[1], d[0]])
    resid = (pts - p0) @ normal
    length = np.ptp((pts - p0) @ d)
    sigma = np.degrees(np.std(resid) * np.sqrt(12 / len(pts)) / max(length, 1.0))
    angle = direction_angle_deg(u, down)
    return RampAngle(angle, abs(angle), float(max(sigma, MIN_SIGMA_DEG)), len(pts))


def path_angle_deg(xy: np.ndarray) -> float:
    """Direction of the principal axis of a path (PCA), folded to (-90, 90]."""
    xy = np.asarray(xy, float)
    v = np.linalg.svd(xy - xy.mean(axis=0))[2][0]
    if v[0] < 0:
        v = -v
    return float(np.degrees(np.arctan2(v[1], v[0])))


def cross_check(edge_angle_deg: float, path_xy_world: np.ndarray) -> tuple[float, Notice | None]:
    """Difference between the board edge and the object's path, and a notice if over 1 degree."""
    diff = abs(edge_angle_deg - path_angle_deg(path_xy_world))
    if diff > CROSS_CHECK_DEG:
        return diff, Notice(
            code="ramp_angle_disagreement",
            message=f"The board edge ({edge_angle_deg:.1f} deg) and the object's path "
            f"({path_angle_deg(path_xy_world):.1f} deg) differ by {diff:.1f} deg. "
            "Check that the right board was segmented and the phone was level.",
        )
    return diff, None
