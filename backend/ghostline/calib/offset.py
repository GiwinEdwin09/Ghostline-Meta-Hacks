"""Plane-offset correction: the object moves in a plane d in front of the sheet.

A point d metres nearer the camera than the sheet plane appears magnified by Z / (Z - d), so
pixel positions mapped through the sheet's homography read too large. Scaling them toward the
point under the camera by (1 - d / Z) puts them in the plane of motion.
"""

from __future__ import annotations

import numpy as np

DEFAULT_F35_MM = 26.0  # the 1x main camera on most phones
F_UNCERTAINTY = 0.15  # relative; the focal length guess is good to about 15%
D_SIGMA_M = 0.005
LONG_SIDE_M = {"A4": 0.297, "Letter": 0.2794}


def focal_px(image_width_px: float, f35_mm: float | None = None) -> float:
    """Focal length in pixels from the 35 mm-equivalent focal length (default 26 mm)."""
    return image_width_px * (f35_mm or DEFAULT_F35_MM) / 36.0


def camera_distance_m(f_px: float, long_side_px: float, paper: str = "A4") -> float:
    """Camera-to-sheet distance: Z = f * (long side in metres) / (long side in pixels)."""
    return f_px * LONG_SIDE_M[paper] / long_side_px


def offset_scale(d_m: float, z_m: float) -> float:
    return 1.0 - d_m / z_m


def _decompose(h_px_to_mm: np.ndarray, f_px: float, image_size: tuple[int, int]):
    """M = K^-1 H^-1 (sheet in metres) = kappa [r1 r2 t]; returns (M, kappa) or None."""
    w, h = image_size
    k = np.array([[f_px, 0, (w - 1) / 2], [0, f_px, (h - 1) / 2], [0, 0, 1.0]])
    a = np.linalg.inv(h_px_to_mm) @ np.diag([1000.0, 1000.0, 1.0])  # px ~ a @ [X, Y, 1] (metres)
    m = np.linalg.inv(k) @ a
    l1, l2 = np.linalg.norm(m[:, 0]), np.linalg.norm(m[:, 1])
    if not 0.7 < l1 / l2 < 1.4:
        return None
    return m, (l1 + l2) / 2 * (1 if m[2, 2] > 0 else -1)


def pose_aspect_ratio(
    h_px_to_mm: np.ndarray, f_px: float, image_size: tuple[int, int]
) -> float | None:
    """How much wider (x) than tall (y) the real sheet is than the paper size assumed in H.

    A rectangle seen by a camera with known focal length has orthogonal, equally long image axes;
    for the wrong paper size they differ by the ratio of the sizes (A4 vs Letter is 9%). About 1
    for the right paper, unless the focal-length guess is far off and the camera strongly tilted.
    """
    dec = _decompose(h_px_to_mm, f_px, image_size)
    if dec is None:
        return None
    m, _ = dec
    return float(np.linalg.norm(m[:, 0]) / np.linalg.norm(m[:, 1]))


def pose_from_homography(
    h_px_to_mm: np.ndarray, f_px: float, image_size: tuple[int, int]
) -> tuple[float, np.ndarray] | None:
    """Camera distance perpendicular to the sheet and the point of the sheet under the camera.

    Decomposes the sheet homography with the focal-length guess. Exact for any tilt, where the
    plan's Z = f * 0.297 / long side and c = H(image centre) hold for a head-on camera. Returns
    (Z in metres, foot point in sheet millimetres), or None if the decomposition is implausible.
    """
    dec = _decompose(h_px_to_mm, f_px, image_size)
    if dec is None:
        return None
    m, kappa = dec
    r1, r2, t = m[:, 0] / kappa, m[:, 1] / kappa, m[:, 2] / kappa
    r3 = np.cross(r1, r2)
    r3 /= np.linalg.norm(r3)
    z = abs(float(r3 @ t))
    if not 0.2 < z < 50:
        return None
    foot = -1000.0 * np.array([r1 @ t, r2 @ t])
    return z, foot


def depth_of_sheet_points(
    h_px_to_mm: np.ndarray, f_px: float, image_size: tuple[int, int], sheet_mm: np.ndarray
) -> np.ndarray | None:
    """Distance along the optical axis (metres) to points of the sheet plane given in sheet mm.

    A sphere looks like a disc of radius f r / depth wherever it is, so area * (depth / f)^2 is
    constant for in-plane motion even when the camera isn't square to the sheet.
    """
    dec = _decompose(h_px_to_mm, f_px, image_size)
    if dec is None:
        return None
    m, kappa = dec
    p = np.atleast_2d(sheet_mm) / 1000.0
    return (p[:, 0] * m[2, 0] + p[:, 1] * m[2, 1] + m[2, 2]) / kappa


def to_plane_of_motion(sheet_mm: np.ndarray, centre_mm: np.ndarray, scale: float) -> np.ndarray:
    """world = c + (H(p) - c) * (1 - d/Z), for sheet-frame points of shape (N, 2)."""
    return centre_mm + (np.asarray(sheet_mm) - centre_mm) * scale


def from_plane_of_motion(plane_mm: np.ndarray, centre_mm: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of `to_plane_of_motion`, used to draw the ghost back on the image."""
    return centre_mm + (np.asarray(plane_mm) - centre_mm) / scale


def long_side_px(corners_px: np.ndarray) -> float:
    """Mean length of the longer pair of sides of an ordered image quad."""
    c = np.asarray(corners_px, float)
    pairs = [
        (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2,
        (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2,
    ]
    return float(max(pairs))
