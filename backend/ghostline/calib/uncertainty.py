"""Monte Carlo uncertainty of the pixel-to-metre scale and the ramp angle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gravity import physics_axes
from .homography import apply_h, compute_homography
from .offset import D_SIGMA_M, F_UNCERTAINTY

N_DRAWS = 500


@dataclass
class ScaleUncertainty:
    sigma_scale_rel: float  # all sources together; the worst of five image points
    components: dict[str, float]  # corners, plane_offset_d, distance_z (relative)
    sigma_ramp_deg: float | None  # from the corner jitter only (the caller adds the line fit)


def homographies(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct linear solves for a batch: src (n, 4, 2) pixel corners -> fixed dst (4, 2) mm."""
    n = len(src)
    a = np.zeros((n, 8, 9))
    x, y = src[..., 0], src[..., 1]
    u, v = dst[:, 0], dst[:, 1]
    a[:, 0::2, 0], a[:, 0::2, 1], a[:, 0::2, 2] = x, y, 1
    a[:, 1::2, 3], a[:, 1::2, 4], a[:, 1::2, 5] = x, y, 1
    a[:, 0::2, 6], a[:, 0::2, 7], a[:, 0::2, 8] = -u * x, -u * y, -u
    a[:, 1::2, 6], a[:, 1::2, 7], a[:, 1::2, 8] = -v * x, -v * y, -v
    h = np.linalg.svd(a)[2][:, -1].reshape(n, 3, 3)
    return h / h[:, 2:3, 2:3]


def _scale(hs: np.ndarray, points_px: np.ndarray, d: np.ndarray, z: np.ndarray) -> np.ndarray:
    """mm per pixel at each point, in the plane of motion: (n, m) for n homographies."""

    def apply(shift):
        p = np.column_stack([points_px + shift, np.ones(len(points_px))])
        q = np.einsum("nij,mj->nmi", hs, p)
        return q[..., :2] / q[..., 2:3]

    eps = 0.5
    dx = (apply([eps, 0]) - apply([-eps, 0])) / (2 * eps)
    dy = (apply([0, eps]) - apply([0, -eps])) / (2 * eps)
    local = np.sqrt(np.abs(dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]))
    return local * (1.0 - d / z)[:, None]


def monte_carlo(
    corners_px: np.ndarray,
    corners_mm: np.ndarray,
    sigma_corner_px: float,
    centre_px: np.ndarray,
    d_m: float,
    z_m: float,
    ramp_edge_px: np.ndarray | None = None,
    down: np.ndarray | None = None,
    n: int = N_DRAWS,
    seed: int = 0,
    points_px: np.ndarray | None = None,
) -> ScaleUncertainty:
    """Spread of the scale over `n` draws: corners ~ N(0, sigma_c), d +/- 0.5 cm, Z +/- 15%.

    The scale is judged at the image centre and four points toward its corners, or at `points_px`
    (for example the path of the object). d and Z get their own Gaussian draws with sigma 0.5 cm
    and 15% of Z (treating the quoted bounds as one sigma, which is conservative).
    """
    rng = np.random.default_rng(seed)
    corners_px = np.asarray(corners_px, float)
    # a homography built from a small sheet is least certain far from it, so look there too
    c = np.asarray(centre_px, float)
    offsets = 0.6 * c * np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    points = np.vstack([c, c + offsets])
    if points_px is not None:
        points = np.atleast_2d(np.asarray(points_px, float))
    corners_mm = np.asarray(corners_mm, float)
    base = _scale(
        homographies(corners_px[None], corners_mm), points, np.array([d_m]), np.array([z_m])
    )[0]

    def run(corners: bool, offset: bool, distance: bool) -> np.ndarray:
        src = (
            corners_px + rng.normal(0, sigma_corner_px, (n, 4, 2))
            if corners
            else np.tile(corners_px, (n, 1, 1))
        )
        d = d_m + rng.normal(0, D_SIGMA_M, n) if offset else np.full(n, d_m)
        z = z_m * (1 + rng.normal(0, F_UNCERTAINTY, n)) if distance else np.full(n, z_m)
        return _scale(homographies(src, corners_mm), points, d, z)

    def rel(draws: np.ndarray) -> float:
        return float(np.max(np.std(draws, axis=0) / base))

    total = run(True, True, True)
    comps = {
        "corners": rel(run(True, False, False)),
        "plane_offset_d": rel(run(False, True, False)),
        "distance_z": rel(run(False, False, True)),
    }
    sigma_ramp = None
    if ramp_edge_px is not None and down is not None:
        angles = np.empty(n)
        for i in range(n):
            h = compute_homography(corners_px + rng.normal(0, sigma_corner_px, (4, 2)), corners_mm)
            pts = apply_h(h, ramp_edge_px)
            u = np.linalg.svd(pts - pts.mean(axis=0))[2][0]
            right, up = physics_axes(down)
            angles[i] = np.degrees(np.arctan2(u @ up, u @ right))
        angles = (angles + 90) % 180 - 90
        sigma_ramp = float(np.std(angles - np.median(angles)))
    return ScaleUncertainty(rel(total), comps, sigma_ramp)
