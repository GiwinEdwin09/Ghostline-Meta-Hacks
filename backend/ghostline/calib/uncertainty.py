"""Monte Carlo uncertainty of the pixel-to-metre scale and the ramp angle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gravity import physics_axes
from .homography import apply_h, compute_homography, local_scale_many
from .offset import D_SIGMA_M, F_UNCERTAINTY

N_DRAWS = 500


@dataclass
class ScaleUncertainty:
    sigma_scale_rel: float  # all sources together; the worst of five image points
    components: dict[str, float]  # corners, plane_offset_d, distance_z (relative)
    sigma_ramp_deg: float | None  # from the corner jitter only (the caller adds the line fit)


def _scale(h: np.ndarray, points_px: np.ndarray, d: float, z: float) -> np.ndarray:
    return local_scale_many(h, points_px) * (1.0 - d / z)


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
    (for example the path of the object). d and Z get their own Gaussian draws with sigma 0.5 cm and 15% of Z (treating the quoted
    bounds as one sigma, which is conservative).
    """
    rng = np.random.default_rng(seed)
    corners_px = np.asarray(corners_px, float)
    # the scale is judged at the image centre and four points toward its corners, because a
    # homography built from a small sheet is least certain far from it
    c = np.asarray(centre_px, float)
    points = np.vstack([c, c + [[0.6 * c[0], 0.6 * c[1]], [-0.6 * c[0], 0.6 * c[1]],
                                [0.6 * c[0], -0.6 * c[1]], [-0.6 * c[0], -0.6 * c[1]]]])  # fmt: skip
    if points_px is not None:
        points = np.atleast_2d(np.asarray(points_px, float))
    base = _scale(compute_homography(corners_px, corners_mm), points, d_m, z_m)

    def run(corners: bool, offset: bool, distance: bool) -> np.ndarray:
        out = np.empty((n, len(points)))
        for i in range(n):
            c = corners_px + rng.normal(0, sigma_corner_px, (4, 2)) if corners else corners_px
            d = d_m + rng.normal(0, D_SIGMA_M) if offset else d_m
            z = z_m * (1 + rng.normal(0, F_UNCERTAINTY)) if distance else z_m
            out[i] = _scale(compute_homography(c, corners_mm), points, d, z)
        return out

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
            h = compute_homography(
                corners_px + rng.normal(0, sigma_corner_px, (4, 2)), corners_mm
            )
            pts = apply_h(h, ramp_edge_px)
            u = np.linalg.svd(pts - pts.mean(axis=0))[2][0]
            right, up = physics_axes(down)
            angles[i] = np.degrees(np.arctan2(u @ up, u @ right))
        angles = (angles + 90) % 180 - 90
        sigma_ramp = float(np.std(angles - np.median(angles)))
    return ScaleUncertainty(rel(total), comps, sigma_ramp)
