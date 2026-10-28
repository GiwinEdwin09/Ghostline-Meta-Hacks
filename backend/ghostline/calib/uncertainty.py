"""Monte Carlo uncertainty of the pixel-to-metre scale and the ramp angle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gravity import physics_axes
from .homography import apply_h, compute_homography, local_scale
from .offset import D_SIGMA_M, F_UNCERTAINTY

N_DRAWS = 500


@dataclass
class ScaleUncertainty:
    sigma_scale_rel: float  # all sources together
    components: dict[str, float]  # corners, plane_offset_d, distance_z (relative)
    sigma_ramp_deg: float | None  # from the corner jitter only (the caller adds the line fit)


def _scale(h: np.ndarray, centre_px: np.ndarray, d: float, z: float) -> float:
    return local_scale(h, centre_px) * (1.0 - d / z)


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
) -> ScaleUncertainty:
    """Spread of the scale over `n` draws: corners ~ N(0, sigma_c), d +/- 0.5 cm, Z +/- 15%.

    d and Z get their own Gaussian draws with sigma 0.5 cm and 15% of Z (treating the quoted
    bounds as one sigma, which is conservative).
    """
    rng = np.random.default_rng(seed)
    corners_px = np.asarray(corners_px, float)
    base = _scale(compute_homography(corners_px, corners_mm), centre_px, d_m, z_m)

    def run(corners: bool, offset: bool, distance: bool) -> np.ndarray:
        out = np.empty(n)
        for i in range(n):
            c = corners_px + rng.normal(0, sigma_corner_px, (4, 2)) if corners else corners_px
            d = d_m + rng.normal(0, D_SIGMA_M) if offset else d_m
            z = z_m * (1 + rng.normal(0, F_UNCERTAINTY)) if distance else z_m
            out[i] = _scale(compute_homography(c, corners_mm), centre_px, d, z)
        return out

    total = run(True, True, True)
    comps = {
        "corners": float(np.std(run(True, False, False)) / base),
        "plane_offset_d": float(np.std(run(False, True, False)) / base),
        "distance_z": float(np.std(run(False, False, True)) / base),
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
    return ScaleUncertainty(float(np.std(total) / base), comps, sigma_ramp)
