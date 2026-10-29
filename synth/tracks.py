"""Per-frame tracks straight from the true motion, for tests that need neither video nor SAM."""

from __future__ import annotations

import numpy as np

from .render import Scene


def analytic_tracks(
    scene: Scene,
    duration_s: float,
    fps: float = 60.0,
    key: str | None = None,
    sigma_px: float = 0.2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """What a segmentation would give for the object `key` (default: the primary one).

    Returns arrays frame, t (real seconds), cx, cy (pixels, with Gaussian noise `sigma_px` on each
    axis), area (a camera-facing disc of the object's radius at its true depth, 1% noise), conf,
    plus xyz_m (the true position) for scoring. Frame n is at t = n / fps.
    """
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    key = key or scene.system.primary
    xyz = scene.system.truth(t)["tracks"][key]
    rng = np.random.default_rng(seed)
    px = scene.cam.project(xyz) + rng.normal(0, sigma_px, (n, 2))
    radius = {"marker": 0.006, "hub": 0.007}.get(key, getattr(scene.system, "r", 0.02))
    area = np.pi * (scene.cam.f_px * radius / scene.cam.depth(xyz)) ** 2 * rng.normal(1, 0.01, n)
    return {
        "frame": np.arange(n), "t": t, "cx": px[:, 0], "cy": px[:, 1], "area": area,
        "conf": np.full(n, 0.95), "xyz_m": xyz,
    }  # fmt: skip
