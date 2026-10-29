"""Synthetic phone-video generator with ground truth (build plan Part 26).

    from synth import ClipSpec, GroundTruth, render_clip

    result = render_clip(ClipSpec(system="pendulum", duration_s=3), tmp_dir)
    truth = GroundTruth.load(result.truth)

`build_scene(spec)` returns only the camera, sheet mask and plate, without rendering frames.
"""

from .render import ClipResult, ClipSpec, Scene, build_scene, render_clip
from .systems import SYSTEMS
from .tracks import analytic_tracks as analytic_track_arrays
from .truth import GroundTruth

__all__ = [
    "SYSTEMS",
    "ClipResult",
    "ClipSpec",
    "GroundTruth",
    "Scene",
    "analytic_track_arrays",
    "build_scene",
    "render_clip",
]
