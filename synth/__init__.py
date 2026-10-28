"""Synthetic phone-video generator with ground truth (build plan Part 26).

    from synth import ClipSpec, GroundTruth, render_clip

    result = render_clip(ClipSpec(system="pendulum", duration_s=3), tmp_dir)
    truth = GroundTruth.load(result.truth)
"""

from .render import ClipResult, ClipSpec, render_clip
from .systems import SYSTEMS
from .truth import GroundTruth

__all__ = ["SYSTEMS", "ClipResult", "ClipSpec", "GroundTruth", "render_clip"]
