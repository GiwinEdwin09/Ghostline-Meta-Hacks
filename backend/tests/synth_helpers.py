"""Shared helpers for tests that use the synthetic generator in the top-level synth/ folder.

Adds the repo root to sys.path (synth/ is not part of the installed package), renders clips once
per test session into a temp folder that is removed at exit, and turns ground truth into the
plain inputs the calibration and trajectory code takes.
"""

from __future__ import annotations

import atexit
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synth import ClipSpec, GroundTruth, build_scene, render_clip  # noqa: E402

from ghostline.traj import TrackSeries  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def have_encoder(name: str) -> bool:
    if not HAVE_FFMPEG:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return name in out.stdout


_tmp = Path(tempfile.mkdtemp(prefix="ghostline-synth-"))
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)
_cache: dict[str, GroundTruth] = {}


def tmp_dir(name: str) -> Path:
    d = _tmp / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip(**kwargs) -> GroundTruth:
    """Render (once per session) the clip described by ClipSpec(**kwargs); returns its truth."""
    key = repr(sorted(kwargs.items(), key=lambda kv: kv[0]))
    if key not in _cache:
        name = hashlib.sha1(key.encode()).hexdigest()[:12]
        result = render_clip(ClipSpec(**kwargs), _tmp, name)
        _cache[key] = GroundTruth.load(result.truth)
    return _cache[key]


def scene(**kwargs):
    return build_scene(ClipSpec(**kwargs))


def tracks_from_truth(
    gt: GroundTruth,
    key: str | None = None,
    sigma_px: float = 0.2,
    seed: int = 0,
    conf: float = 0.95,
) -> TrackSeries:
    """What SAM would give: mask centroids and areas of the visible frames, plus centroid noise."""
    key = key or gt.data["primary"]
    tr = gt.track(key)
    c = tr["mask_centroid_px"].copy()
    if sigma_px:
        c = c + np.random.default_rng(seed).normal(0, sigma_px, c.shape)
    vis = tr["visible"]
    n = int(vis.sum())
    return TrackSeries.from_arrays(
        np.nonzero(vis)[0], c[vis, 0], c[vis, 1], tr["area_px"][vis], np.full(n, conf), None,
        t=gt.t_real[vis],
    )  # fmt: skip


def analytic_tracks(
    system: str,
    duration_s: float,
    fps: float = 60.0,
    sigma_px: float = 0.2,
    seed: int = 0,
    key: str | None = None,
    **spec,
):
    """Tracks straight from the true motion, with no video: (scene, calibration, track, t, truth).

    The calibration uses the exact sheet corners, so what remains is the trajectory stage itself.
    Mask areas follow a camera-facing disc of the object's radius at its true depth.
    """
    from synth.tracks import analytic_tracks as arrays

    from ghostline.calib import calibrate
    from ghostline.calib.homography import order_corners

    sc = scene(system=system, duration_s=duration_s, capture_fps=fps, **spec)
    cal = calibrate((854, 480), corners_px=order_corners(sc.corners_px), paper="A4")
    a = arrays(sc, duration_s, fps, key, sigma_px, seed)
    track = TrackSeries.from_arrays(
        a["frame"], a["cx"], a["cy"], a["area"], a["conf"], None, t=a["t"]
    )
    truth = sc.system.truth(a["t"])
    return sc, cal, track, a["t"], truth
