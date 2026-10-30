"""Cleaning: drop bad frames, never interpolate.

Frames are dropped for low confidence, a mask touching the frame border, a mask area far from its
rolling median, or a centroid that jumps away from its neighbours. Every dropped frame stays a
gap in the series.

SAM's confidence `c` is one number per object (the smoke test saw the same value on every frame),
so it can't tell good frames from bad ones. It is used per frame only to single out frames that
are less confident than the rest of their track, which happens when stitching joins a weak
re-detection onto a strong track. A track whose typical confidence is below the limit is a
low-confidence object, and dropping all its frames would throw away the whole clip: those frames
are kept, and `object_confidence_low` tells the caller to warn the user instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import TrackSeries
from .noise import robust_std

MIN_CONF = 0.5
AREA_MAD_K = 4.0
AREA_WINDOW = 15
AREA_MIN_REL_SCALE = 0.02  # pixelization noise floor on the area scale, as a fraction of area
JUMP_K = 5.0  # a jump is this many times the typical step
JUMP_NEIGHBOURS = 3
JUMP_NOISE_K = 2.5
JUMP_PASSES = 2


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    out = np.empty(n)
    half = window // 2
    for i in range(n):
        out[i] = np.median(x[max(0, i - half) : i + half + 1])
    return out


def area_outliers(area: np.ndarray) -> np.ndarray:
    """|area - rolling median| over 4 MAD (with a small floor); NaN areas are never flagged."""
    ok = np.isfinite(area)
    out = np.zeros(len(area), bool)
    if ok.sum() < 5:
        return out
    a = area[ok]
    resid = a - _rolling_median(a, AREA_WINDOW)
    scale = max(robust_std(resid), AREA_MIN_REL_SCALE * float(np.median(a)))
    out[ok] = np.abs(resid) > AREA_MAD_K * scale
    return out


def jump_outliers(px: np.ndarray) -> np.ndarray:
    """Centroids further from their neighbours' median than 5x the local step.

    The local step is the median consecutive step among the neighbours (not the point itself, whose
    own steps a glitch inflates), with a floor from the noise level. A spike and a short run of
    wrong frames are both caught; a permanent move to somewhere else isn't a jump.
    """
    n = len(px)
    out = np.zeros(n, bool)
    if n < 2 * JUMP_NEIGHBOURS + 1:
        return out
    second = px[2:] - 2 * px[1:-1] + px[:-2]
    sigma = np.mean([robust_std(second[:, j]) for j in range(2)]) / np.sqrt(6)
    floor = JUMP_NOISE_K * sigma
    steps = np.linalg.norm(np.diff(px, axis=0), axis=1)  # steps[i] joins i and i+1
    for i in range(n):
        lo, hi = max(0, i - JUMP_NEIGHBOURS), min(n, i + JUMP_NEIGHBOURS + 1)
        nb = [j for j in range(lo, hi) if j != i]
        if len(nb) < 4:
            continue
        pred = np.median(px[nb], axis=0)
        local = [steps[j] for j in range(lo, hi - 1) if j != i and j + 1 != i]
        ref = max(float(np.median(local)) if local else 0.0, floor)
        out[i] = np.linalg.norm(px[i] - pred) > JUMP_K * ref
    return out


def typical_confidence(track: TrackSeries) -> float:
    """The track's object-level confidence: the median over its frames (NaN if none is known)."""
    known = track.conf[np.isfinite(track.conf)]
    return float(np.median(known)) if len(known) else float("nan")


def object_confidence_low(track: TrackSeries, min_conf: float = MIN_CONF) -> bool:
    """True when the object as a whole scores below `min_conf`, so confidence can't be used to
    pick frames (see the module docstring)."""
    return bool(typical_confidence(track) < min_conf)


def clean_track(track: TrackSeries, min_conf: float = MIN_CONF) -> pd.DataFrame:
    """One row per input frame with the reason it was dropped ('' if kept)."""
    n = len(track)
    drop = np.full(n, "", dtype=object)
    low = (track.conf < min_conf) & (not object_confidence_low(track, min_conf))
    drop[low] = "low_conf"
    border = track.border & ~low
    drop[border] = "border"
    alive = drop == ""
    flagged = np.zeros(n, bool)
    idx = np.nonzero(alive)[0]
    flagged[idx] = area_outliers(track.area[idx])
    drop[flagged] = "area_outlier"
    for _ in range(JUMP_PASSES):
        idx = np.nonzero(drop == "")[0]
        bad = jump_outliers(track.px[idx])
        if not bad.any():
            break
        drop[idx[bad]] = "jump"
    return pd.DataFrame(
        {
            "frame": track.frame, "t": track.t, "cx": track.cx, "cy": track.cy,
            "area": track.area, "conf": track.conf, "drop": drop,
        }
    )  # fmt: skip
