"""Auto-trim: find the window where something moves."""

from __future__ import annotations

import numpy as np

from .frames import iter_frames
from .models import TrimSuggestion

ANALYSIS_WIDTH = 160
CHANGE_GREY = 15.0  # a pixel has changed when it moved by more than this; sensor noise is ~1-2
MIN_CHANGED = 4  # pixels; a 3 px ball moving by one pixel changes about this many
MAD_K = 5.0
PAD_S = 0.3
BRIDGE_S = 0.25  # gaps shorter than this inside the motion are closed (turning points)
MIN_RUN = 3  # frames
MIN_RUN_SCORE = 0.25  # a run must hold this share of the strongest run's excess energy


def motion_energy(path, lag: int = 1) -> tuple[np.ndarray, float, int]:
    """How many pixels changed by more than CHANGE_GREY between frames `lag` apart, at 160 px
    grayscale.

    The plan's mean absolute difference hides a small object (a 5 px pendulum bob barely moves a
    frame mean) and has no zero, so motion that fills the clip looks like noise. Counting strong
    changes leaves a static tripod scene at about zero and keeps small objects visible.

    A lag of about real_fps / 60 keeps the energy comparable across frame rates. Returns
    (energy per frame, median luma 0-255, frame count); the first `lag` entries repeat energy[lag].
    """
    energy, lumas, window = [], [], []
    for _, _, gray in iter_frames(path, width=ANALYSIS_WIDTH, gray=True):
        g = gray.astype(np.float32)
        lumas.append(float(g.mean()))
        changed = np.abs(g - window[0]) > CHANGE_GREY if len(window) == lag else None
        energy.append(float(changed.sum()) if changed is not None else np.nan)
        window.append(g)
        window = window[-lag:]
    energy = np.asarray(energy)
    if len(energy) > lag:
        energy[:lag] = energy[lag]
    return energy, float(np.median(lumas)) if lumas else 0.0, len(lumas)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0], strict=True))


def suggest_trim(
    energy: np.ndarray, t_real: np.ndarray, usable: tuple[int, int] | None = None
) -> TrimSuggestion:
    """Window where the changed-pixel count exceeds the stillest frames' level + 5 MAD (and at
    least MIN_CHANGED), padded 0.3 s each side.

    `usable` restricts the search to frames [lo, hi), for clips with a mixed frame rate.
    """
    lo, hi = usable or (0, len(energy))
    t_real, energy = t_real[lo:hi], energy[lo:hi]
    n = len(energy)
    # The floor comes from the stillest few frames, not the median: a swing or a spin-down can fill
    # nearly the whole clip, and only its turning points or the stopped end are still.
    base = float(np.percentile(energy, 2))
    still = energy[energy <= np.percentile(energy, 5)]
    mad = float(np.median(np.abs(still - base))) * 1.4826
    threshold = max(base + MAD_K * mad, float(MIN_CHANGED))
    active = energy > threshold
    dt = float(np.median(np.diff(t_real))) if n > 1 else 1.0
    bridge = max(1, round(BRIDGE_S / dt))
    closed = active.copy()
    found = _runs(active)
    for (_, gap_start), (gap_stop, _) in zip(found, found[1:], strict=False):
        if gap_stop - gap_start <= bridge:
            closed[gap_start:gap_stop] = True
    runs = [(a, b) for a, b in _runs(closed) if b - a >= MIN_RUN]
    if runs:  # keep the strongest runs; a lone keyframe pulse or a hand moving away is minor
        score = [float(np.sum(energy[a:b] - threshold)) for a, b in runs]
        runs = [r for r, sc in zip(runs, score, strict=True) if sc >= MIN_RUN_SCORE * max(score)]
    if not runs:
        start, stop, reason = 0, n, "no_motion_detected"
    else:
        pad = max(1, round(PAD_S / dt))
        start, stop, reason = max(runs[0][0] - pad, 0), min(runs[-1][1] + pad, n), "motion_window"
    return TrimSuggestion(
        start_frame=lo + start, stop_frame=lo + stop, start_t=float(t_real[start]),
        stop_t=float(t_real[stop - 1]), reason=reason, threshold=threshold,
        n_active=int(active.sum()),
    )  # fmt: skip
