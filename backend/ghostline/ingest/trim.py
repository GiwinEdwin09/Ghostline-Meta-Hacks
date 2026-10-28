"""Auto-trim: find the window where something moves."""

from __future__ import annotations

import numpy as np

from .frames import iter_frames
from .models import TrimSuggestion

ANALYSIS_WIDTH = 160
MAD_K = 5.0
ENERGY_FLOOR = 0.003  # grey levels, so a noiseless static clip doesn't give a zero threshold
PAD_S = 0.3
BRIDGE_S = 0.25  # gaps shorter than this inside the motion are closed (turning points)
MIN_RUN = 3  # frames
MIN_RUN_SCORE = 0.25  # a run must hold this share of the strongest run's excess energy


def motion_energy(path, lag: int = 1) -> tuple[np.ndarray, float, int]:
    """Mean absolute difference between frames `lag` apart, at 160 px grayscale.

    A lag of about real_fps / 60 keeps the energy comparable across frame rates. Returns
    (energy per frame, median luma 0-255, frame count); the first `lag` entries repeat energy[lag].
    """
    energy, lumas, window = [], [], []
    for _, _, gray in iter_frames(path, width=ANALYSIS_WIDTH, gray=True):
        g = gray.astype(np.float32)
        lumas.append(float(g.mean()))
        energy.append(float(np.abs(g - window[0]).mean()) if len(window) == lag else np.nan)
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
    """Window where the frame difference exceeds median + 5 MAD, padded by 0.3 s each side.

    `usable` restricts the search to frames [lo, hi), for clips with a mixed frame rate.
    """
    lo, hi = usable or (0, len(energy))
    t_real, energy = t_real[lo:hi], energy[lo:hi]
    n = len(energy)
    median = float(np.median(energy))
    mad = float(np.median(np.abs(energy - median))) * 1.4826
    threshold = median + MAD_K * max(mad, ENERGY_FLOOR)
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
