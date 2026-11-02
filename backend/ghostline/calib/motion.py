"""Camera motion: how far the sheet's corners move between frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

WARN_PX = 1.0
STOP_PX = 3.0


@dataclass
class CameraMotion:
    max_shift_px: float
    status: Literal["ok", "warn", "stop"]
    shifts_px: list[float]  # per compared frame, relative to the first


def camera_motion(corner_sets: np.ndarray | list) -> CameraMotion:
    """How far the sheet's corners moved from the first frame (use the first, middle and last).

    Each frame's shift is the median over its four corners: a rotation, zoom or shift moves all
    four, while one corner a hand or the ball happens to cover moves alone and isn't camera
    motion.
    """
    sets = np.asarray(corner_sets, float).reshape(-1, 4, 2)
    moves = np.linalg.norm(sets[1:] - sets[0], axis=2)
    shifts = np.median(moves, axis=1) if len(sets) > 1 else [0.0]
    worst = float(np.max(shifts))
    status = "stop" if worst > STOP_PX else "warn" if worst > WARN_PX else "ok"
    return CameraMotion(worst, status, [float(s) for s in shifts])
