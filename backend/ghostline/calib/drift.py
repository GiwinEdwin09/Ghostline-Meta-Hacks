"""Camera drift: how far the whole picture shifts during a clip, and taking it back out.

A phone settling on its mount after the record tap, or a tripod sagging, moves every pixel by
about the same few pixels. Following a few hundred background corners from the first sampled
frame (pyramidal Lucas-Kanade) and taking the median of their moves measures that shift without
SAM; a ball or a hand holds few of the corners, so the median ignores it. Subtracting the shift
from the tracked positions puts them back in the
reference picture the homography was fitted on. Only translation is corrected: the corner check
that runs on the corrected corners still stops a clip whose camera rotated or zoomed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ghostline.ingest.frames import iter_frames

SAMPLE_HZ = 30.0  # drift is smooth; 30 samples a second is plenty
ANALYSIS_WIDTH = 320
SMOOTH = 5  # samples in the running median, so a hand sweeping across one frame doesn't count
MAX_CORNERS = 300
MIN_CORNERS = 12  # fewer than this (a blank scene) and the shift isn't measured at all
MAX_CORRECTABLE_PX = 60.0  # master pixels; beyond this the view changes too much to trust a shift


@dataclass
class Drift:
    """The picture's shift at sampled master frames, relative to the reference picture (the
    per-axis median position, which is where the median image the sheet was found on sits)."""

    frames: np.ndarray  # master frame indices, ascending
    # (n, 2): where the picture sits at that frame minus where it sits in the reference
    shift_px: np.ndarray

    @property
    def max_px(self) -> float:
        return float(np.linalg.norm(self.shift_px, axis=1).max()) if len(self.frames) else 0.0

    def at(self, frames) -> np.ndarray:
        """The shift at any master frames, interpolated between samples, as (len, 2)."""
        f = np.asarray(frames, float)
        if not len(self.frames):
            return np.zeros((len(f), 2))
        return np.column_stack([np.interp(f, self.frames, self.shift_px[:, k]) for k in range(2)])

    def save(self, path: str | Path) -> None:
        np.savez(path, frames=self.frames, shift_px=self.shift_px)

    @classmethod
    def load(cls, path: str | Path) -> Drift:
        data = np.load(path)
        return cls(data["frames"], data["shift_px"])


def shifts_from_frames(frames: Iterable[tuple[int, np.ndarray]], scale: float = 1.0) -> Drift:
    """Follow the first image's corners into each (master frame index, grayscale image) and take
    the median move. An empty Drift when the scene has too few corners to measure.

    `scale` converts the images' pixels to master pixels (master width / image width).
    """
    ref = corners = None
    idx, raw = [], []
    for i, gray in frames:
        g = np.clip(gray, 0, 255).astype(np.uint8)
        if ref is None:
            ref = g
            corners = cv2.goodFeaturesToTrack(ref, MAX_CORNERS, 0.01, 6)
            if corners is None or len(corners) < MIN_CORNERS:
                return Drift(np.zeros(0), np.zeros((0, 2)))
        moved, found, _ = cv2.calcOpticalFlowPyrLK(
            ref, g, corners, None, winSize=(21, 21), maxLevel=3
        )
        ok = found.ravel() == 1
        if ok.sum() < MIN_CORNERS:
            continue  # this sample is mostly covered (a hand over the lens); interpolate across
        dx, dy = np.median((moved[ok] - corners[ok]).reshape(-1, 2), axis=0)
        idx.append(i)
        raw.append((dx * scale, dy * scale))
    if not idx:
        return Drift(np.zeros(0), np.zeros((0, 2)))
    shifts = np.asarray(raw, float)
    if len(shifts) >= SMOOTH:
        half = SMOOTH // 2
        padded = np.pad(shifts, ((half, half), (0, 0)), mode="edge")
        shifts = np.stack([np.median(padded[k : k + SMOOTH], axis=0) for k in range(len(raw))])
    return Drift(np.asarray(idx), shifts - np.median(shifts, axis=0))


def estimate_drift(video_path: str | Path, real_fps: float, master_width: int) -> Drift:
    """Sample the master at about SAMPLE_HZ and measure the picture's shift at each sample."""
    stride = max(1, round(real_fps / SAMPLE_HZ))
    frames = (
        (i, gray)
        for i, _, gray in iter_frames(video_path, width=ANALYSIS_WIDTH, gray=True)
        if i % stride == 0
    )
    return shifts_from_frames(frames, scale=master_width / ANALYSIS_WIDTH)


def undrift(cx: np.ndarray, cy: np.ndarray, frames: np.ndarray, drift: Drift):
    """Tracked pixel positions moved back into the reference picture."""
    shift = drift.at(frames)
    return np.asarray(cx, float) - shift[:, 0], np.asarray(cy, float) - shift[:, 1]
