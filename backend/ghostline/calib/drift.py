"""Camera drift: how far the whole picture shifts during a clip, and taking it back out.

A phone settling on its mount after the record tap, or a tripod sagging, moves every pixel by
about the same few pixels. Following a few hundred background corners from the first sampled
frame (pyramidal Lucas-Kanade) and taking the median of their moves measures that shift without
SAM; a ball or a hand holds few of the corners, so the median ignores it. Subtracting the shift
from the tracked positions puts them back in the
reference picture the homography was fitted on. Only translation is corrected: the corner check
that runs on the corrected corners still stops a clip whose camera rotated or zoomed. A camera that
jumps too far only at the start or end (picked up to stop the recording) is trimmed off instead:
see `steady_span`.
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
STEADY_MARGIN_S = 0.5  # a hand reaching for the phone moves it a little before the big jump


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

    def between(self, start: int, stop: int) -> Drift:
        """The samples in master frames [start, stop), relative to where the picture mostly sits
        there (the median image the sheet is found on comes from those frames)."""
        keep = (self.frames >= start) & (self.frames < stop)
        if not keep.any():
            return Drift(np.zeros(0), np.zeros((0, 2)))
        shift = self.shift_px[keep]
        return Drift(self.frames[keep], shift - np.median(shift, axis=0))

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


def steady_span(drift: Drift, start: int, stop: int, real_fps: float) -> tuple[int, int]:
    """The part of master frames [start, stop) to analyse when the camera moves too far to correct
    only at the start or end (settling after the record tap, a hand picking the phone up to stop):
    the longest run of samples within MAX_CORRECTABLE_PX, less STEADY_MARGIN_S where it was cut.

    [start, stop) itself when the camera stays within reach, or when the steady run is under half
    of it: a camera that moves throughout is left for calibration to refuse.
    """
    part = drift.between(start, stop)
    ok = np.linalg.norm(part.shift_px, axis=1) <= MAX_CORRECTABLE_PX
    if ok.all() or not ok.any():
        return start, stop
    runs = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0])).reshape(-1, 2)
    a, b = max(runs, key=lambda r: r[1] - r[0])  # sample indices, [a, b)
    margin = round(STEADY_MARGIN_S * real_fps)
    lo = start if a == 0 else int(part.frames[a - 1]) + margin
    hi = stop if b == len(ok) else int(part.frames[b]) - margin
    return (lo, hi) if hi - lo >= (stop - start) / 2 else (start, stop)


def undrift(cx: np.ndarray, cy: np.ndarray, frames: np.ndarray, drift: Drift):
    """Tracked pixel positions moved back into the reference picture."""
    shift = drift.at(frames)
    return np.asarray(cx, float) - shift[:, 0], np.asarray(cy, float) - shift[:, 1]
