"""A small synthetic clip for smoke checks and fixtures: a bouncing orange ball and a blue dot
on a white sheet (A4 aspect, mild perspective) over a dark textured background, 854x480.

The ball's true centre is known to sub-pixel precision, so decoded SAM masks can be checked
against ground truth. Not the full generator of build plan Part 26 (`synth/render.py`).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 854, 480
SHEET_CORNERS = np.array([[140, 45], [705, 60], [690, 440], [155, 430]], dtype=np.float64)
BALL_RADIUS = 18.0
DOT_CENTER, DOT_RADIUS = (560.0, 120.0), 9.0
_FLOOR_Y = 405.0  # lowest y of the ball's lowest point
_GRAVITY = 1100.0  # px / s^2
_RESTITUTION = 0.82
_SHIFT = 4  # fractional bits for sub-pixel drawing


@dataclass(frozen=True)
class ClipTruth:
    """Ground truth for a rendered clip."""

    fps: float
    centers: np.ndarray  # (n_frames, 2) true ball centres in pixels (pixel-centre convention)
    radius: float
    dot_center: tuple[float, float]
    sheet_corners: np.ndarray

    def to_json(self) -> str:
        return json.dumps(
            {
                "fps": self.fps,
                "radius": self.radius,
                "dot_center": list(self.dot_center),
                "sheet_corners": self.sheet_corners.tolist(),
                "centers": np.round(self.centers, 4).tolist(),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ClipTruth:
        data = json.loads(text)
        return cls(
            fps=data["fps"],
            centers=np.array(data["centers"]),
            radius=data["radius"],
            dot_center=tuple(data["dot_center"]),
            sheet_corners=np.array(data["sheet_corners"]),
        )


def ball_center(t: float) -> tuple[float, float]:
    """Bouncing ball: free fall from y=90, restitution 0.82, drifting right at 55 px/s."""
    y_rest = _FLOOR_Y - BALL_RADIUS
    y0 = 90.0
    fall = float(np.sqrt(2 * (y_rest - y0) / _GRAVITY))
    x = 250.0 + 55.0 * t
    if t < fall:
        return x, y0 + 0.5 * _GRAVITY * t**2
    speed, start = _GRAVITY * fall * _RESTITUTION, fall
    for _ in range(60):
        duration = 2 * speed / _GRAVITY
        if t < start + duration:
            s = t - start
            return x, y_rest - speed * s + 0.5 * _GRAVITY * s**2
        start, speed = start + duration, speed * _RESTITUTION
    return x, y_rest


def _background() -> np.ndarray:
    rng = np.random.default_rng(7)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 6)
    gradient = np.linspace(30, 48, WIDTH, dtype=np.float32)[None, :]
    gray = gradient + 9.0 * texture / texture.std()
    return np.repeat(gray[:, :, None], 3, axis=2) * np.array([1.0, 1.02, 1.05], np.float32)


def _disc(image: np.ndarray, center: tuple[float, float], radius: float, color) -> None:
    scale = 1 << _SHIFT
    cv2.circle(
        image,
        (round(center[0] * scale), round(center[1] * scale)),
        round(radius * scale),
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
        shift=_SHIFT,
    )


def render_frame(t: float, index: int = 0, background: np.ndarray | None = None) -> np.ndarray:
    """Render one BGR uint8 frame at time `t` seconds."""
    image = (_background() if background is None else background).copy()
    sheet = np.round(SHEET_CORNERS * 16).astype(np.int32)
    cv2.fillConvexPoly(image, sheet, (238, 242, 244), lineType=cv2.LINE_AA, shift=4)
    _disc(image, DOT_CENTER, DOT_RADIUS, (200, 90, 20))
    cx, cy = ball_center(t)
    _disc(image, (cx, cy), BALL_RADIUS, (20, 110, 240))
    _disc(image, (cx - 5, cy - 6), BALL_RADIUS * 0.45, (90, 170, 255))
    noise = np.random.default_rng(1000 + index).normal(0, 2.5, image.shape).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 0.6) + noise
    return np.clip(image, 0, 255).astype(np.uint8)


def render_frames(
    seconds: float = 10.0, fps: float = 30.0
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time, BGR frame) for the whole clip."""
    background = _background()
    for index in range(round(seconds * fps)):
        t = index / fps
        yield index, t, render_frame(t, index, background)


def truth_for(seconds: float = 10.0, fps: float = 30.0) -> ClipTruth:
    centers = np.array([ball_center(i / fps) for i in range(round(seconds * fps))])
    return ClipTruth(fps, centers, BALL_RADIUS, DOT_CENTER, SHEET_CORNERS)


def write_video(path: Path, frames: Iterable[np.ndarray], fps: float = 30.0) -> None:
    """Encode BGR frames as an H.264 MP4 (faststart, yuv420p)."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "veryfast"}
        frames = iter(frames)
        first = next(frames)
        stream.height, stream.width = first.shape[:2]
        for bgr in itertools.chain([first], frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(bgr, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def render_clip(path: Path, seconds: float = 10.0, fps: float = 30.0) -> ClipTruth:
    """Write the synthetic clip as an H.264 MP4 and return its ground truth."""
    write_video(path, (bgr for _, _, bgr in render_frames(seconds, fps)), fps)
    return truth_for(seconds, fps)
