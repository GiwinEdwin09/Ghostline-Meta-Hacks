"""Draw the ghost on trial 2's video (build plan Part 16). CPU only, sized for `cpu=4`.

    plan = RenderPlan(...)                       # per-frame positions, errors and colours
    render_ghost_video(master, plan, out_mp4, slow_path=out_slow_mp4)
    stills = render_stills(master, plan, [120, 180, 240, 300])

One pass: decode the master with PyAV, draw on each frame, and encode H.264 with the master's real
timestamps (variable frame rate is kept) and faststart. A slow-motion clip is written a second
time at 30 fps, in the same pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from .model import Status
from .sprite import Sprite, draw_sprite

TIME_BASE = Fraction(1, 90000)
SLOW_FPS = 30
TRAIL_S = 0.5  # the real trail: thin white, the last half second
PATH_BEHIND_S = 0.5
PATH_AHEAD_S = 1.5
DOT_SPACING_PX = 10.0
WHITE = (255, 255, 255)
CYAN_TUPLE = (255, 255, 0)
VIOLET = (250, 139, 167)  # BGR of the web's launch-window violet
STATUS_TEXT = {"green": (110, 215, 90), "amber": (40, 185, 255), "red": (90, 90, 245)}  # BGR


@dataclass
class RenderPlan:
    """Everything the drawing needs, per frame of the ghost video (index i)."""

    frames: np.ndarray  # master frame indices, consecutive
    t_rel: np.ndarray  # real seconds since the first frame of the video
    predicted_px: np.ndarray  # (n, 2)
    actual_px: np.ndarray  # (n, 2)
    error_mm: np.ndarray
    status: list[Status]
    sprite: Sprite
    launch_end_rel: float
    label: str = ""
    anchor_px: np.ndarray | None = None  # the pivot, for the ghost string
    angle_deg: np.ndarray | None = None  # sprite rotation relative to its source frame
    slow_motion: bool = False
    _dots: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._dots = _dots_along(self.predicted_px, self.t_rel)


def _dots_along(px: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points spaced DOT_SPACING_PX apart along the predicted path, as rows (t, x, y)."""
    if len(px) == 0:
        return np.zeros((0, 3))
    rows = [(t[0], px[0, 0], px[0, 1])]
    travelled = 0.0
    for i in range(1, len(px)):
        step = float(np.hypot(*(px[i] - px[i - 1])))
        travelled += step
        if travelled >= DOT_SPACING_PX:
            rows.append((t[i], px[i, 0], px[i, 1]))
            travelled = 0.0
    return np.array(rows)


@lru_cache(maxsize=1)
def _font_path() -> str:
    return font_manager.findfont("DejaVu Sans", fallback_to_default=True)


@lru_cache(maxsize=512)
def _text_layer(text: str, bgr: tuple[int, int, int], size: int) -> tuple[np.ndarray, np.ndarray]:
    """(colour, alpha) layers of a text with a dark outline. Cached: HUD strings repeat."""
    font = ImageFont.truetype(_font_path(), size)
    stroke = max(1, size // 10)
    x0, y0, x1, y1 = font.getbbox(text, stroke_width=stroke)
    image = Image.new("RGBA", (x1 - x0 + 2 * stroke, y1 - y0 + 2 * stroke), (0, 0, 0, 0))
    ImageDraw.Draw(image).text(
        (stroke - x0, stroke - y0),
        text,
        font=font,
        fill=(bgr[2], bgr[1], bgr[0], 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )
    arr = np.asarray(image, dtype=np.float32)
    return arr[..., 2::-1].copy(), arr[..., 3] / 255.0  # RGB -> BGR


def put_text(frame: np.ndarray, text: str, xy: tuple[int, int], bgr, size: int) -> tuple[int, int]:
    """Draw `text` with its top-left at `xy`. Returns the size of what was drawn."""
    color, alpha = _text_layer(text, tuple(int(c) for c in bgr), size)
    h, w = alpha.shape
    fh, fw = frame.shape[:2]
    x0, y0 = max(xy[0], 0), max(xy[1], 0)
    x1, y1 = min(xy[0] + w, fw), min(xy[1] + h, fh)
    if x1 > x0 and y1 > y0:
        a = alpha[y0 - xy[1] : y1 - xy[1], x0 - xy[0] : x1 - xy[0], None]
        c = color[y0 - xy[1] : y1 - xy[1], x0 - xy[0] : x1 - xy[0]]
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(roi * (1 - a) + c * a, 0, 255).astype(np.uint8)
    return w, h


def _blend_line(frame, p0, p1, bgr, alpha: float, thickness: int) -> None:
    """A translucent anti-aliased line, blended inside its bounding box only."""
    h, w = frame.shape[:2]
    pad = thickness + 2
    x0, x1 = int(max(min(p0[0], p1[0]) - pad, 0)), int(min(max(p0[0], p1[0]) + pad, w))
    y0, y1 = int(max(min(p0[1], p1[1]) - pad, 0)), int(min(max(p0[1], p1[1]) + pad, h))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    over = roi.copy()
    a = (round(p0[0] - x0), round(p0[1] - y0))
    b = (round(p1[0] - x0), round(p1[1] - y0))
    cv2.line(over, a, b, bgr, thickness, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)


def delta_text(error_mm: float) -> str:
    if error_mm < 1.0:
        return f"Δ {error_mm:.1f} mm"
    if error_mm < 10.0:
        return f"Δ {error_mm:.1f} mm"
    return f"Δ {error_mm:.0f} mm"


class Annotator:
    """Draws one frame's overlay. Sizes scale with the frame so 480p and 1080p look alike."""

    def __init__(self, plan: RenderPlan, width: int, height: int):
        self.plan = plan
        self.w, self.h = width, height
        self.unit = height / 480.0
        self.font = max(int(round(17 * self.unit)), 12)
        self.small = max(int(round(13 * self.unit)), 10)
        self.pad = max(int(round(10 * self.unit)), 6)
        self.thick = max(int(round(self.unit)), 1)
        self._t = plan.t_rel

    def draw(self, frame: np.ndarray, i: int) -> np.ndarray:
        plan = self.plan
        t = float(plan.t_rel[i])
        self._string(frame, i)
        self._path(frame, t)
        self._trail(frame, i, t)
        angle = 0.0 if plan.angle_deg is None else float(plan.angle_deg[i])
        draw_sprite(
            frame,
            plan.sprite,
            (float(plan.predicted_px[i, 0]), float(plan.predicted_px[i, 1])),
            angle,
            outline=plan.status[i],
            outline_px=max(int(round(2 * self.unit)), 2),
        )
        self._hud(frame, i, t)
        return frame

    # --- layers -----------------------------------------------------------------------------
    def _string(self, frame, i: int) -> None:
        if self.plan.anchor_px is not None:
            _blend_line(
                frame, self.plan.anchor_px, self.plan.predicted_px[i], CYAN_TUPLE, 0.3,
                self.thick,
            )  # fmt: skip

    def _path(self, frame, t: float) -> None:
        dots = self.plan._dots
        near = dots[(dots[:, 0] >= t - PATH_BEHIND_S) & (dots[:, 0] <= t + PATH_AHEAD_S)]
        radius = max(int(round(1.6 * self.unit)), 1)
        for dt, x, y in near:
            fade = 1.0 if dt >= t else 0.45
            color = tuple(int(c * fade) for c in CYAN_TUPLE)
            cv2.circle(frame, (round(x), round(y)), radius, color, -1, cv2.LINE_AA)

    def _trail(self, frame, i: int, t: float) -> None:
        t_arr = self._t
        lo = int(np.searchsorted(t_arr, t - TRAIL_S, side="left"))
        pts = self.plan.actual_px[lo : i + 1]
        if len(pts) >= 2:
            cv2.polylines(
                frame,
                [np.round(pts).astype(np.int32).reshape(-1, 1, 2)],
                False,
                WHITE,
                self.thick,
                cv2.LINE_AA,
            )

    def _hud(self, frame, i: int, t: float) -> None:
        plan = self.plan
        x, y = self.pad, self.pad
        status = plan.status[i]
        _, h1 = put_text(
            frame, delta_text(float(plan.error_mm[i])), (x, y), STATUS_TEXT[status], self.font
        )
        y += h1 + self.pad // 2
        ahead = t - plan.launch_end_rel
        if t < plan.launch_end_rel:
            put_text(frame, "launch window", (x, y), VIOLET, self.font)
            put_text(
                frame, "fitting the start", (x, y + self.font + self.pad // 2), WHITE, self.small
            )
        elif ahead > 0:
            put_text(frame, f"predicting {ahead:.1f} s ahead", (x, y), WHITE, self.small + 2)
        if plan.label:
            lh = _text_layer(plan.label, WHITE, self.small)[1].shape[0]
            put_text(frame, plan.label, (x, self.h - self.pad - lh), WHITE, self.small)


def _even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


class _Encoder:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, time_base: Fraction):
        self.container = av.open(str(path), "w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.thread_type = "AUTO"
        self.stream.codec_context.time_base = time_base
        self.stream.codec_context.options = {"crf": "20", "preset": "veryfast", "bf": "0"}
        self.time_base = time_base

    def write(self, bgr: np.ndarray, pts: int) -> None:
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


@dataclass
class RenderStats:
    n_frames: int
    width: int
    height: int
    seconds: float


def _decode(master: Path, first: int, last: int):
    """Yield (frame index, BGR array) for frames first..last of the master."""
    container = av.open(str(master))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index > last:
                break
            if index >= first:
                yield index, frame.to_ndarray(format="bgr24")
    finally:
        container.close()


def render_ghost_video(
    master: Path,
    plan: RenderPlan,
    out_path: Path,
    slow_path: Path | None = None,
    progress: Callable[[float], None] | None = None,
) -> RenderStats:
    """Decode the master, draw the overlay and encode `out_path` (and the 30 fps slow-motion
    version at `slow_path`). Frames come out with the master's real timestamps."""
    import time

    started = time.monotonic()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, last = int(plan.frames[0]), int(plan.frames[-1])
    encoder = slow = annotator = None
    size = (0, 0)
    n = 0
    try:
        for index, bgr in _decode(master, first, last):
            i = index - first
            if i >= len(plan.frames):
                break
            bgr = np.ascontiguousarray(_even(bgr))
            if encoder is None:
                size = (bgr.shape[1], bgr.shape[0])
                annotator = Annotator(plan, *size)
                real_fps = 1.0 / max(float(np.median(np.diff(plan.t_rel))), 1e-3)
                encoder = _Encoder(out_path, size, real_fps, TIME_BASE)
                if slow_path is not None:
                    slow = _Encoder(slow_path, size, SLOW_FPS, Fraction(1, SLOW_FPS))
            assert annotator is not None
            annotator.draw(bgr, i)
            encoder.write(bgr, int(round(float(plan.t_rel[i]) / float(TIME_BASE))))
            if slow is not None:
                slow.write(bgr, n)
            n += 1
            if progress is not None and n % 30 == 0:
                progress(n / len(plan.frames))
    finally:
        if encoder is not None:
            encoder.close()
        if slow is not None:
            slow.close()
    if encoder is None:
        raise ValueError(f"no frames in {master} between {first} and {last}")
    return RenderStats(n, size[0], size[1], time.monotonic() - started)


def render_stills(master: Path, plan: RenderPlan, indices: list[int]) -> dict[int, np.ndarray]:
    """Annotated frames at plan indices (positions in `plan.frames`), as BGR arrays."""
    if not indices:
        return {}
    wanted = {int(plan.frames[i]): i for i in indices}
    out: dict[int, np.ndarray] = {}
    annotator = None
    for index, bgr in _decode(master, min(wanted), max(wanted)):
        if index in wanted:
            bgr = np.ascontiguousarray(_even(bgr))
            annotator = annotator or Annotator(plan, bgr.shape[1], bgr.shape[0])
            out[wanted[index]] = annotator.draw(bgr, wanted[index])
    return out
