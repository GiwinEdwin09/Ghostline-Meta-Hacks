"""The ghost sprite (build plan Part 16): the object cut out of a trial-2 frame with its sharpest,
highest-confidence mask, tinted toward cyan, and drawn translucent.

    alpha  = the mask, feathered with a 2 px Gaussian blur, times 0.55
    colour = the object's own pixels blended 50% toward cyan
    glow   = the mask dilated and blurred, at 0.25
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

FEATHER_SIGMA_PX = 2.0
CYAN_BGR = np.array([255.0, 255.0, 0.0], np.float32)
CYAN_MIX = 0.5
OPACITY = 0.55
GLOW_OPACITY = 0.25
GLOW_GROW_PX = 3
GLOW_SIGMA_PX = 4.0
MARGIN_PX = 14  # room around the mask for the glow
CANDIDATES = 12
SHARPNESS_POOL = 5

OUTLINE_BGR = {"green": (96, 205, 74), "amber": (24, 176, 255), "red": (70, 70, 235)}


class MaskedTrack(Protocol):
    """The part of `vision.Track` that picks a sprite frame. A `Track` satisfies it."""

    frame: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area: np.ndarray
    conf: np.ndarray
    touches_border: np.ndarray

    def mask(self, frame: int) -> np.ndarray: ...


@dataclass
class Sprite:
    color: np.ndarray  # (h, w, 3) float32 BGR, already tinted
    alpha: np.ndarray  # (h, w) float32 in [0, 1], opacity included
    glow: np.ndarray  # (h, w) float32 in [0, 1], opacity included
    centre: tuple[float, float]  # the mask's centroid in sprite pixels
    radius_px: float
    source_frame: int
    source_angle: float | None = None  # the object's rotation angle at the source frame

    @property
    def size(self) -> tuple[int, int]:
        return self.alpha.shape[1], self.alpha.shape[0]

    def to_rgba(self) -> np.ndarray:
        """A BGRA image of the sprite on its own, for sprite.png."""
        a = np.clip(np.maximum(self.alpha, self.glow), 0, 1)
        bgr = np.where(
            a[..., None] > 0,
            (self.color * self.alpha[..., None] + CYAN_BGR * self.glow[..., None])
            / np.maximum(self.alpha[..., None] + self.glow[..., None], 1e-6),
            0,
        )
        return np.dstack([np.clip(bgr, 0, 255), a * 255]).astype(np.uint8)


def _speed(track: MaskedTrack) -> np.ndarray:
    """Pixels the centroid moves per frame, from central differences."""
    frame = track.frame.astype(float)
    if len(frame) < 3:
        return np.zeros(len(frame))
    dx = np.gradient(track.cx.astype(float), frame)
    dy = np.gradient(track.cy.astype(float), frame)
    return np.hypot(dx, dy)


def rank_frames(track: MaskedTrack, frames: Sequence[int] | None = None) -> list[int]:
    """Candidate frames for the sprite, best first: high confidence, the object whole (not at
    the border, an area near the median) and slow, so the least motion blur."""
    idx = np.arange(len(track.frame))
    if frames is not None:
        idx = idx[np.isin(track.frame, list(frames))]
    if len(idx) == 0:
        return []
    area = track.area.astype(float)
    median_area = float(np.nanmedian(area[idx]))
    conf = np.where(np.isfinite(track.conf), track.conf, 0.5).astype(float)
    speed = _speed(track)
    whole = (~np.asarray(track.touches_border, bool)) & (
        np.abs(area / max(median_area, 1e-9) - 1.0) < 0.2
    )
    usable = idx[whole[idx]]
    pool = usable if len(usable) else idx
    top = pool[np.argsort(-np.round(conf[pool], 2), kind="stable")][: max(CANDIDATES * 3, 1)]
    best = top[np.lexsort((speed[top], -np.round(conf[top], 2)))][:CANDIDATES]
    return [int(track.frame[i]) for i in best]


def sharpness(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian over the mask's bounding box: higher means a crisper object."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def pick_frame(
    track: MaskedTrack, images: dict[int, np.ndarray], ranked: Sequence[int]
) -> int | None:
    """The sharpest of the best few ranked frames whose image is available."""
    scored = []
    for frame in list(ranked)[:SHARPNESS_POOL]:
        image = images.get(frame)
        if image is not None:
            scored.append((sharpness(image, track.mask(frame)), frame))
    return max(scored)[1] if scored else None


def build_sprite(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    source_frame: int,
    source_angle: float | None = None,
) -> Sprite | None:
    """Cut the object out of `image_bgr` (full frame) with its full-frame 0/1 `mask`."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 4:
        return None
    h, w = mask.shape
    y0, y1 = max(ys.min() - MARGIN_PX, 0), min(ys.max() + 1 + MARGIN_PX, h)
    x0, x1 = max(xs.min() - MARGIN_PX, 0), min(xs.max() + 1 + MARGIN_PX, w)
    crop = image_bgr[y0:y1, x0:x1].astype(np.float32)
    m = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    feathered = cv2.GaussianBlur(m, (0, 0), FEATHER_SIGMA_PX)
    color = (1 - CYAN_MIX) * crop + CYAN_MIX * CYAN_BGR
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GLOW_GROW_PX + 1,) * 2)
    glow = cv2.GaussianBlur(cv2.dilate(m, kernel), (0, 0), GLOW_SIGMA_PX)
    moments = cv2.moments(m, binaryImage=True)
    cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return Sprite(
        color=color,
        alpha=np.clip(feathered, 0, 1) * OPACITY,
        glow=np.clip(glow, 0, 1) * GLOW_OPACITY,
        centre=(float(cx), float(cy)),
        radius_px=math.sqrt(moments["m00"] / math.pi),
        source_frame=source_frame,
        source_angle=source_angle,
    )


def disc_sprite(radius_px: float, color_bgr: tuple[float, float, float], source_frame: int = -1):
    """A plain disc, for when no mask is available (no SAM events were kept)."""
    r = max(int(round(radius_px)), 3)
    size = 2 * (r + MARGIN_PX) + 1
    image = np.full((size, size, 3), color_bgr, np.uint8)
    yy, xx = np.mgrid[:size, :size]
    mask = (np.hypot(yy - size // 2, xx - size // 2) <= r).astype(np.uint8)
    return build_sprite(image, mask, source_frame=source_frame)


def _patch(sprite: Sprite, centre: tuple[float, float], angle_deg: float):
    """Warp the sprite's layers so its centroid lands at `centre` (frame pixels): returns the
    layers and the integer frame position of the patch's top-left corner."""
    w, h = sprite.size
    diag = int(math.ceil(math.hypot(w, h))) + 4
    ox, oy = math.floor(centre[0] - diag / 2), math.floor(centre[1] - diag / 2)
    m = cv2.getRotationMatrix2D(sprite.centre, angle_deg, 1.0)
    m[0, 2] += centre[0] - ox - sprite.centre[0]
    m[1, 2] += centre[1] - oy - sprite.centre[1]
    kw = {"flags": cv2.INTER_LINEAR, "borderMode": cv2.BORDER_CONSTANT, "borderValue": 0}
    color = cv2.warpAffine(sprite.color, m, (diag, diag), **kw)
    alpha = cv2.warpAffine(sprite.alpha, m, (diag, diag), **kw)
    glow = cv2.warpAffine(sprite.glow, m, (diag, diag), **kw)
    return color, alpha, glow, ox, oy


def draw_sprite(
    frame: np.ndarray,
    sprite: Sprite,
    centre: tuple[float, float],
    angle_deg: float = 0.0,
    outline: str | None = None,
    outline_px: int = 2,
) -> None:
    """Composite the ghost onto `frame` (BGR uint8) in place: glow, then the tinted object, then
    a thin outline coloured by how well the ghost agrees with reality."""
    color, alpha, glow, ox, oy = _patch(sprite, centre, angle_deg)
    fh, fw = frame.shape[:2]
    ph, pw = alpha.shape
    x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + pw, fw), min(oy + ph, fh)
    if x1 <= x0 or y1 <= y0:
        return
    sl_f = (slice(y0, y1), slice(x0, x1))
    sl_p = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
    roi = frame[sl_f].astype(np.float32)
    g = glow[sl_p][..., None]
    roi = roi * (1 - g) + CYAN_BGR * g
    a = alpha[sl_p][..., None]
    roi = roi * (1 - a) + color[sl_p] * a
    frame[sl_f] = np.clip(roi, 0, 255).astype(np.uint8)
    if outline:
        edge = (alpha[sl_p] > 0.5 * OPACITY).astype(np.uint8)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.polylines(frame[sl_f], contours, True, OUTLINE_BGR[outline], outline_px, cv2.LINE_AA)
