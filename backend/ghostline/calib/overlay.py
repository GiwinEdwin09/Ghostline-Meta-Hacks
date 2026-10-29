"""calib_overlay.png (sheet outline and a 5 cm grid) and rectified.png."""

from __future__ import annotations

import cv2
import numpy as np

from .homography import apply_h
from .model import Calibration

GREEN = (80, 220, 80)
CYAN = (230, 200, 40)


def draw_overlay(image_bgr: np.ndarray, calib: Calibration, grid_mm: float = 50.0) -> np.ndarray:
    """The sheet outline plus a grid drawn through H^-1, so the user can see the fit."""
    img = image_bgr.copy()
    h, w = img.shape[:2]
    h_inv = calib.H_inv_np
    frame_mm = apply_h(calib.H_np, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    lo = np.floor(frame_mm.min(axis=0) / grid_mm) * grid_mm
    hi = np.ceil(frame_mm.max(axis=0) / grid_mm) * grid_mm
    lo, hi = np.maximum(lo, -3000), np.minimum(hi, 3000)
    layer = img.copy()
    for x in np.arange(lo[0], hi[0] + 1, grid_mm):
        _line(layer, h_inv, (x, lo[1]), (x, hi[1]), CYAN, w, h)
    for y in np.arange(lo[1], hi[1] + 1, grid_mm):
        _line(layer, h_inv, (lo[0], y), (hi[0], y), CYAN, w, h)
    img = cv2.addWeighted(layer, 0.55, img, 0.45, 0)
    corners = np.asarray(calib.corners_px)
    cv2.polylines(img, [np.round(corners).astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
    for label, c in zip(("TL", "TR", "BR", "BL"), corners, strict=True):
        cv2.circle(img, tuple(np.round(c).astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(img, label, tuple(np.round(c + [4, -4]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)  # fmt: skip
    text = f"{calib.paper}  grid {grid_mm / 10:g} cm  {calib.mm_per_px_center:.3f} mm/px"
    cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _line(img, h_inv, a_mm, b_mm, color, w, h) -> None:
    a, b = apply_h(h_inv, np.array([a_mm, b_mm]))
    ok, p, q = cv2.clipLine(
        (0, 0, w, h), tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int))
    )
    if ok:
        cv2.line(img, p, q, color, 1, cv2.LINE_AA)


def rectified_image(
    image_bgr: np.ndarray, calib: Calibration, px_per_mm: float = 2.0, margin_mm: float = 60.0
) -> np.ndarray:
    """The sheet plane seen head-on, with a margin around the sheet."""
    pw, ph = calib.paper_mm
    size = (
        int(round((pw + 2 * margin_mm) * px_per_mm)),
        int(round((ph + 2 * margin_mm) * px_per_mm)),
    )
    to_rect = np.array(
        [[px_per_mm, 0, margin_mm * px_per_mm], [0, px_per_mm, margin_mm * px_per_mm], [0, 0, 1]]
    )
    return cv2.warpPerspective(image_bgr, to_rect @ calib.H_np, size, flags=cv2.INTER_LINEAR)
