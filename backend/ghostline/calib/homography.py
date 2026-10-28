"""Paper sizes, corner ordering and the pixel-to-millimetre homography.

The sheet frame is in millimetres with its origin at the top-left corner, x along the top edge
and y pointing down the sheet (like image coordinates, so the mapping never mirrors).
"""

from __future__ import annotations

import numpy as np

PAPER_MM: dict[str, tuple[float, float]] = {"A4": (210.0, 297.0), "Letter": (215.9, 279.4)}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points clockwise on screen, starting at the one nearest the top-left."""
    pts = np.asarray(pts, float).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])  # y down: clockwise
    ring = pts[np.argsort(angles)]
    start = int(np.argmin(ring.sum(axis=1)))
    return np.roll(ring, -start, axis=0)


def side_lengths(corners: np.ndarray) -> tuple[float, float]:
    """(mean top/bottom length, mean left/right length) of an ordered quad, in pixels."""
    c = np.asarray(corners, float)
    top = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    side = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    return float(top), float(side)


def image_aspect(corners: np.ndarray) -> float:
    a, b = side_lengths(corners)
    return max(a, b) / min(a, b)


def guess_paper(corners: np.ndarray) -> tuple[str, float, bool]:
    """Pick A4 or Letter from the quad's aspect. Returns (paper, aspect, confident).

    Only reliable when the camera is nearly head-on, so the confirm step can override it.
    """
    aspect = image_aspect(corners)
    a4, letter = (max(v) / min(v) for v in (PAPER_MM["A4"], PAPER_MM["Letter"]))
    paper = "A4" if abs(np.log(aspect / a4)) <= abs(np.log(aspect / letter)) else "Letter"
    confident = abs(np.log(aspect / np.sqrt(a4 * letter))) > 0.04
    return paper, float(aspect), bool(confident)


def corners_mm(corners_px: np.ndarray, paper: str) -> np.ndarray:
    """Sheet-frame corners for an ordered image quad: the longer pair of sides gets the long edge."""
    short, long = sorted(PAPER_MM[paper])
    top, side = side_lengths(corners_px)
    w, h = (long, short) if top >= side else (short, long)
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography taking four src points to four dst points (direct linear solve, float64).

    Same result as cv2.getPerspectiveTransform without its float32 rounding of the inputs.
    """
    a = []
    for (x, y), (u, v) in zip(np.asarray(src, float), np.asarray(dst, float), strict=True):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    h = np.linalg.svd(np.array(a))[2][-1].reshape(3, 3)
    return h / h[2, 2]


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    p = np.atleast_2d(np.asarray(pts, float))
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(h).T
    return q[:, :2] / q[:, 2:3]


def local_scale(h: np.ndarray, p: np.ndarray, eps: float = 0.5) -> float:
    """Millimetres per pixel at image point p: sqrt(|det J|) of the mapping, by central differences."""
    p = np.asarray(p, float)
    dx = (apply_h(h, p + [eps, 0]) - apply_h(h, p - [eps, 0]))[0] / (2 * eps)
    dy = (apply_h(h, p + [0, eps]) - apply_h(h, p - [0, eps]))[0] / (2 * eps)
    return float(np.sqrt(abs(dx[0] * dy[1] - dx[1] * dy[0])))
