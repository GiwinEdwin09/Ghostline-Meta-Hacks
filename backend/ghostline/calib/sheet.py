"""Sub-pixel sheet corners from a segmentation mask."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from ghostline.ingest.models import Notice

from .homography import order_corners

EDGE_TRIM = 0.12  # ignore this fraction of each side next to a corner (rounded or curled corners)
RANSAC_ITERS = 150
INLIER_PX = 0.75
MIN_SIGMA_PX = 0.3  # corner uncertainty floor from a binary mask
MIN_SIGMA_GRAY_PX = 0.1  # ... after the grey-level edge refinement (synthetic errors are under 0.2)
SUBPIX_MAX_MOVE = 1.0  # px a refined side may shift
SUBPIX_MAX_TURN = 2.0  # px its ends may move (a binary mask can be a pixel off over a side)
PROFILE_SEGMENTS = 12
PROFILE_HALF_WIDTH = 5.0
MIN_CONTRAST = 20.0  # grey levels between paper and background for the edge refinement


@dataclass
class SheetDetection:
    corners_px: np.ndarray  # (4, 2) clockwise from top-left
    sigma_corner_px: float
    edge_rms_px: float
    n_edge_points: list[int]
    method: str  # "ransac_lines" or "min_area_rect"
    subpix: bool = False
    notices: list[Notice] = field(default_factory=list)


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(filled boolean mask of the biggest external contour, the contour)."""
    contours, _ = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("the sheet mask is empty")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape[:2], np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, -1)
    return filled.astype(bool), contour


def mask_edge_points(filled: np.ndarray) -> np.ndarray:
    """Midpoints between adjacent foreground and background pixel centres.

    Unlike the boundary pixels themselves, these are unbiased estimates of the true edge.
    """
    ys, xs = np.nonzero(filled[:, :-1] != filled[:, 1:])
    horizontal = np.column_stack([xs + 0.5, ys])
    ys, xs = np.nonzero(filled[:-1] != filled[1:])
    vertical = np.column_stack([xs, ys + 0.5])
    return np.vstack([horizontal, vertical]).astype(float)


def _initial_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))
    return None


def _fit_line_ransac(pts: np.ndarray, rng: np.random.Generator):
    """RANSAC then total-least-squares line fit. Returns (point, unit direction, rms, n_inliers)."""
    best = None
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            continue
        d /= norm
        resid = np.abs((pts - pts[i]) @ np.array([-d[1], d[0]]))
        count = int(np.sum(resid < INLIER_PX))
        if best is None or count > best[0]:
            best = (count, pts[i], d, resid < INLIER_PX)
    inliers = pts[best[3]]
    centre = inliers.mean(axis=0)
    direction = np.linalg.svd(inliers - centre)[2][0]
    resid = (inliers - centre) @ np.array([-direction[1], direction[0]])
    return centre, direction, float(np.sqrt(np.mean(resid**2))), len(inliers)


def _intersect(p1, d1, p2, d2) -> np.ndarray:
    a = np.array([d1, -d2]).T
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def detect_sheet_corners(
    mask: np.ndarray, gray: np.ndarray | None = None, seed: int = 0
) -> SheetDetection:
    """Corners of the sheet in a binary mask, to sub-pixel accuracy.

    Edge points of each of the four sides are fitted with a RANSAC line, so rounded or curled
    corners don't bias the result, and adjacent lines are intersected. If `gray` (the median
    image) is given, each side is re-fitted to the sub-pixel grey-level edge where that moves it
    less than 1 px (more accurate than cornerSubPix on anti-aliased edges).
    Falls back to the minimum-area rectangle when a side can't be fitted.
    """
    filled, contour = largest_component(mask)
    quad = _initial_quad(contour)
    notices: list[Notice] = []
    if quad is None:
        return min_area_rect_detection(contour, notices)
    points = mask_edge_points(filled)
    rng = np.random.default_rng(seed)
    lines, rms, counts = [], [], []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        length = np.linalg.norm(b - a)
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        s = (points - a) @ u / length
        dist = np.abs((points - a) @ n)
        keep = (s > EDGE_TRIM) & (s < 1 - EDGE_TRIM) & (dist < max(3.0, 0.05 * length))
        if keep.sum() < 8:
            return min_area_rect_detection(contour, notices)
        centre, direction, r, c = _fit_line_ransac(points[keep], rng)
        lines.append((centre, direction))
        rms.append(r)
        counts.append(c)
    lengths = [float(np.linalg.norm(quad[(k + 1) % 4] - quad[k])) for k in range(4)]
    subpix = False
    if gray is not None:
        lines, subpix = _refine_with_gray(gray, lines, lengths, quad.mean(axis=0))
    try:
        corners = np.array([_intersect(*lines[k - 1], *lines[k]) for k in range(4)])
    except np.linalg.LinAlgError:
        return min_area_rect_detection(contour, notices)
    # corners[k] is the intersection of side k-1 and side k, i.e. quad[k]
    if np.max(np.linalg.norm(corners - quad, axis=1)) > 0.2 * np.linalg.norm(quad[1] - quad[0]):
        return min_area_rect_detection(contour, notices)
    corners = order_corners(corners)
    edge_rms = float(np.mean(rms))
    sigma = MIN_SIGMA_GRAY_PX if subpix else max(MIN_SIGMA_PX, edge_rms)
    return SheetDetection(corners, sigma, edge_rms, counts, "ransac_lines", subpix, notices)


def _refine_line_with_gray(
    gray: np.ndarray, centre: np.ndarray, direction: np.ndarray, outward: np.ndarray, length: float
):
    """Re-fit one side from the grey-level edge: the half-way crossing of the paper/background
    step, found along the normal in a few segments of the side. Returns None if it can't."""
    points = []
    for seg in np.linspace(EDGE_TRIM + 0.04, 1 - EDGE_TRIM - 0.04, PROFILE_SEGMENTS):
        ts = (seg + np.linspace(-0.025, 0.025, 7)) * length
        origin = centre + (ts[:, None] - 0.5 * length) * direction  # samples along the side
        s = np.arange(-PROFILE_HALF_WIDTH, PROFILE_HALF_WIDTH + 1e-9, 0.25)
        coords = origin[:, None, :] + s[None, :, None] * outward  # (n_t, n_s, 2) as x, y
        vals = ndimage.map_coordinates(
            gray.astype(np.float32), [coords[..., 1].ravel(), coords[..., 0].ravel()],
            order=1, mode="nearest",
        ).reshape(len(ts), len(s)).mean(axis=0)  # fmt: skip
        inside, outside = np.median(vals[s < -3]), np.median(vals[s > 3])
        if abs(inside - outside) < MIN_CONTRAST:
            return None
        half = (inside + outside) / 2
        sign = np.sign(outside - inside)
        crossing = np.nonzero(((vals[:-1] - half) * sign < 0) & ((vals[1:] - half) * sign >= 0))[0]
        crossing = crossing[crossing >= np.searchsorted(s, -3)]
        if not len(crossing):
            return None
        i = crossing[np.argmin(np.abs(s[crossing]))]
        frac = (half - vals[i]) / (vals[i + 1] - vals[i])
        offset = s[i] + frac * 0.25
        points.append(centre + (seg - 0.5) * length * direction + offset * outward)
    pts = np.array(points)
    c = pts.mean(axis=0)
    d = np.linalg.svd(pts - c)[2][0]
    return c, d


def _refine_with_gray(gray, lines, lengths, centroid):
    """Replace each fitted line by its grey-level refinement where that moves it < 1 px."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    out, used = [], False
    for (centre, direction), length in zip(lines, lengths, strict=True):
        normal = np.array([-direction[1], direction[0]])
        outward = normal if (centre - centroid) @ normal > 0 else -normal
        refined = _refine_line_with_gray(g, centre, direction, outward, length)
        if refined is not None:
            moved = abs((refined[0] - centre) @ normal)
            turned = abs(direction[0] * refined[1][1] - direction[1] * refined[1][0]) * length
            if moved < SUBPIX_MAX_MOVE and turned < SUBPIX_MAX_TURN:
                out.append(refined)
                used = True
                continue
        out.append((centre, direction))
    return out, used


def min_area_rect_detection(contour: np.ndarray, notices: list[Notice]) -> SheetDetection:
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    notices.append(
        Notice(
            code="sheet_min_area_rect",
            message="The sheet's edges could not be fitted as four straight sides, so its "
            "bounding rectangle was used. Scale accuracy is reduced.",
        )
    )
    return SheetDetection(order_corners(box), 1.0, float("nan"), [], "min_area_rect", False, notices)
