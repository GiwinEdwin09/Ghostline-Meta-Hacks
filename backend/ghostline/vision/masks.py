"""Mask decoding and mask features (build plan Part 7).

Coordinates are source-frame pixels with the OpenCV convention: the centre of the top-left
pixel is (0, 0), so a pixel at array position (row, col) has its centre at x = col, y = row.
A mask raster from SAM is box-sized and anchored at the box's top-left corner (checked in the
Day 0 smoke test), so a feature computed on the raster is offset by the box origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from meta_sam_parser import SegmentationMask, decode_mask_to_raster


@dataclass(frozen=True)
class MaskFeatures:
    area: float  # pixels
    cx: float  # sub-pixel centroid, m10/m00
    cy: float  # sub-pixel centroid, m01/m00
    theta: float  # major-axis orientation, radians in (-pi/2, pi/2], image y pointing down
    eccentricity: float  # 0 for a disc, towards 1 for a line
    bbox: tuple[int, int, int, int]  # set pixels' tight box: left, top, right, bottom (half-open)
    touches_border: bool  # the mask reaches the frame edge, so the object may be cut off


def decode_mask(mask: SegmentationMask) -> np.ndarray:
    """Decode a SAM mask payload into a (height, width) uint8 array of 0 and 1."""
    flat = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8)
    return flat.reshape(mask.height, mask.width)


def place_mask(
    raster: np.ndarray, left: float, top: float, frame_size: tuple[int, int]
) -> np.ndarray:
    """Paste a box-sized raster into a full-frame array, its top-left at the box origin."""
    width, height = frame_size
    left, top = round(left), round(top)
    frame = np.zeros((height, width), dtype=np.uint8)
    h, w = raster.shape
    frame[top : top + h, left : left + w] = raster[: height - top, : width - left]
    return frame


def mask_features(
    raster: np.ndarray, left: float, top: float, frame_size: tuple[int, int]
) -> MaskFeatures | None:
    """Features of a box-sized raster whose top-left corner sits at (left, top). None if empty."""
    moments = cv2.moments(raster.astype(np.uint8), binaryImage=True)
    area = moments["m00"]
    if area <= 0:
        return None
    cx, cy = moments["m10"] / area, moments["m01"] / area
    mu20, mu02, mu11 = (moments[k] / area for k in ("mu20", "mu02", "mu11"))
    theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
    spread = math.sqrt(max((mu20 - mu02) ** 2 / 4 + mu11**2, 0.0))
    major, minor = (mu20 + mu02) / 2 + spread, (mu20 + mu02) / 2 - spread
    eccentricity = math.sqrt(max(1 - minor / major, 0.0)) if major > 0 else 0.0
    rows, cols = np.nonzero(raster)
    x0, x1 = int(cols.min() + round(left)), int(cols.max() + round(left)) + 1
    y0, y1 = int(rows.min() + round(top)), int(rows.max() + round(top)) + 1
    width, height = frame_size
    return MaskFeatures(
        area=float(area),
        cx=float(cx + left),
        cy=float(cy + top),
        theta=float(theta),
        eccentricity=float(eccentricity),
        bbox=(x0, y0, x1, y1),
        touches_border=x0 <= 0 or y0 <= 0 or x1 >= width or y1 >= height,
    )
