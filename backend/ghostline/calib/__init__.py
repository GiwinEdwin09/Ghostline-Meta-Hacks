"""Sheet corners, homography, plane-offset correction and gravity (build plan Part 8).

Main entry point: `calibrate(image_size, sheet_mask=..., plane_offset_m=...)`, which returns a
`Calibration` with `px_to_world` / `world_to_px` for the trajectory and ghost code.
"""

from .gravity import Gravity, from_camera_level, from_free_fall, from_plumb, plumb_from_oscillation
from .homography import PAPER_MM, apply_h, compute_homography
from .model import Calibration
from .pipeline import calibrate, write_calibration
from .sheet import SheetDetection, detect_sheet_corners

__all__ = [
    "PAPER_MM",
    "Calibration",
    "Gravity",
    "SheetDetection",
    "apply_h",
    "calibrate",
    "compute_homography",
    "detect_sheet_corners",
    "from_camera_level",
    "from_free_fall",
    "from_plumb",
    "plumb_from_oscillation",
    "write_calibration",
]
