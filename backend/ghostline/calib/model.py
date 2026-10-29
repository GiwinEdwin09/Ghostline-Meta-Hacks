"""The calibration result (calibration.json) and the pixel <-> world mapping built on it.

World frame: metres, origin at the sheet's centre, x to the right and y up, with 'up' opposite
to the gravity direction. Positions are first corrected from the sheet plane to the plane of
motion: world = c + (H(p) - c) * (1 - d/Z).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ghostline.ingest.models import Notice

from . import offset as offset_math
from .gravity import Gravity, physics_axes
from .homography import apply_h


class AspectCheck(BaseModel):
    method: Literal["rectified_mask", "corners_only"]
    rectified_aspect: float | None = None
    paper_aspect: float
    error_rel: float | None = None
    passed: bool = True
    pose_ratio: float | None = None  # real x/y proportions over the assumed ones; about 1 if right


class OffsetInfo(BaseModel):
    d_m: float
    f35_mm: float
    f35_source: Literal["default", "metadata", "declared"]
    f_px: float
    z_m: float
    scale: float  # 1 - d/Z
    method: Literal["pose", "long_side"] = "pose"  # how Z and the point under the camera came


class GravityInfo(BaseModel):
    source: Literal["data", "plumb", "camera_level"]
    down_sheet: tuple[float, float]  # unit vector in the sheet frame (mm, y down)
    angle_deg: float  # of 'down' from the sheet's +y axis
    sigma_deg: float


class RampInfo(BaseModel):
    angle_deg: float
    incline_deg: float
    sigma_deg: float  # line fit, Monte Carlo and gravity direction combined
    path_check_deg: float | None = None


class MotionInfo(BaseModel):
    max_shift_px: float
    status: Literal["ok", "warn", "stop"]
    shifts_px: list[float]


class UncertaintyInfo(BaseModel):
    sigma_scale_rel: float
    components: dict[str, float]
    n_draws: int = 500


class Calibration(BaseModel):
    image_size: tuple[int, int]  # width, height
    paper: Literal["A4", "Letter"]
    paper_mm: tuple[float, float]  # sheet-frame width and height
    paper_source: Literal["declared", "auto"]
    paper_auto_confident: bool | None = None
    corners_px: list[list[float]]  # clockwise from top-left
    corners_mm: list[list[float]]
    corner_method: str
    sigma_corner_px: float
    H: list[list[float]]  # image pixels -> sheet millimetres
    H_inv: list[list[float]]
    centre_px: tuple[float, float]
    centre_mm: tuple[float, float]  # sheet point under the camera, c in the offset formula
    mm_per_px_center: float
    aspect_check: AspectCheck
    offset: OffsetInfo
    gravity: GravityInfo
    ramp: RampInfo | None = None
    camera_motion: MotionInfo | None = None
    uncertainty: UncertaintyInfo
    timing: dict | None = None
    notices: list[Notice] = Field(default_factory=list)

    @property
    def H_np(self) -> np.ndarray:
        return np.asarray(self.H)

    @property
    def H_inv_np(self) -> np.ndarray:
        return np.asarray(self.H_inv)

    @property
    def gravity_down(self) -> np.ndarray:
        return np.asarray(self.gravity.down_sheet)

    def with_gravity(self, g: Gravity) -> Calibration:
        info = GravityInfo(
            source=g.source,
            down_sheet=(float(g.down[0]), float(g.down[1])),
            angle_deg=g.angle_deg,
            sigma_deg=g.sigma_deg,
        )
        return self.model_copy(update={"gravity": info})

    def _origin_mm(self) -> np.ndarray:
        return np.asarray(self.paper_mm) / 2

    def px_to_plane_mm(self, px: np.ndarray, plane_of_motion: bool = True) -> np.ndarray:
        """Pixels (N, 2) -> sheet-frame millimetres, corrected to the plane of motion."""
        sheet = apply_h(self.H_np, px)
        if plane_of_motion:
            sheet = offset_math.to_plane_of_motion(
                sheet, np.asarray(self.centre_mm), self.offset.scale
            )
        return sheet

    def plane_mm_to_world(self, sheet_mm: np.ndarray, down: np.ndarray | None = None) -> np.ndarray:
        """Sheet-frame millimetres -> metres in the physics frame (origin at the sheet centre)."""
        right, up = physics_axes(self.gravity_down if down is None else down)
        rel = (np.asarray(sheet_mm) - self._origin_mm()) / 1000.0
        return np.column_stack([rel @ right, rel @ up])

    def px_to_world(
        self, px: np.ndarray, down: np.ndarray | None = None, plane_of_motion: bool = True
    ) -> np.ndarray:
        """Pixels (N, 2) -> metres in the physics frame (N, 2)."""
        return self.plane_mm_to_world(self.px_to_plane_mm(px, plane_of_motion), down)

    def world_to_px(
        self, xy_m: np.ndarray, down: np.ndarray | None = None, plane_of_motion: bool = True
    ) -> np.ndarray:
        """Inverse of `px_to_world`: metres in the physics frame -> pixels."""
        right, up = physics_axes(self.gravity_down if down is None else down)
        xy = np.atleast_2d(np.asarray(xy_m, float))
        sheet = self._origin_mm() + 1000.0 * (np.outer(xy[:, 0], right) + np.outer(xy[:, 1], up))
        if plane_of_motion:
            sheet = offset_math.from_plane_of_motion(
                sheet, np.asarray(self.centre_mm), self.offset.scale
            )
        return apply_h(self.H_inv_np, sheet)

    def scale_sigma_at(self, points_px: np.ndarray, n: int = 300, seed: int = 0) -> float:
        """Relative scale uncertainty (Monte Carlo, as in calibrate) at the given image points."""
        from .uncertainty import monte_carlo

        return monte_carlo(
            np.asarray(self.corners_px), np.asarray(self.corners_mm), self.sigma_corner_px,
            np.asarray(self.centre_px), self.offset.d_m, self.offset.z_m, n=n, seed=seed,
            points_px=points_px,
        ).sigma_scale_rel  # fmt: skip

    @property
    def m_per_px(self) -> float:
        """Metres per pixel at the image centre, in the plane of motion."""
        return self.mm_per_px_center * self.offset.scale / 1000.0
