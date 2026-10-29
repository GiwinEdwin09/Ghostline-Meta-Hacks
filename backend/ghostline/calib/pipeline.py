"""calibrate(): sheet mask or corners -> homography, plane offset, gravity and uncertainty."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ghostline.ingest.models import Notice

from . import offset as offset_math
from .gravity import from_camera_level
from .homography import (
    PAPER_MM,
    apply_h,
    compute_homography,
    corners_mm,
    guess_paper,
    local_scale,
    order_corners,
)
from .model import (
    AspectCheck,
    Calibration,
    GravityInfo,
    MotionInfo,
    OffsetInfo,
    RampInfo,
    UncertaintyInfo,
)
from .motion import camera_motion
from .overlay import draw_overlay, rectified_image
from .ramp import ramp_angle_from_mask, top_edge_points
from .sheet import (
    MIN_SIGMA_PX,
    SheetDetection,
    detect_sheet_corners,
    largest_component,
    mask_edge_points,
    min_area_rect_detection,
)
from .uncertainty import N_DRAWS, monte_carlo

ASPECT_TOLERANCE = 0.02
POSE_RATIO_TOLERANCE = 0.05
TIMING_KEYS = ("classification", "time_factor", "time_factor_source", "jitter_rel", "real_fps",
               "sigma_time_rel", "retimed", "needs_confirmation")  # fmt: skip


def _aspect_check(edge_px: np.ndarray, h: np.ndarray, paper_mm: tuple[float, float]) -> AspectCheck:
    mm = apply_h(h, edge_px).astype(np.float32)
    (_, _), (w, hh), _ = cv2.minAreaRect(mm)
    aspect = max(w, hh) / max(min(w, hh), 1e-9)
    paper_aspect = max(paper_mm) / min(paper_mm)
    error = abs(aspect / paper_aspect - 1)
    return AspectCheck(
        method="rectified_mask", rectified_aspect=float(aspect), paper_aspect=float(paper_aspect),
        error_rel=float(error), passed=bool(error <= ASPECT_TOLERANCE),
    )  # fmt: skip


def calibrate(
    image_size: tuple[int, int],
    *,
    sheet_mask: np.ndarray | None = None,
    corners_px: np.ndarray | None = None,
    gray: np.ndarray | None = None,
    paper: str = "auto",
    plane_offset_m: float = 0.02,
    f35_mm: float | None = None,
    ramp_mask: np.ndarray | None = None,
    corner_sets: np.ndarray | None = None,
    timing: dict | None = None,
    seed: int = 0,
) -> Calibration:
    """Calibrate from the sheet mask (or four corners) of the clip's median image.

    `image_size` is (width, height). `plane_offset_m` is how far in front of the sheet the object
    moves. `corner_sets` holds the sheet corners found in the first, middle and last frames, for
    the camera-motion check. `timing` is the dumped Part 5 TimingModel.
    """
    notices: list[Notice] = []
    edge_px = None
    if sheet_mask is not None:
        det = detect_sheet_corners(sheet_mask, gray, seed)
        filled, contour = largest_component(sheet_mask)
        edge_px = mask_edge_points(filled)
    elif corners_px is not None:
        det = SheetDetection(order_corners(corners_px), MIN_SIGMA_PX, float("nan"), [], "given")
    else:
        raise ValueError("calibrate needs a sheet mask or four corners")
    notices += det.notices

    if paper == "auto":
        paper_name, _, confident = guess_paper(det.corners_px)
        source = "auto"
    else:
        paper_name, confident, source = paper, None, "declared"
    if confident is False:
        notices.append(
            Notice(
                code="paper_size_unconfirmed",
                message=f"Paper size guessed as {paper_name} from the sheet's shape, which is "
                "not conclusive at this camera angle. Please confirm A4 or Letter.",
            )
        )
    mm = corners_mm(det.corners_px, paper_name)
    h = compute_homography(det.corners_px, mm)
    paper_mm = (float(mm[2, 0]), float(mm[2, 1]))

    if edge_px is not None:
        check = _aspect_check(edge_px, h, paper_mm)
        if not check.passed and det.method != "min_area_rect":
            notices.append(
                Notice(
                    code="sheet_aspect_mismatch",
                    message=f"The sheet's rectified aspect is {check.error_rel:.1%} off "
                    f"{paper_name}; its bounding rectangle was used instead.",
                )
            )
            det = min_area_rect_detection(contour, notices)
            mm = corners_mm(det.corners_px, paper_name)
            h = compute_homography(det.corners_px, mm)
            paper_mm = (float(mm[2, 0]), float(mm[2, 1]))
            check = _aspect_check(edge_px, h, paper_mm)
    else:
        paper_aspect = max(paper_mm) / min(paper_mm)
        check = AspectCheck(method="corners_only", paper_aspect=float(paper_aspect))

    w, hgt = image_size
    ratio = offset_math.pose_aspect_ratio(h, offset_math.focal_px(w, f35_mm), image_size)
    check = check.model_copy(update={"pose_ratio": ratio})
    if ratio is not None and abs(ratio - 1) > POSE_RATIO_TOLERANCE:
        notices.append(
            Notice(
                code="paper_size_mismatch",
                message=f"The sheet's proportions don't look like {paper_name} (they are off by "
                f"{abs(ratio - 1):.0%}). Check that the paper size is right.",
            )
        )
    centre_px = np.array([(w - 1) / 2, (hgt - 1) / 2])
    f_px = offset_math.focal_px(w, f35_mm)
    pose = offset_math.pose_from_homography(h, f_px, image_size)
    if pose is not None:
        z_m, centre_mm = pose
        z_method = "pose"
    else:
        z_m = offset_math.camera_distance_m(
            f_px, offset_math.long_side_px(det.corners_px), paper_name
        )
        centre_mm = apply_h(h, centre_px)[0]
        z_method = "long_side"
    scale = offset_math.offset_scale(plane_offset_m, z_m)
    if plane_offset_m / z_m > 0.1:
        notices.append(
            Notice(
                code="large_plane_offset",
                message="The object moves far in front of the sheet compared with the camera "
                "distance, so the plane-offset correction is large. Stand further back.",
            )
        )

    gravity = from_camera_level(h, image_size)
    ramp_info = None
    ramp_edge = None
    if ramp_mask is not None:
        ramp_edge = top_edge_points(ramp_mask)
        angle = ramp_angle_from_mask(ramp_mask, h, gravity.down)
    mc = monte_carlo(
        det.corners_px, mm, det.sigma_corner_px, centre_px, plane_offset_m, z_m,
        ramp_edge, gravity.down, N_DRAWS, seed,
    )  # fmt: skip
    if ramp_mask is not None:
        sigma = float(
            np.sqrt(angle.sigma_deg**2 + (mc.sigma_ramp_deg or 0) ** 2 + gravity.sigma_deg**2)
        )
        ramp_info = RampInfo(
            angle_deg=angle.angle_deg, incline_deg=angle.incline_deg, sigma_deg=sigma
        )

    motion_info = None
    if corner_sets is not None:
        cm = camera_motion(corner_sets)
        motion_info = MotionInfo(
            max_shift_px=cm.max_shift_px, status=cm.status, shifts_px=cm.shifts_px
        )
        if cm.status == "stop":
            notices.append(
                Notice(
                    code="camera_moved",
                    severity="error",
                    message=f"The camera moved {cm.max_shift_px:.1f} px during the clip. Please "
                    "film again with the phone on a tripod.",
                )
            )
        elif cm.status == "warn":
            notices.append(
                Notice(
                    code="camera_moved",
                    message=f"The camera moved {cm.max_shift_px:.1f} px during the clip; "
                    "scale uncertainty is increased. A tripod avoids this.",
                )
            )

    return Calibration(
        image_size=(w, hgt),
        paper=paper_name,
        paper_mm=paper_mm,
        paper_source=source,
        paper_auto_confident=confident,
        corners_px=det.corners_px.tolist(),
        corners_mm=mm.tolist(),
        corner_method=det.method,
        sigma_corner_px=float(det.sigma_corner_px),
        H=h.tolist(),
        H_inv=np.linalg.inv(h).tolist(),
        centre_px=(float(centre_px[0]), float(centre_px[1])),
        centre_mm=(float(centre_mm[0]), float(centre_mm[1])),
        mm_per_px_center=local_scale(h, centre_px),
        aspect_check=check,
        offset=OffsetInfo(
            d_m=plane_offset_m,
            f35_mm=f35_mm or offset_math.DEFAULT_F35_MM,
            f35_source="metadata" if f35_mm else "default",
            f_px=f_px,
            z_m=z_m,
            scale=scale,
            method=z_method,
        ),  # fmt: skip
        gravity=GravityInfo(
            source=gravity.source,
            down_sheet=(float(gravity.down[0]), float(gravity.down[1])),
            angle_deg=gravity.angle_deg,
            sigma_deg=gravity.sigma_deg,
        ),  # fmt: skip
        ramp=ramp_info,
        camera_motion=motion_info,
        uncertainty=UncertaintyInfo(
            sigma_scale_rel=mc.sigma_scale_rel, components=mc.components, n_draws=N_DRAWS
        ),
        timing={k: timing[k] for k in TIMING_KEYS if k in timing} if timing else None,
        notices=notices,
    )


def write_calibration(
    calib: Calibration, out_dir: str | Path, image_bgr: np.ndarray | None = None
) -> None:
    """Write calibration.json and, given the median image, calib_overlay.png and rectified.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(calib.model_dump_json(indent=2))
    if image_bgr is not None:
        cv2.imwrite(str(out / "calib_overlay.png"), draw_overlay(image_bgr, calib))
        cv2.imwrite(str(out / "rectified.png"), rectified_image(image_bgr, calib))


__all__ = ["PAPER_MM", "calibrate", "write_calibration"]
