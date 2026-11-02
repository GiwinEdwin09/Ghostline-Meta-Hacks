import cv2
import numpy as np
import pytest
from synth_helpers import scene

from ghostline.calib import (
    Calibration,
    calibrate,
    from_free_fall,
    from_plumb,
    plumb_from_oscillation,
    write_calibration,
)
from ghostline.calib.homography import local_scale, order_corners
from ghostline.calib.motion import camera_motion
from ghostline.calib.overlay import draw_overlay, rectified_image
from ghostline.calib.ramp import cross_check, ramp_angle_from_mask
from ghostline.calib.uncertainty import monte_carlo

SIZE = (854, 480)
CAMERAS = [
    dict(azimuth_deg=10, elevation_deg=4),
    dict(azimuth_deg=-25, elevation_deg=8, sheet_rot_deg=5),
    dict(azimuth_deg=35, elevation_deg=15, sheet_rot_deg=20, sheet_landscape=True),
    dict(azimuth_deg=15, elevation_deg=-6, roll_deg=3, sheet_rot_deg=12),
    dict(azimuth_deg=0, elevation_deg=0),
]


def calibrated(use_gray=True, paper="A4", **cam):
    sc = scene(system="free_fall", **cam)
    g = cv2.cvtColor(sc.plate_u8, cv2.COLOR_BGR2GRAY) if use_gray else None
    return sc, calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, gray=g, paper=paper)


def true_scale_mm_per_px(sc, z):
    """Millimetres per pixel at the image centre on the plane at height z, from the camera."""
    inv = np.linalg.inv(sc.cam.plane_homography(z))  # pixels -> metres on that plane
    return 1000 * local_scale(inv, np.array([(SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2]))


@pytest.mark.parametrize("cam", CAMERAS)
def test_scale_error_on_a_synthetic_perspective_render_is_under_0_2_percent(cam):
    sc, cal = calibrated(**cam)
    truth = true_scale_mm_per_px(sc, 0.02)  # the plane of motion, 2 cm in front of the sheet
    assert abs(cal.m_per_px * 1000 / truth - 1) < 0.002
    assert abs(cal.mm_per_px_center / true_scale_mm_per_px(sc, 0.0) - 1) < 0.002


def test_a_binary_mask_alone_is_good_to_about_a_percent():
    sc, cal = calibrated(use_gray=False, azimuth_deg=10, elevation_deg=4)
    assert cal.corner_method == "ransac_lines"
    assert abs(cal.m_per_px * 1000 / true_scale_mm_per_px(sc, 0.02) - 1) < 0.01


def test_calibration_fields_are_consistent():
    sc, cal = calibrated(azimuth_deg=10, elevation_deg=4)
    assert cal.paper == "A4" and cal.paper_source == "declared" and cal.paper_mm == (210.0, 297.0)
    assert cal.aspect_check.method == "rectified_mask" and cal.aspect_check.passed
    assert cal.aspect_check.error_rel < 0.02
    assert cal.offset.z_m == pytest.approx(abs(sc.cam.center[2]), rel=0.01)
    assert cal.offset.scale == pytest.approx(1 - 0.02 / cal.offset.z_m)
    assert cal.offset.f35_source == "default" and cal.offset.f_px == pytest.approx(854 * 26 / 36)
    assert np.allclose(np.asarray(cal.H) @ np.asarray(cal.H_inv), np.eye(3), atol=1e-9)
    assert not [n for n in cal.notices if n.severity == "error"]
    assert 0.0005 < cal.uncertainty.sigma_scale_rel < 0.05
    assert set(cal.uncertainty.components) == {"corners", "plane_offset_d", "distance_z"}


def test_plane_offset_changes_the_scale_by_d_over_z():
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    mask = sc.sheet_mask > 0
    near, far = (
        calibrate(SIZE, sheet_mask=mask, paper="A4", plane_offset_m=d) for d in (0.0, 0.05)
    )
    assert far.m_per_px / near.m_per_px == pytest.approx(1 - 0.05 / near.offset.z_m, rel=2e-3)


def test_paper_size_is_guessed_when_head_on_and_can_be_declared():
    a4 = scene(system="free_fall", azimuth_deg=0, elevation_deg=0)
    cal = calibrate(SIZE, sheet_mask=a4.sheet_mask > 0)
    assert cal.paper == "A4" and cal.paper_source == "auto" and cal.paper_auto_confident
    letter = scene(system="free_fall", azimuth_deg=0, elevation_deg=0, paper_m=(0.2159, 0.2794))
    cal = calibrate(SIZE, sheet_mask=letter.sheet_mask > 0)
    assert cal.paper == "Letter" and cal.paper_auto_confident
    assert cal.paper_mm == pytest.approx((215.9, 279.4))
    ambiguous = scene(system="free_fall", azimuth_deg=0, elevation_deg=0, paper_m=(0.2159, 0.2870))
    cal = calibrate(SIZE, sheet_mask=ambiguous.sheet_mask > 0)
    assert cal.paper_auto_confident is False
    assert "paper_size_unconfirmed" in {n.code for n in cal.notices}
    forced = calibrate(SIZE, sheet_mask=a4.sheet_mask > 0, paper="Letter")
    assert forced.paper == "Letter" and forced.paper_source == "declared"


def test_wrong_paper_size_is_noticed_from_the_sheets_proportions():
    sc = scene(system="free_fall", azimuth_deg=0, elevation_deg=0)
    right = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, paper="A4")
    wrong = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, paper="Letter")
    assert right.aspect_check.pose_ratio == pytest.approx(1.0, abs=0.025)
    assert "paper_size_mismatch" not in {n.code for n in right.notices}
    assert abs(wrong.aspect_check.pose_ratio - 1) > 0.07
    assert "paper_size_mismatch" in {n.code for n in wrong.notices}


def test_a_mask_that_is_not_a_rectangle_falls_back_to_the_bounding_rectangle():
    mask = np.zeros((480, 854), np.uint8)
    cv2.ellipse(mask, (400, 240), (150, 50), 0, 0, 360, 1, -1)  # a lozenge, not a sheet
    cal = calibrate(SIZE, sheet_mask=mask > 0, paper="A4")
    assert not cal.aspect_check.passed or cal.corner_method == "min_area_rect"
    assert cal.corner_method == "min_area_rect"
    assert {"sheet_aspect_mismatch", "sheet_min_area_rect"} & {n.code for n in cal.notices}


def test_camera_motion_thresholds():
    c = order_corners(scene(system="free_fall").corners_px)
    assert camera_motion([c, c + 0.3, c + 0.4]).status == "ok"
    assert camera_motion([c, c + [0.9, 0.9], c]).status == "warn"
    moved = camera_motion([c, c, c + [3.0, 2.0]])
    assert moved.status == "stop" and moved.max_shift_px == pytest.approx(np.hypot(3, 2))
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    cal = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, corner_sets=np.stack([c, c, c + 4.0]))
    assert cal.camera_motion.status == "stop"
    assert any(n.code == "camera_moved" and n.severity == "error" for n in cal.notices)


def test_one_covered_corner_is_not_camera_motion():
    c = order_corners(scene(system="free_fall").corners_px)
    late = c.copy()
    late[1] += [5.0, 3.0]  # a hand over one corner in the last probe frame
    assert camera_motion([c, c, late]).status == "ok"
    turned = c + (c - c.mean(axis=0)) * 0.06  # a zoom moves all four corners (about 4 px here)
    assert camera_motion([c, c, turned]).status == "stop"


def test_camera_level_gravity_follows_a_rolled_phone():
    _, level = calibrated(azimuth_deg=0, elevation_deg=0)
    _, rolled = calibrated(azimuth_deg=0, elevation_deg=0, roll_deg=6)
    assert abs(level.gravity.angle_deg) < 0.3 and level.gravity.source == "camera_level"
    assert abs(abs(rolled.gravity.angle_deg) - 6) < 0.5


def test_gravity_from_a_free_flight_and_from_a_plumb_bob():
    t = np.linspace(0, 0.5, 30)
    tilt = np.radians(4)
    g_dir = np.array([np.sin(tilt), np.cos(tilt)])  # sheet frame, y down
    xy = 0.5 * 9810 * t[:, None] ** 2 * g_dir + np.random.default_rng(0).normal(0, 0.5, (30, 2))
    g = from_free_fall(t, xy)
    assert g.source == "data" and np.degrees(np.arccos(np.clip(g.down @ g_dir, -1, 1))) < 0.5
    assert 0 < g.sigma_deg < 1
    assert from_free_fall(t[:5], xy[:5]) is None
    p = from_plumb(
        np.array([100.0, 0.0]), np.array([100.0 + 600 * np.sin(tilt), 600 * np.cos(tilt)])
    )
    assert p.source == "plumb" and np.allclose(p.down, g_dir)


def test_plumb_direction_from_a_decaying_swing_that_started_on_one_side():
    t = np.linspace(0, 6, 400)
    centre = np.radians(5)  # the true rest direction, 5 degrees from the sheet's down axis
    alpha = np.pi / 2 - centre + 0.5 * np.exp(-0.25 * t) * np.cos(2 * np.pi * t / 1.4)
    pivot = np.array([200.0, 50.0])
    xy = pivot + 600 * np.column_stack([np.cos(alpha), np.sin(alpha)])
    g = plumb_from_oscillation(t, xy, pivot)
    expected = np.array([np.cos(np.pi / 2 - centre), np.sin(np.pi / 2 - centre)])
    assert g.source == "plumb"
    assert np.degrees(np.arccos(np.clip(g.down @ expected, -1, 1))) < 0.3
    assert plumb_from_oscillation(t[:5], xy[:5], pivot) is None


def board_mask(sc):
    prim = sc.system.static()[0]
    pts = np.round(sc.cam.project(prim.pts) * 16).astype(np.int32)
    mask = np.zeros((480, 854), np.uint8)
    cv2.fillPoly(mask, [pts], 1, cv2.LINE_8, 4)
    return mask


def test_ramp_angle_from_the_board_edge_and_the_path_cross_check():
    sc = scene(system="ramp", azimuth_deg=10, elevation_deg=4)
    cal = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, paper="A4", ramp_mask=board_mask(sc))
    assert cal.ramp.incline_deg == pytest.approx(20.0, abs=0.5)
    assert cal.ramp.angle_deg < 0  # the board descends to the right
    assert cal.ramp.sigma_deg > 0.5  # dominated by the camera-level gravity direction
    path = np.column_stack(
        [np.linspace(0, 0.5, 20), -np.tan(np.radians(20)) * np.linspace(0, 0.5, 20)]
    )
    diff, note = cross_check(cal.ramp.angle_deg, path)
    assert diff < 1.0 and note is None
    off = np.column_stack([path[:, 0], -np.tan(np.radians(25)) * path[:, 0]])
    diff, note = cross_check(cal.ramp.angle_deg, off)
    assert diff > 4 and note.code == "ramp_angle_disagreement"
    direct = ramp_angle_from_mask(board_mask(sc), np.asarray(cal.H), cal.gravity_down)
    assert direct.angle_deg == pytest.approx(cal.ramp.angle_deg)


def test_monte_carlo_grows_with_corner_noise_and_with_distance_from_the_sheet():
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    cal = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, paper="A4")
    args = (np.asarray(cal.corners_px), np.asarray(cal.corners_mm))
    centre = np.asarray(cal.centre_px)
    small = monte_carlo(*args, 0.1, centre, 0.02, cal.offset.z_m)
    large = monte_carlo(*args, 0.5, centre, 0.02, cal.offset.z_m)
    assert large.components["corners"] > 3 * small.components["corners"]
    assert large.sigma_scale_rel >= max(large.components.values()) * 0.9
    near = cal.scale_sigma_at(np.asarray(cal.corners_px).mean(axis=0)[None])
    far = cal.scale_sigma_at(np.array([[10.0, 10.0]]))
    assert far > 3 * near
    again = monte_carlo(*args, 0.1, centre, 0.02, cal.offset.z_m)
    assert again.sigma_scale_rel == small.sigma_scale_rel  # seeded


def test_calibration_files_and_json_round_trip(tmp_path):
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    timing = {"time_factor": 0.125, "classification": "cfr", "sigma_time_rel": 1e-3, "ignored": 1}
    cal = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, paper="A4", timing=timing)
    assert cal.timing == {"time_factor": 0.125, "classification": "cfr", "sigma_time_rel": 1e-3}
    write_calibration(cal, tmp_path, sc.plate_u8)
    again = Calibration.model_validate_json((tmp_path / "calibration.json").read_text())
    assert again == cal
    overlay = cv2.imread(str(tmp_path / "calib_overlay.png"))
    rect = cv2.imread(str(tmp_path / "rectified.png"))
    assert overlay.shape == sc.plate_u8.shape and (overlay != sc.plate_u8).any()
    assert rect.shape[:2] == (int((297 + 120) * 2), int((210 + 120) * 2))
    inside = rect[120:-120, 120:-120]  # the sheet, seen head-on
    assert inside.mean() > 200


def test_overlay_grid_lines_are_5cm_apart_in_the_world():
    sc = scene(system="free_fall", azimuth_deg=0, elevation_deg=0)
    cal = calibrate(SIZE, sheet_mask=sc.sheet_mask > 0, paper="A4")
    blank = np.zeros((480, 854, 3), np.uint8)
    img = draw_overlay(blank, cal)
    column = (img[240, :, :].sum(axis=1) > 60).nonzero()[0]
    # vertical grid lines cross row 240; spacing in pixels = 50 mm / mm-per-px
    gaps = np.diff(column[column > 600])
    assert gaps.size and np.median(gaps[gaps > 3]) == pytest.approx(
        50 / cal.mm_per_px_center, rel=0.1
    )
    assert rectified_image(img, cal).shape[2] == 3


def test_calibrate_needs_a_mask_or_corners():
    with pytest.raises(ValueError):
        calibrate(SIZE)
