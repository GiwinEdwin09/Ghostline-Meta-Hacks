import cv2
import numpy as np
import pytest
from synth_helpers import scene

from ghostline.calib import calibrate
from ghostline.calib.homography import (
    apply_h,
    compute_homography,
    corners_mm,
    guess_paper,
    local_scale,
    local_scale_many,
    order_corners,
)
from ghostline.calib.offset import (
    camera_distance_m,
    depth_of_sheet_points,
    focal_px,
    from_plane_of_motion,
    offset_scale,
    pose_from_homography,
    to_plane_of_motion,
)

CAMERAS = [
    dict(azimuth_deg=0, elevation_deg=0),
    dict(azimuth_deg=10, elevation_deg=4),
    dict(azimuth_deg=-25, elevation_deg=8, sheet_rot_deg=5),
    dict(azimuth_deg=35, elevation_deg=15, sheet_rot_deg=20, sheet_landscape=True),
    dict(azimuth_deg=15, elevation_deg=-6, roll_deg=3, sheet_rot_deg=12),
]


ORIGIN_MM = np.array([105.0, 148.5])


def sheet_homography(sc):
    """Exact pixel -> millimetre homography of a synthetic scene, with x right and y down and the
    origin at the sheet's top-left (the world's X right / Y up, scaled to mm)."""
    to_mm = np.array([[1000, 0, ORIGIN_MM[0]], [0, -1000, ORIGIN_MM[1]], [0, 0, 1.0]])
    return to_mm @ np.linalg.inv(sc.cam.plane_homography(0.0))


def test_corners_are_ordered_clockwise_from_the_top_left():
    rect = np.array([[10, 10], [110, 10], [110, 80], [10, 80]], float)
    for shuffle in ([2, 0, 3, 1], [1, 3, 0, 2]):
        assert np.allclose(order_corners(rect[shuffle]), rect)
    ang = np.radians(35)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    tilted = rect @ rot.T
    ordered = order_corners(tilted[[3, 1, 0, 2]])
    area = np.sum(
        ordered[:, 0] * np.roll(ordered[:, 1], -1) - np.roll(ordered[:, 0], -1) * ordered[:, 1]
    )
    assert area > 0  # clockwise on screen (y down)
    assert ordered.sum(axis=1).argmin() == 0


def test_paper_size_guess_and_orientation_of_the_long_edge():
    a4 = np.array([[0, 0], [100, 0], [100, 141.4], [0, 141.4]], float)
    letter = np.array([[0, 0], [100, 0], [100, 129.4], [0, 129.4]], float)
    assert guess_paper(a4)[0] == "A4" and guess_paper(a4)[2]
    assert guess_paper(letter)[0] == "Letter" and guess_paper(letter)[2]
    assert not guess_paper(np.array([[0, 0], [100, 0], [100, 135.4], [0, 135.4]], float))[2]
    assert np.allclose(corners_mm(a4, "A4")[2], [210, 297])  # tall quad: long edge down the side
    landscape = a4[:, ::-1] * [1, 1]
    assert np.allclose(corners_mm(order_corners(landscape), "A4")[2], [297, 210])


def test_homography_matches_opencv_and_maps_the_corners_exactly():
    src = np.array([[100, 50], [420, 80], [400, 300], [90, 280]], float)
    dst = np.array([[0, 0], [210, 0], [210, 297], [0, 297]], float)
    h = compute_homography(src, dst)
    assert np.allclose(apply_h(h, src), dst, atol=1e-9)
    opencv = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    assert np.allclose(h, opencv, rtol=1e-4, atol=1e-6)


def test_local_scale_of_a_pure_scaling_and_vectorised_version():
    h = np.diag([0.5, 0.5, 1.0])
    assert local_scale(h, np.array([10.0, 20.0])) == pytest.approx(0.5)
    sc = scene(system="free_fall", azimuth_deg=20, elevation_deg=5)
    hh = sheet_homography(sc)
    pts = np.array([[100.0, 100.0], [400.0, 250.0], [700.0, 400.0]])
    many = local_scale_many(hh, pts)
    assert np.allclose(many, [local_scale(hh, p) for p in pts])
    assert many[0] != pytest.approx(many[2], rel=0.01)  # perspective changes the scale


@pytest.mark.parametrize("cam", CAMERAS)
def test_pose_decomposition_recovers_distance_and_the_point_under_the_camera(cam):
    sc = scene(system="free_fall", **cam)
    pose = pose_from_homography(sheet_homography(sc), sc.cam.f_px, (854, 480))
    assert pose is not None
    z, foot = pose
    c = sc.cam.center
    assert z == pytest.approx(abs(c[2]), rel=1e-6)
    assert np.allclose(foot, ORIGIN_MM + c[:2] * 1000 * [1, -1], atol=0.05)
    wrong_f = pose_from_homography(sheet_homography(sc), sc.cam.f_px * 1.15, (854, 480))
    assert 1.0 < wrong_f[0] / z < 1.16  # a 15% focal-length error moves Z by at most that


def test_depth_of_sheet_points_matches_the_camera():
    sc = scene(system="free_fall", azimuth_deg=30, elevation_deg=10)
    h = sheet_homography(sc)
    world = np.array([[0.3, 0.2, 0.0], [-0.4, -0.3, 0.0], [0.0, 0.0, 0.0]])
    w_mm = np.column_stack([world[:, 0] * 1000 + 105, -world[:, 1] * 1000 + 148.5])
    depth = depth_of_sheet_points(h, sc.cam.f_px, (854, 480), w_mm)
    assert np.allclose(depth, sc.cam.depth(world), rtol=1e-6)


def test_plane_offset_formulas_round_trip_and_match_the_plan():
    centre = np.array([100.0, 150.0])
    pts = np.array([[400.0, 500.0], [0.0, 0.0]])
    s = offset_scale(0.02, 2.0)
    assert s == pytest.approx(0.99)
    assert np.allclose(to_plane_of_motion(pts, centre, s), centre + (pts - centre) * 0.99)
    assert np.allclose(from_plane_of_motion(to_plane_of_motion(pts, centre, s), centre, s), pts)
    f = focal_px(1920, None)
    assert f == pytest.approx(1920 * 26 / 36)
    assert camera_distance_m(f, 0.297 * f / 2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("cam", CAMERAS)
def test_exact_corners_give_an_exact_world_transform_in_the_plane_of_motion(cam):
    """The scale error on a synthetic perspective render is far below the 0.2% target."""
    sc = scene(system="free_fall", **cam)
    cal = calibrate((854, 480), corners_px=order_corners(sc.corners_px), paper="A4")
    rng = np.random.default_rng(0)
    xyz = np.column_stack(
        [rng.uniform(-0.5, 0.5, 40), rng.uniform(-0.5, 0.5, 40), np.full(40, 0.02)]
    )
    px = sc.cam.project(xyz)
    world = cal.px_to_world(px)
    # distances don't depend on the orientation of the world frame, only on scale and perspective
    iu = np.triu_indices(len(xyz), 1)
    d_true = np.linalg.norm(xyz[:, None, :2] - xyz[None, :, :2], axis=2)[iu]
    d_est = np.linalg.norm(world[:, None] - world[None], axis=2)[iu]
    assert np.abs(d_est / d_true - 1).max() < 5e-4
    assert np.allclose(cal.world_to_px(world), px, atol=1e-6)
    assert cal.offset.method == "pose"
    assert cal.m_per_px == pytest.approx(cal.mm_per_px_center * cal.offset.scale / 1000)


def test_world_frame_is_centred_on_the_sheet_with_y_up_when_the_camera_is_level():
    sc = scene(system="free_fall", azimuth_deg=12, elevation_deg=5)
    cal = calibrate((854, 480), corners_px=order_corners(sc.corners_px), paper="A4")
    xyz = np.array([[0.0, 0.0, 0.02], [0.3, 0.4, 0.02], [-0.2, -0.5, 0.02]])
    world = cal.px_to_world(sc.cam.project(xyz))
    assert np.allclose(world, xyz[:, :2], atol=5e-4)
    assert abs(cal.gravity.angle_deg) < 0.5 and cal.gravity.source == "camera_level"
