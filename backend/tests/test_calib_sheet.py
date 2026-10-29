import cv2
import numpy as np
import pytest
from synth_helpers import scene

from ghostline.calib.homography import order_corners
from ghostline.calib.sheet import detect_sheet_corners

CAMERAS = [
    dict(azimuth_deg=10, elevation_deg=4),
    dict(azimuth_deg=-25, elevation_deg=8, sheet_rot_deg=5),
    dict(azimuth_deg=35, elevation_deg=15, sheet_rot_deg=20, sheet_landscape=True),
    dict(azimuth_deg=15, elevation_deg=-6, sheet_rot_deg=40),
]


def gray(sc):
    return cv2.cvtColor(sc.plate_u8, cv2.COLOR_BGR2GRAY)


@pytest.mark.parametrize("cam", CAMERAS)
def test_corners_from_a_binary_mask_are_within_a_pixel(cam):
    sc = scene(system="free_fall", **cam)
    det = detect_sheet_corners(sc.sheet_mask > 0)
    err = np.linalg.norm(det.corners_px - order_corners(sc.corners_px), axis=1)
    assert det.method == "ransac_lines" and not det.subpix
    assert err.max() < 1.0
    assert det.sigma_corner_px >= 0.3


@pytest.mark.parametrize("cam", CAMERAS)
def test_grey_level_refinement_brings_corners_under_a_third_of_a_pixel(cam):
    sc = scene(system="free_fall", **cam)
    det = detect_sheet_corners(sc.sheet_mask > 0, gray(sc))
    err = np.linalg.norm(det.corners_px - order_corners(sc.corners_px), axis=1)
    assert det.subpix and err.max() < 0.3


def test_noisy_median_image_still_refines():
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    noisy = gray(sc).astype(float) + np.random.default_rng(0).normal(0, 1.5, sc.plate.shape[:2])
    det = detect_sheet_corners(sc.sheet_mask > 0, noisy.clip(0, 255).astype(np.uint8))
    err = np.linalg.norm(det.corners_px - order_corners(sc.corners_px), axis=1)
    assert err.max() < 0.4


def test_rounded_corners_do_not_bias_the_fit():
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    mask = (sc.sheet_mask > 0).astype(np.uint8)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    rounded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disc)
    assert rounded.sum() < mask.sum()
    det = detect_sheet_corners(rounded > 0)
    err = np.linalg.norm(det.corners_px - order_corners(sc.corners_px), axis=1)
    assert err.max() < 1.2


def test_a_curled_corner_and_an_object_in_front_are_tolerated():
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    mask = (sc.sheet_mask > 0).astype(np.uint8)
    corners = order_corners(sc.corners_px)
    cut = np.array([corners[0], corners[0] + (corners[1] - corners[0]) * 0.07,
                    corners[0] + (corners[3] - corners[0]) * 0.07])  # fmt: skip
    cv2.fillPoly(mask, [np.round(cut).astype(np.int32)], 0)
    centre = tuple(np.round(corners.mean(axis=0)).astype(int))
    cv2.circle(mask, centre, 9, 0, -1)  # a hole where a ball sits
    det = detect_sheet_corners(mask > 0)
    err = np.linalg.norm(det.corners_px[1:] - corners[1:], axis=1)
    assert err.max() < 1.2 and det.method == "ransac_lines"


def test_the_biggest_blob_is_the_sheet():
    sc = scene(system="free_fall", azimuth_deg=10, elevation_deg=4)
    mask = (sc.sheet_mask > 0).astype(np.uint8)
    mask[10:16, 10:16] = 1
    det = detect_sheet_corners(mask > 0)
    assert np.linalg.norm(det.corners_px - order_corners(sc.corners_px), axis=1).max() < 1.0


def test_a_shape_that_is_not_a_sheet_falls_back_with_a_notice():
    mask = np.zeros((240, 320), np.uint8)
    cv2.circle(mask, (160, 120), 60, 1, -1)
    det = detect_sheet_corners(mask > 0)
    assert det.method == "min_area_rect"
    assert "sheet_min_area_rect" in {n.code for n in det.notices}


def test_an_empty_mask_is_an_error():
    with pytest.raises(ValueError):
        detect_sheet_corners(np.zeros((50, 50), bool))
