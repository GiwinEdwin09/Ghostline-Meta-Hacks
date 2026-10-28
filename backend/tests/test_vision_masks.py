import math

import cv2
import numpy as np
import pytest

from ghostline.vision import mask_features, place_mask

FRAME = (200, 120)  # width, height


def disc(radius, size=60):
    raster = np.zeros((size, size), np.uint8)
    cv2.circle(raster, (size // 2, size // 2), radius, 1, thickness=-1)
    return raster


def test_disc_features():
    features = mask_features(disc(10), left=50, top=30, frame_size=FRAME)

    assert features.area == pytest.approx(math.pi * 10**2, rel=0.05)
    assert (features.cx, features.cy) == pytest.approx((50 + 30, 30 + 30), abs=0.01)
    assert features.eccentricity < 0.05
    assert not features.touches_border


def test_centroid_is_offset_by_the_box_origin_and_exact_for_a_rectangle():
    raster = np.ones((7, 11), np.uint8)  # 11 wide, 7 tall

    features = mask_features(raster, left=100, top=40, frame_size=FRAME)

    assert features.area == 77
    # Pixel centres sit at integers, so the centre of a 11x7 block starting at (100, 40):
    assert (features.cx, features.cy) == pytest.approx((105.0, 43.0))
    assert features.bbox == (100, 40, 111, 47)


def test_moving_the_box_moves_the_centroid_one_for_one():
    a = mask_features(disc(8), 10, 10, FRAME)
    b = mask_features(disc(8), 13, 17, FRAME)

    assert (b.cx - a.cx, b.cy - a.cy) == pytest.approx((3.0, 7.0))


@pytest.mark.parametrize("degrees", [0, 30, 60, 90, 120, -45])
def test_orientation_of_a_tilted_ellipse(degrees):
    raster = np.zeros((101, 101), np.uint8)
    cv2.ellipse(raster, (50, 50), (40, 10), degrees, 0, 360, 1, thickness=-1)

    features = mask_features(raster, 0, 0, (101, 101))

    expected = math.radians(degrees)
    # The major axis has no direction, so compare modulo pi.
    difference = (features.theta - expected + math.pi / 2) % math.pi - math.pi / 2
    assert abs(difference) < math.radians(1.5)
    assert features.eccentricity == pytest.approx(math.sqrt(1 - (10 / 40) ** 2), abs=0.03)


def test_orientation_follows_the_image_axes_with_y_down():
    raster = np.zeros((60, 60), np.uint8)
    cv2.line(raster, (10, 10), (50, 50), 1, thickness=5)  # down and to the right on screen

    assert mask_features(raster, 0, 0, (60, 60)).theta == pytest.approx(math.pi / 4, abs=0.03)


@pytest.mark.parametrize(
    ("left", "top", "touches"),
    [(0, 50, True), (50, 0, True), (50, 50, False), (200 - 11, 50, True), (50, 120 - 7, True)],
)
def test_border_touch(left, top, touches):
    raster = np.ones((7, 11), np.uint8)

    assert mask_features(raster, left, top, FRAME).touches_border is touches


def test_an_empty_mask_has_no_features():
    assert mask_features(np.zeros((5, 5), np.uint8), 0, 0, FRAME) is None


def test_place_mask_puts_the_raster_at_the_box_origin():
    raster = np.ones((3, 4), np.uint8)

    placed = place_mask(raster, left=10, top=20, frame_size=FRAME)

    assert placed.shape == (120, 200) and placed.sum() == 12
    assert placed[20:23, 10:14].all()


def test_place_mask_clips_at_the_frame_edge():
    placed = place_mask(np.ones((10, 10), np.uint8), left=195, top=115, frame_size=FRAME)

    assert placed.sum() == 5 * 5
