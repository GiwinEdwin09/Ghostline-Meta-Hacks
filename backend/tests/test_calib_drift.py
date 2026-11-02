import cv2
import numpy as np
import pytest

from ghostline.calib.drift import Drift, shifts_from_frames, undrift


def scene(seed=0, size=(180, 320)):
    """A textured background, like a wood-grain cabinet."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.uniform(0, 255, size).astype(np.float32), (0, 0), 2)


def shifted(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, img.shape[::-1], borderMode=cv2.BORDER_REFLECT)


def test_a_settling_camera_is_measured_relative_to_where_it_mostly_sits():
    img = scene()
    path = [(0.0, 0.0), (0.0, 4.0), (0.0, 8.0)] + [(1.5, 10.0)] * 6  # slides, then settles
    drift = shifts_from_frames(((i, shifted(img, *d)) for i, d in enumerate(path)), scale=2.0)
    # in master pixels (scale 2), relative to the settled position
    assert drift.shift_px[-1] == pytest.approx([0, 0], abs=0.5)
    assert drift.shift_px[0] == pytest.approx([-3, -20], abs=1.0)
    assert drift.max_px == pytest.approx(np.hypot(3, 20), abs=1.0)


def test_a_small_moving_object_on_a_still_scene_is_not_drift():
    img = scene()
    frames = []
    for i in range(30):
        f = img.copy()
        cv2.circle(f, (40 + 8 * i, 90), 6, 255, -1)  # a ball crossing the frame
        frames.append((i, f))
    assert shifts_from_frames(frames).max_px < 0.5


def test_a_hand_sweeping_through_one_sample_is_smoothed_away():
    img = scene()
    frames = [(i, img) for i in range(12)]
    frames[6] = (6, shifted(img, 15, 0))
    assert shifts_from_frames(frames).max_px < 0.5


def test_shift_is_interpolated_between_samples_and_removed_from_tracks(tmp_path):
    drift = Drift(np.array([0, 10]), np.array([[0.0, 0.0], [2.0, -4.0]]))
    assert drift.at([5])[0] == pytest.approx([1.0, -2.0])
    drift.save(tmp_path / "drift.npz")
    again = Drift.load(tmp_path / "drift.npz")
    cx, cy = undrift(np.array([101.0]), np.array([48.0]), np.array([5]), again)
    assert (cx[0], cy[0]) == pytest.approx((100.0, 50.0))
