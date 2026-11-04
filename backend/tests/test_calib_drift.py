import cv2
import numpy as np
import pytest

from ghostline.calib.drift import (
    Drift,
    _consensus_shift,
    shifts_from_frames,
    steady_span,
    undrift,
)


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


def test_a_window_is_measured_from_where_the_camera_sits_inside_it():
    drift = Drift(np.arange(0, 100, 10), np.array([[0.0, 0.0]] * 5 + [[30.0, 0.0]] * 5))
    inside = drift.between(50, 100)
    assert list(inside.frames) == [50, 60, 70, 80, 90] and inside.max_px == pytest.approx(0)
    assert drift.between(200, 300).max_px == 0


def test_the_steady_span_ends_before_the_phone_is_picked_up():
    frames = np.arange(600)  # one sample per frame at 30 fps
    shift = np.zeros((600, 2))
    shift[570:, 0] = np.linspace(0, 400, 30)  # the last second: a hand picks the phone up
    # 60 px is first passed at frame 575; the cut is half a second (15 frames) before it
    assert steady_span(Drift(frames, shift), 0, 600, 30.0) == (0, 560)


def test_the_steady_span_starts_after_the_camera_settles():
    frames = np.arange(600)
    shift = np.zeros((600, 2))
    shift[:20, 1] = np.linspace(-150, 0, 20)  # settling after the record tap
    assert steady_span(Drift(frames, shift), 0, 600, 30.0) == (11 + 15, 600)


@pytest.mark.parametrize(
    "shift",
    [
        np.zeros((600, 2)),  # a tripod
        np.c_[np.linspace(0, 50, 600), np.zeros(600)],  # a slow sag, all of it correctable
        np.c_[100 * np.sin(np.arange(600) / 5), np.zeros(600)],  # handheld throughout
    ],
)
def test_the_steady_span_keeps_the_trim_when_there_is_no_single_jump(shift):
    assert steady_span(Drift(np.arange(600), shift), 30, 590, 30.0) == (30, 590)


def test_the_background_outvotes_a_spinning_object_that_has_most_of_the_corners():
    """A spinner on a blank sheet: 60% of the tracked corners are on its arms, each moving its own
    way; the 40% on the background agree. The median would be pulled off; the consensus is not."""
    rng = np.random.default_rng(3)
    arms = rng.normal(0, 3, (180, 2)) + [8.0, -6.0]  # tracking errors, lopsided like a real swirl
    background = rng.normal(0, 0.1, (120, 2)) + [0.3, -0.2]
    moves = np.vstack([arms, background])
    assert _consensus_shift(moves) == pytest.approx([0.3, -0.2], abs=0.05)
    assert np.linalg.norm(np.median(moves, axis=0) - [0.3, -0.2]) > 1.0
