import numpy as np

from ghostline.ingest.trim import suggest_trim


def signal(n=200, start=60, stop=120, noise=0.01, seed=0):
    e = np.abs(np.random.default_rng(seed).normal(0.05, noise, n))
    e[start:stop] += 0.5
    return e


def test_motion_window_is_padded_by_three_tenths_of_a_second():
    t = np.arange(200) / 60
    trim = suggest_trim(signal(), t)
    assert trim.reason == "motion_window"
    assert trim.start_frame == 60 - 18 and trim.stop_frame == 120 + 18
    assert trim.start_t == t[42] and trim.stop_t == t[137]


def test_padding_is_in_real_seconds_after_the_time_factor():
    t = np.arange(200) / 240  # 240 fps real time
    trim = suggest_trim(signal(), t)
    assert trim.start_frame == 0 and trim.stop_frame == 120 + 72


def test_no_motion_gives_the_whole_clip():
    t = np.arange(200) / 60
    trim = suggest_trim(np.full(200, 0.05) + np.random.default_rng(1).normal(0, 0.005, 200), t)
    assert trim.reason == "no_motion_detected"
    assert (trim.start_frame, trim.stop_frame) == (0, 200)


def test_a_noiseless_static_clip_does_not_divide_by_zero():
    trim = suggest_trim(np.zeros(100), np.arange(100) / 60)
    assert trim.reason == "no_motion_detected"


def test_a_lone_keyframe_pulse_is_not_motion():
    e = signal()
    e[190:192] += 3.0  # an I-frame pulse well after the motion
    trim = suggest_trim(e, np.arange(200) / 60)
    assert trim.stop_frame == 138


def test_turning_points_inside_a_swing_do_not_split_the_window():
    e = signal(start=40, stop=160)
    e[80:86] = 0.05  # a pendulum pausing at its turning point
    trim = suggest_trim(e, np.arange(200) / 60)
    assert trim.start_frame == 22 and trim.stop_frame == 178


def test_search_can_be_limited_to_the_usable_range():
    e = signal(start=60, stop=120)
    e[10:20] += 2.0  # a different rate section outside the usable range
    trim = suggest_trim(e, np.arange(200) / 60, usable=(30, 200))
    assert trim.start_frame == 42
