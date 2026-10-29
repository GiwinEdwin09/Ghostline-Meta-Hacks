import numpy as np
import pytest

from ghostline.traj.events import (
    detect_impacts,
    detect_release_stop,
    flight_segments,
    impact_drop_mask,
    out_of_plane,
    retime_check,
    speed_and_threshold,
)
from ghostline.traj.noise import local_fit

SIGMA = 0.0015  # metres, about 0.2 px


def fall_series(n=240, fps=120.0, release=0.5, stop=1.0, seed=0, g=9.81):
    """Rest, free fall until `stop`, rest again; positions in metres with noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    tau = np.clip(t - release, 0, stop - release)
    y = 0.5 - 0.5 * g * tau**2
    xy = np.column_stack([np.full(n, 0.2), y]) + rng.normal(0, SIGMA, (n, 2))
    return t, xy


def test_release_and_stop_are_found_to_within_a_few_frames():
    t, xy = fall_series()
    ev = detect_release_stop(t, xy, local_fit(t, xy), SIGMA)
    # the object is first seen to leave its rest position at 4 sigma, a few frames after release
    assert ev.release_t == pytest.approx(0.5, abs=0.05)
    assert ev.release_t >= 0.5 - 1 / 120
    assert ev.stop_t == pytest.approx(1.0, abs=0.05)
    assert ev.window_end_t == ev.stop_t
    assert ev.last_moving_t is not None


def test_a_clip_that_starts_in_motion_releases_at_the_first_frame():
    t, xy = fall_series(release=0.0, stop=5.0)
    ev = detect_release_stop(t, xy, local_fit(t, xy), SIGMA)
    assert ev.release_index <= 2


def test_nothing_moving_gives_no_release():
    t = np.arange(200) / 60
    xy = np.random.default_rng(0).normal(0, SIGMA, (200, 2))
    ev = detect_release_stop(t, xy, local_fit(t, xy), SIGMA)
    assert ev.release_t is None and ev.stop_t is None and ev.window_end_t is None


def test_a_short_rest_at_the_end_still_counts_as_the_stop():
    t, xy = fall_series(n=180, release=0.5, stop=1.2)  # only 0.3 s at rest
    ev = detect_release_stop(t, xy, local_fit(t, xy), SIGMA)
    assert ev.stop_t == pytest.approx(1.2, abs=0.05)


def test_an_oscillation_never_stops_because_it_only_pauses_at_turning_points():
    t = np.arange(360) / 60
    x = 0.2 * np.sin(2 * np.pi * t / 1.4) * (t > 0.5)
    xy = np.column_stack([x, np.zeros_like(x)]) + np.random.default_rng(1).normal(
        0, SIGMA, (360, 2)
    )
    ev = detect_release_stop(t, xy, local_fit(t, xy), SIGMA)
    assert ev.release_t == pytest.approx(0.5, abs=0.1) and ev.stop_t is None
    assert ev.window_end_t > 5.5


def test_speed_threshold_follows_the_noise_level():
    t, xy = fall_series()
    fit = local_fit(t, xy)
    _, low = speed_and_threshold(fit, SIGMA)
    _, high = speed_and_threshold(fit, 4 * SIGMA)
    assert np.allclose(high[fit.valid], 4 * low[fit.valid])


def bounce_series(e=0.75, fps=120.0, n=480, sigma=SIGMA, seed=0, g=9.81):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    h0, v = 0.9, 0.0
    hit = np.sqrt(2 * h0 / g)
    impacts = [hit]
    v = g * hit
    while v * e * 2 / g > 0.1:
        v *= e
        impacts.append(impacts[-1] + 2 * v / g)
    y = np.empty(n)
    for i, ti in enumerate(t):
        if ti < impacts[0]:
            y[i] = h0 - 0.5 * g * ti**2
        else:
            k = np.searchsorted(impacts, ti, side="right") - 1
            v_up = g * hit * e**k * e
            tau = ti - impacts[k]
            y[i] = max(v_up * tau - 0.5 * g * tau**2, 0.0)
    return t, y + rng.normal(0, sigma, n), impacts


def test_impacts_are_located_to_a_fraction_of_a_frame_and_two_frames_are_dropped_each_side():
    t, y, impacts = bounce_series()
    frames = np.arange(len(t))
    found = detect_impacts(t, y, frames, SIGMA)
    assert len(found) >= 3
    for imp, truth in zip(found, impacts, strict=False):
        assert imp.t == pytest.approx(truth, abs=1.5 / 120)
    drop = impact_drop_mask(frames, found)
    assert drop.sum() == 5 * len(found)
    assert drop[found[0].frame - 2 : found[0].frame + 3].all() and not drop[found[0].frame - 3]


def test_segments_count_the_flights():
    t, y, impacts = bounce_series()
    found = detect_impacts(t, y, np.arange(len(t)), SIGMA)
    seg = flight_segments(t, found)
    assert seg[0] == 0 and seg.max() == len(found)
    assert (np.diff(seg) >= 0).all()
    assert flight_segments(t, []).max() == 0


def test_a_ball_at_rest_or_a_smooth_minimum_is_not_an_impact():
    t = np.arange(300) / 60
    rest = np.random.default_rng(0).normal(0, SIGMA, 300)
    assert detect_impacts(t, rest, np.arange(300), SIGMA) == []
    swing = 0.3 * (1 - np.cos(2 * np.pi * t / 2.0))  # a pendulum's height: smooth minima
    smooth_min = detect_impacts(
        t, swing + np.random.default_rng(1).normal(0, SIGMA, 300), np.arange(300), SIGMA
    )
    assert smooth_min == []


def flight(g, fps=120.0, n=60):
    t = np.arange(n) / fps
    y = 1.0 - 0.5 * g * t**2 + np.random.default_rng(0).normal(0, 0.001, n)
    return t, y


@pytest.mark.parametrize("k", [2, 4, 8])
def test_a_clip_slowed_k_times_looks_like_gravity_over_k_squared(k):
    t, y = flight(9.81 / k**2, n=int(60 * np.sqrt(k)))
    t = t * 3 if k == 2 else t * 12 if k == 4 else t * 40  # stretch so the flight lasts long enough
    y = 1.0 - 0.5 * (9.81 / k**2) * t**2
    out = retime_check(t, y, np.zeros(len(t), int), np.ones(len(t), bool))
    assert out["suspected_factor"] == k and out["flights"][0]["g_fit"] == pytest.approx(
        9.81 / k**2, rel=1e-6
    )


def test_normal_gravity_and_drag_are_not_flagged():
    t, y = flight(9.81)
    assert retime_check(t, y, np.zeros(60, int), np.ones(60, bool))["suspected_factor"] is None
    t, y = flight(7.5)  # a foam ball with drag
    assert retime_check(t, y, np.zeros(60, int), np.ones(60, bool))["suspected_factor"] is None


def test_short_or_sparse_flights_are_not_checked():
    t, y = flight(9.81 / 16, n=6)
    assert retime_check(t, y, np.zeros(6, int), np.ones(6, bool))["checked"] is False


def test_each_bounce_flight_is_checked_separately():
    t, y, impacts = bounce_series(n=420)
    found = detect_impacts(t, y, np.arange(len(t)), SIGMA)
    keep = ~impact_drop_mask(np.arange(len(t)), found)
    seg = flight_segments(t[keep], found)
    out = retime_check(t[keep], y[keep], seg, np.ones(keep.sum(), bool))
    assert len(out["flights"]) >= 3 and out["suspected_factor"] is None
    assert all(f["g_fit"] == pytest.approx(9.81, rel=0.05) for f in out["flights"][:3])


def area_case(kind, n=200, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60
    x = 0.3 * np.sin(2 * np.pi * t / 1.4)
    speed = np.abs(0.3 * 2 * np.pi / 1.4 * np.cos(2 * np.pi * t / 1.4))
    base = 400.0
    if kind == "steady":
        area = base + rng.normal(0, 4, n)
    elif kind == "toward_camera":  # swings toward the camera: area follows the position
        area = base * (1 + 0.15 * np.sin(2 * np.pi * t / 1.4)) + rng.normal(0, 4, n)
    elif kind == "drift":  # slowly approaches the camera
        area = base * (1 + 0.12 * t / t[-1]) + rng.normal(0, 4, n)
    else:  # motion blur: area grows with speed
        area = base + 60 * speed + rng.normal(0, 4, n)
    return area, {"x": x, "t": t}, speed


def test_out_of_plane_flags_area_that_follows_the_position_or_drifts():
    for kind in ("toward_camera", "drift"):
        area, cols, speed = area_case(kind)
        res = out_of_plane(area, cols, speed)
        assert res["flag"], kind
        assert res["dz_over_z"] == pytest.approx(res["blur_corrected_variation"] / 2)


def test_out_of_plane_ignores_steady_area_and_blur():
    for kind in ("steady", "blur"):
        area, cols, speed = area_case(kind)
        res = out_of_plane(area, cols, speed)
        assert not res["flag"], kind
    assert area_case("blur")[0].std() > 4  # the blur case really does vary


def test_out_of_plane_needs_enough_samples():
    assert out_of_plane(np.array([1.0, 2.0]), {"x": np.array([1.0, 2.0])}, None)["checked"] is False
    assert out_of_plane(np.full(20, np.nan), {"x": np.arange(20.0)}, None)["flag"] is False
