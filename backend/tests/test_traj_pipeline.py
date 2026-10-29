import json

import cv2
import numpy as np
import pandas as pd
import pytest
from synth_helpers import analytic_tracks, clip, tracks_from_truth

from ghostline.calib import calibrate
from ghostline.calib.homography import local_scale_many, order_corners
from ghostline.traj import (
    FLAG_ALIAS_RISK,
    FLAG_AT_REST,
    FLAG_NEAR_IMPACT,
    TrackSeries,
    TrajectoryError,
    build_trajectory,
)

SIZE = (854, 480)


def m_per_px_along(cal, track):
    """True metre-per-pixel scale along the path, for turning a pixel noise into metres."""
    scale = local_scale_many(cal.H_np, track.px) * cal.offset.scale / 1000
    return float(np.mean(scale))


def pos_error(res, truth_xyz):
    fr = res.trajectory["frame"].to_numpy()
    return res.trajectory[["x", "y"]].to_numpy() - truth_xyz[fr, :2]


@pytest.fixture(scope="module")
def drop():
    sc, cal, track, t, truth = analytic_tracks("free_fall", 2.0, fps=120.0)
    return sc, cal, track, truth, build_trajectory(track, cal, "free_fall")


def test_free_fall_comes_back_in_metres_within_noise(drop):
    sc, cal, track, truth, res = drop
    err = pos_error(res, truth["tracks"]["object"])
    sigma_m = 0.2 * m_per_px_along(cal, track)
    assert np.sqrt((err**2).sum(axis=1).mean()) < 3 * sigma_m
    assert np.abs(err).max() < 8 * sigma_m
    assert res.trajectory["y"].max() == pytest.approx(0.5, abs=0.01)


def test_free_fall_events_and_gravity_direction(drop):
    sc, _, _, _, res = drop
    ev = res.summary["events"]
    assert ev["release_t"] == pytest.approx(0.4, abs=0.05)
    assert ev["stop_t"] == pytest.approx(sc.system.events()["stop_t_s"], abs=0.05)
    g = res.summary["gravity"]
    assert g["source"] == "data" and abs(g["angle_deg"]) < 0.5
    window = res.trajectory[(res.trajectory["flags"] & FLAG_AT_REST) == 0]
    assert window["t"].min() == pytest.approx(ev["release_t"], abs=1e-6)
    assert (res.trajectory.loc[res.trajectory["t"] < ev["release_t"], "flags"] & FLAG_AT_REST).all()


def test_gravity_from_the_flight_is_9_81_and_the_clip_is_not_called_retimed(drop):
    _, _, _, _, res = drop
    flight = res.summary["retime_check"]["flights"][0]
    assert flight["g_fit"] == pytest.approx(9.81, rel=0.03)
    assert res.summary["retime_check"]["suspected_factor"] is None
    assert "looks_retimed" not in {n["code"] for n in res.summary["notices"]}


def test_a_clip_whose_time_base_is_4x_too_slow_is_flagged():
    _, cal, track, _, _ = analytic_tracks("free_fall", 2.0, fps=120.0)
    slow = TrackSeries(
        track.frame, track.t * 4, track.cx, track.cy, track.area, track.conf, track.border
    )
    res = build_trajectory(slow, cal, "free_fall")
    assert res.summary["retime_check"]["suspected_factor"] == 4
    assert "looks_retimed" in {n["code"] for n in res.summary["notices"]}


@pytest.mark.parametrize("system,params", [
    ("free_fall", {}),
    ("pendulum", {"params": {"gamma": 0.08, "theta0": 0.6}}),
    ("ramp", {}),
    ("spin_down", {"params": {"omega0": 30.0, "coulomb": 3.0, "viscous": 0.4}}),
])  # fmt: skip
def test_noise_estimate_is_within_20_percent_of_the_true_noise(system, params):
    ratios = []
    for seed in range(4):
        _, cal, track, _, _ = analytic_tracks(system, 4.0, seed=seed, sigma_px=0.25, **params)
        res = build_trajectory(track, cal, system)
        true_m = 0.25 * m_per_px_along(cal, track)
        ratios.append(res.summary["noise"]["sigma_pos_m"] / true_m)
    assert np.abs(np.mean(ratios) - 1) < 0.1 and np.abs(np.array(ratios) - 1).max() < 0.2


def test_pendulum_angle_length_and_plumb_direction():
    sc, cal, track, _, truth = analytic_tracks(
        "pendulum", 4.0, params={"gamma": 0.08, "theta0": 0.6}
    )
    res = build_trajectory(track, cal, "pendulum")
    df, s = res.trajectory, res.summary
    fr = df["frame"].to_numpy()
    assert np.sqrt(np.mean((df["theta"].to_numpy() - truth["series"]["theta"][fr]) ** 2)) < 0.003
    p = s["pendulum"]
    assert p["pivot_source"] == "circle_fit"
    assert p["length_m"] == pytest.approx(0.6, rel=0.01)
    assert p["length_eff_m"] == pytest.approx(sc.system.L_eff, rel=0.01)
    assert p["bob_radius_m"] == pytest.approx(0.02, rel=0.1)
    assert p["pivot_xy_m"] == pytest.approx(list(sc.system.pivot), abs=0.01)
    assert s["gravity"]["source"] == "plumb" and abs(s["gravity"]["angle_deg"]) < 0.5
    assert s["angle_column"] == "theta" and s["noise"]["sigma_angle_rad"] > 0
    assert {"theta", "omega"} <= set(df.columns)


def test_a_tracked_pivot_gives_the_length_without_a_circle_fit():
    sc, cal, track, _, _ = analytic_tracks("pendulum", 3.0, params={"theta0": 0.1})
    _, _, pivot, _, _ = analytic_tracks(
        "pendulum", 3.0, key="pivot", params={"theta0": 0.1}, seed=4
    )
    res = build_trajectory(track, cal, "pendulum", pivot=pivot)
    p = res.summary["pendulum"]
    assert p["pivot_source"] == "tracked" and p["sigma_length_m"] is None
    assert p["length_m"] == pytest.approx(0.6, rel=0.01)
    assert "pivot_unconstrained" not in {n["code"] for n in res.summary["notices"]}


def test_a_small_swing_without_a_pivot_track_warns_that_the_pivot_is_unconstrained():
    _, cal, track, _, _ = analytic_tracks("pendulum", 3.0, params={"theta0": 0.1})
    res = build_trajectory(track, cal, "pendulum")
    assert "pivot_unconstrained" in {n["code"] for n in res.summary["notices"]}


def test_bounce_is_split_into_flights_with_two_frames_dropped_around_each_impact():
    sc, cal, track, _, truth = analytic_tracks("bounce", 3.0, fps=120.0)
    res = build_trajectory(track, cal, "bounce")
    df, s = res.trajectory, res.summary
    found = s["events"]["impacts_t"]
    assert len(found) >= 4
    for got, want in zip(found, sc.system.impacts, strict=False):
        assert got == pytest.approx(want, abs=2 / 120)
    assert s["n_dropped"]["impact"] == 5 * len(found)
    assert df["segment_id"].max() == len(found) and s["events"]["n_flights"] == len(found) + 1
    assert (df["segment_id"].diff().dropna() >= 0).all()
    assert (df["flags"] & FLAG_NEAR_IMPACT).sum() >= len(found)
    assert s["gravity"]["source"] == "data"
    flights = s["retime_check"]["flights"]
    assert all(f["g_fit"] == pytest.approx(9.81, rel=0.05) for f in flights[:4])
    gaps = np.diff(df["frame"].to_numpy())
    assert gaps.max() >= 5  # the dropped frames stay gaps, nothing is interpolated


def test_spin_down_angle_is_unwrapped_and_matches_the_truth():
    spec = {"params": {"omega0": 30.0, "coulomb": 3.0, "viscous": 0.4}}
    sc, cal, track, _, truth = analytic_tracks("spin_down", 4.0, **spec)
    _, _, hub, _, _ = analytic_tracks("spin_down", 4.0, key="hub", seed=9, **spec)
    res = build_trajectory(track, cal, "spin_down", hub=hub)
    df, s = res.trajectory, res.summary
    fr = df["frame"].to_numpy()
    err = df["phi"].to_numpy() - truth["series"]["phi"][fr]
    err -= np.median(err)  # the angle origin depends on the frame's orientation
    assert np.sqrt(np.mean(err**2)) < 0.01
    assert s["spinner"]["direction"] == "ccw" and s["spinner"]["direction_consistent"]
    assert s["spinner"]["total_rotation_rev"] > 5
    assert s["spinner"]["hub_source"] == "tracked"
    assert not (df["flags"] & FLAG_ALIAS_RISK).any()
    assert s["angle_column"] == "phi" and "theta" not in df.columns


def test_spinner_hub_can_come_from_a_circle_fit_of_the_marker_path():
    sc, cal, track, _, truth = analytic_tracks(
        "spin_down", 4.0, params={"omega0": 30.0, "coulomb": 3.0, "viscous": 0.4}
    )
    res = build_trajectory(track, cal, "spin_down")
    assert res.summary["spinner"]["hub_source"] == "circle_fit"
    assert res.summary["spinner"]["marker_radius_m"] == pytest.approx(0.035, rel=0.03)
    assert res.summary["spinner"]["hub_xy_m"] == pytest.approx(list(sc.system.hub), abs=0.003)


def test_a_spinner_too_fast_for_the_frame_rate_is_flagged():
    _, cal, track, _, _ = analytic_tracks(
        "spin_down", 1.0, sigma_px=0.05, params={"omega0": 150.0, "coulomb": 5.0, "viscous": 0.1}
    )
    res = build_trajectory(track, cal, "spin_down")
    assert res.summary["spinner"]["max_step_deg_per_frame"] > 120
    assert (res.trajectory["flags"] & FLAG_ALIAS_RISK).any()
    assert "aliasing_risk" in {n["code"] for n in res.summary["notices"]}


def board_mask(sc):
    pts = np.round(sc.cam.project(sc.system.static()[0].pts) * 16).astype(np.int32)
    mask = np.zeros((480, 854), np.uint8)
    cv2.fillPoly(mask, [pts], 1, cv2.LINE_8, 4)
    return mask


def test_ramp_has_an_along_incline_coordinate_and_is_checked_against_the_board():
    sc, cal0, track, _, truth = analytic_tracks("ramp", 2.0)
    cal = calibrate(
        SIZE, corners_px=order_corners(sc.corners_px), paper="A4", ramp_mask=board_mask(sc)
    )
    res = build_trajectory(track, cal, "ramp")
    df, ramp = res.trajectory, res.summary["ramp"]
    moving = df[(df["flags"] & FLAG_AT_REST) == 0]
    assert (np.diff(moving["s"].to_numpy()) > -0.002).all() and moving["s"].iloc[-1] > 0.5
    assert ramp["path_angle_deg"] == pytest.approx(-20, abs=0.5)
    assert ramp["board_angle_deg"] == pytest.approx(-20, abs=0.7)
    assert ramp["angle_diff_deg"] < 1.0
    assert "ramp_angle_disagreement" not in {n["code"] for n in res.summary["notices"]}


def test_bad_frames_are_dropped_not_filled_and_counted():
    sc, cal, track, _, truth = analytic_tracks("free_fall", 2.0, fps=120.0)
    track.conf[[30, 31]] = 0.2
    track.border[[100]] = True
    track.cy[150] += 80
    res = build_trajectory(track, cal, "free_fall")
    drops = res.summary["n_dropped"]
    assert drops == {"low_conf": 2, "border": 1, "jump": 1}
    assert res.summary["n_points"] == 240 - 4
    assert not {30, 31, 100, 150} & set(res.trajectory["frame"])
    assert set(res.trajectory["t"]) <= set(track.t)
    assert (res.frames["drop"] != "").sum() == 4
    assert res.trajectory["weight"].between(0, 1).all()


def test_out_of_plane_motion_is_flagged_but_perspective_alone_is_not():
    _, cal, track, _, _ = analytic_tracks("pendulum", 4.0, params={"gamma": 0.05, "theta0": 0.5})
    steady = build_trajectory(track, cal, "pendulum")
    assert not steady.summary["out_of_plane"]["flag"]
    assert steady.summary["uncertainty"]["out_of_plane_rel"] == 0.0
    _, cal, track, _, _ = analytic_tracks(
        "pendulum", 4.0, params={"gamma": 0.05, "theta0": 0.5, "dz_rate": 0.12}
    )
    toward = build_trajectory(track, cal, "pendulum")
    oop = toward.summary["out_of_plane"]
    assert oop["flag"] and oop["dz_over_z"] > 0.02
    assert "out_of_plane" in {n.code for n in toward.notices}
    assert toward.summary["uncertainty"]["out_of_plane_rel"] == pytest.approx(oop["dz_over_z"])


def test_too_few_usable_frames_is_an_error():
    _, cal, track, _, _ = analytic_tracks("free_fall", 0.5, fps=60.0)
    track.conf[:] = 0.2
    with pytest.raises(TrajectoryError):
        build_trajectory(track, cal, "free_fall")


def test_outputs_have_the_columns_and_summary_the_physics_code_expects(drop, tmp_path):
    _, _, _, _, res = drop
    res.write(tmp_path)
    df = pd.read_parquet(tmp_path / "trajectory.parquet")
    assert list(df.columns[:3]) == ["t", "x", "y"]
    assert {"t", "x", "y", "weight", "flags", "segment_id"} <= set(df.columns)
    assert df["t"].is_monotonic_increasing and df["segment_id"].eq(0).all()
    s = json.loads((tmp_path / "traj_summary.json").read_text())
    for key in ("n_points", "duration_s", "noise", "events", "ranges", "out_of_plane", "samples",
                "uncertainty", "gravity", "units", "motion_window"):  # fmt: skip
        assert key in s, key
    assert 50 <= len(s["samples"]["t"]) <= 60 and set(s["samples"]) == {"t", "x", "y"}
    assert s["noise"]["sigma_pos_m"] > 0 and s["uncertainty"]["scale_rel_at_object"] > 0
    assert s["n_points"] == len(df)


def test_a_rendered_clip_goes_through_masks_calibration_and_trajectory():
    gt = clip(system="free_fall", size=(640, 360), duration_s=1.6, capture_fps=60.0)
    gray = cv2.cvtColor(gt.plate, cv2.COLOR_BGR2GRAY)
    cal = calibrate(gt.size, sheet_mask=gt.sheet_mask, gray=gray, paper="A4")
    track = tracks_from_truth(gt, sigma_px=0.1)
    res = build_trajectory(track, cal, "free_fall")
    err = pos_error(res, gt.track("object")["xyz_m"])
    assert np.sqrt((err**2).sum(axis=1).mean()) < 0.006
    assert res.summary["events"]["release_t"] == pytest.approx(0.4, abs=0.1)
    assert res.summary["retime_check"]["flights"][0]["g_fit"] == pytest.approx(9.81, rel=0.08)
