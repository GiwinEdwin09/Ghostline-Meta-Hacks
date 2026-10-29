import numpy as np
import pandas as pd

from ghostline.traj.clean import area_outliers, clean_track, jump_outliers
from ghostline.traj.model import FLAG_BORDER, TrackSeries


def line_track(n=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60
    cx = 100 + 4.0 * np.arange(n) + rng.normal(0, 0.2, n)
    cy = 50 + 0.5 * np.arange(n) ** 1.2 + rng.normal(0, 0.2, n)
    area = 200 + rng.normal(0, 2, n)
    conf = np.full(n, 0.95)
    border = np.zeros(n, bool)
    return TrackSeries.from_arrays(np.arange(n), cx, cy, area, conf, border, t=t)


def test_clean_data_is_all_kept():
    frames = clean_track(line_track())
    assert (frames["drop"] == "").all()
    assert list(frames.columns) == ["frame", "t", "cx", "cy", "area", "conf", "drop"]


def test_low_confidence_and_border_frames_are_dropped_with_their_reason():
    tr = line_track()
    tr.conf[[10, 11]] = 0.3
    tr.border[[20, 11]] = True  # frame 11 is both: confidence wins
    drop = clean_track(tr)["drop"]
    assert list(drop[[10, 11]]) == ["low_conf", "low_conf"]
    assert drop[20] == "border"
    assert (drop == "").sum() == 120 - 3


def test_a_confidence_exactly_at_the_limit_is_kept():
    tr = line_track()
    tr.conf[5] = 0.5
    assert clean_track(tr)["drop"][5] == ""


def test_area_outliers_use_a_rolling_median():
    area = 200 + np.random.default_rng(0).normal(0, 2, 120)
    area[40] = 400
    area[41] = 60
    flagged = area_outliers(area)
    assert flagged[40] and flagged[41] and flagged.sum() == 2


def test_a_steady_area_trend_is_not_an_outlier():
    area = 200 + 0.5 * np.arange(120) + np.random.default_rng(1).normal(0, 2, 120)
    assert not area_outliers(area).any()


def test_nan_areas_are_never_flagged():
    area = np.full(50, np.nan)
    assert not area_outliers(area).any()
    track = line_track(50)
    track.area[:] = np.nan
    assert (clean_track(track)["drop"] == "").all()


def test_centroid_spikes_and_short_runs_are_dropped_but_motion_is_not():
    tr = line_track()
    tr.cx[30] += 60  # one frame jumps to another object
    tr.cy[70:72] -= 80  # two frames
    drop = clean_track(tr)["drop"]
    assert drop[30] == "jump" and drop[70] == "jump" and drop[71] == "jump"
    assert (drop == "jump").sum() == 3


def test_a_permanent_move_is_not_a_jump():
    px = np.column_stack([np.arange(100) * 2.0, np.zeros(100)])
    px[50:, 1] += 40  # the track switches to a parallel path for good
    assert not jump_outliers(px)[[51, 52, 60]].any()


def test_fast_circular_motion_and_bounce_kinks_are_not_jumps():
    ang = np.arange(200) * 1.0  # 57 degrees per frame
    circle = 4.5 * np.column_stack([np.cos(ang), np.sin(ang)]) + 300
    assert not jump_outliers(circle).any()
    y = np.abs(np.arange(-30, 30)) * 6.0  # a V: the impact
    kink = np.column_stack([np.zeros(60), y])
    assert not jump_outliers(kink).any()


def test_motion_at_rest_is_not_flagged_by_a_noise_sized_step():
    px = np.random.default_rng(2).normal(0, 0.1, (150, 2)) + [50, 50]
    assert not jump_outliers(px).any()


def test_nothing_is_interpolated_dropped_frames_stay_gaps():
    tr = line_track()
    tr.conf[[10, 11, 12]] = 0.1
    frames = clean_track(tr)
    kept = frames[frames["drop"] == ""]
    assert len(kept) == 117 and not {10, 11, 12} & set(kept["frame"])
    assert np.array_equal(kept["cx"], tr.cx[frames["drop"] == ""])


def test_track_series_sorts_validates_and_reads_a_tracks_table():
    tr = TrackSeries.from_arrays([3, 1, 2], [3.0, 1, 2], [0.0, 0, 0], t=[0.3, 0.1, 0.2])
    assert list(tr.frame) == [1, 2, 3] and list(tr.cx) == [1, 2, 3] and not np.isnan(tr.conf).any()
    df = pd.DataFrame(
        {
            "object_key": ["ball"] * 3 + ["pivot"], "frame": [0, 1, 2, 0], "t": [0, 0.1, 0.2, 0],
            "cx": [1.0, 2, 3, 9], "cy": [0.0] * 4, "area": [10.0] * 4, "conf": [0.9] * 4,
            "flags": [0, FLAG_BORDER, 0, 0],
        }
    )  # fmt: skip
    ball = TrackSeries.from_dataframe(df, "ball")
    assert len(ball) == 3 and list(ball.border) == [False, True, False]
    from_ts = TrackSeries.from_arrays([0, 2], [1.0, 2], [0.0, 0], timestamps=[0.0, 0.5, 1.0])
    assert list(from_ts.t) == [0.0, 1.0]
    import pytest

    with pytest.raises(ValueError):
        TrackSeries.from_arrays([0, 1], [1.0], [0.0, 0.0], t=[0, 1])
