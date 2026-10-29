import av
import cv2
import numpy as np
import pytest
from synth.systems import Bounce, FreeFall, Pendulum, Ramp, SpinDown
from synth_helpers import clip, have_encoder, scene

G = 9.81


def test_free_fall_truth_is_the_analytic_parabola():
    s = FreeFall(y0=0.5, t_release=0.4, y_floor=None)
    t = np.array([0.1, 0.4, 0.7, 1.0])
    y = s.truth(t)["tracks"]["object"][:, 1]
    assert np.allclose(y, 0.5 - 0.5 * G * np.maximum(t - 0.4, 0) ** 2, atol=1e-7)


def test_free_fall_with_drag_is_slower_and_stops_at_the_floor():
    free, drag = FreeFall(y_floor=None), FreeFall(drag=2.0, y_floor=None)
    t = np.array([1.2])
    assert drag.truth(t)["tracks"]["object"][0, 1] > free.truth(t)["tracks"]["object"][0, 1]
    floor = FreeFall(y_floor=-0.55)
    assert floor.truth(np.array([5.0]))["tracks"]["object"][0, 1] == pytest.approx(-0.53, abs=1e-6)
    assert 0.8 < floor.events()["stop_t_s"] < 0.9


def test_pendulum_period_matches_small_angle_formula():
    p = Pendulum(theta0=0.05, length=0.6, radius=0.02)
    t = np.linspace(0.4, 4.4, 40001)
    theta = p.truth(t)["series"]["theta"]
    crossings = t[:-1][(theta[:-1] > 0) & (theta[1:] <= 0)]
    period = np.mean(np.diff(crossings))
    assert period == pytest.approx(2 * np.pi * np.sqrt(p.L_eff / G), rel=2e-3)


def test_pendulum_damping_shrinks_the_swing():
    p = Pendulum(theta0=0.3, gamma=0.2)
    theta = p.truth(np.linspace(0.4, 8, 4000))["series"]["theta"]
    assert np.abs(theta[-500:]).max() < 0.5 * np.abs(theta[:500]).max()


def test_ramp_acceleration_for_a_block_and_a_rolling_ball():
    a = np.radians(20)
    block = Ramp(alpha_deg=20, mu=0.2, mode="block", s0=0.0)
    ball = Ramp(alpha_deg=20, mode="ball", beta=0.4, s0=0.0)
    t = np.array([0.4, 0.9])
    for system, accel in ((block, G * (np.sin(a) - 0.2 * np.cos(a))), (ball, G * np.sin(a) / 1.4)):
        s = system.truth(t)["series"]["s"]
        assert s[1] == pytest.approx(0.5 * accel * 0.5**2, rel=1e-6)


def test_bounce_speeds_shrink_by_the_restitution():
    b = Bounce(e=0.7)
    assert np.allclose(np.array(b.speeds[1:]) / np.array(b.speeds[:-1]), 0.7)
    flights = np.diff(b.impacts)
    assert np.allclose(flights[1:] / flights[:-1], 0.7, rtol=1e-6)
    y = b.truth(np.array([b.impacts[2] + 0.001]))["tracks"]["object"][0, 1]
    assert y == pytest.approx(b.y_floor + b.r, abs=0.01)


def test_spin_down_closed_form_and_stop():
    s = SpinDown(omega0=30, coulomb=3, viscous=0.4, t_release=0.3)
    t = np.linspace(0.31, 2.0, 2000)
    phi = s.truth(t)["series"]["phi"]
    omega = np.gradient(phi, t)
    expected = (30 + 3 / 0.4) * np.exp(-0.4 * (t - 0.3)) - 3 / 0.4
    assert np.allclose(omega[5:-5], expected[5:-5], rtol=2e-3)
    late = s.truth(np.array([s.t_release + s.t_stop_rel + 1, s.t_release + s.t_stop_rel + 2]))
    assert late["series"]["phi"][0] == pytest.approx(late["series"]["phi"][1])


def test_scene_sheet_mask_matches_the_projected_quad_and_stays_in_frame():
    sc = scene(system="free_fall", azimuth_deg=20, elevation_deg=8, sheet_rot_deg=10)
    c = sc.corners_px
    shoelace = 0.5 * abs(np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1]))
    assert (sc.sheet_mask > 0).sum() == pytest.approx(shoelace, rel=0.01)
    assert c.min() > 0 and c[:, 0].max() < 853 and c[:, 1].max() < 479
    assert (
        sc.plate_u8[tuple(np.round(c.mean(axis=0)[::-1]).astype(int)) + (slice(None),)].min() > 180
    )


def test_scene_camera_has_perspective_and_a4_proportions():
    sc = scene(system="free_fall", azimuth_deg=30, elevation_deg=10)
    c = sc.corners_px
    assert np.linalg.norm(c[0] + c[2] - c[1] - c[3]) > 0.5  # not a parallelogram: keystone
    assert sc.cam.depth(sc.corners_world).min() > 1.0


@pytest.fixture(scope="module")
def drop():
    return clip(system="free_fall", size=(320, 180), duration_s=1.2, params={"radius": 0.03})


def _pts(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return np.sort([float(p.pts * s.time_base) for p in c.demux(s) if p.pts is not None])


def test_video_has_the_truth_frames_and_timestamps(drop):
    pts = _pts(drop.video_path)
    assert len(pts) == drop.n_frames == 72
    assert np.allclose(pts - pts[0], drop.pts - drop.pts[0], atol=1e-5)
    assert drop.time_factor == 1.0
    assert drop.sheet_corners_px.shape == (72, 4, 2)


def test_ball_is_drawn_where_the_truth_says(drop):
    tr = drop.track("object")
    assert tr["visible"].all()
    err = np.linalg.norm(tr["mask_centroid_px"] - tr["centroid_px"], axis=1)
    assert err.max() < 0.5
    frame = 40
    with av.open(str(drop.video_path)) as c:
        img = [f for i, f in enumerate(c.decode(video=0)) if i == frame][0].to_ndarray(
            format="bgr24"
        )
    x, y = np.round(tr["centroid_px"][frame]).astype(int)
    b, g, r = img[y, x]
    assert r > 180 and b < 120  # orange ball


def test_object_masks_round_trip(drop):
    for frame in (0, 40, 70):
        mask = drop.object_mask("object", frame)
        assert mask.sum() == drop.track("object")["area_px"][frame] > 0
        ys, xs = np.nonzero(mask)
        assert np.allclose([xs.mean(), ys.mean()], drop.track("object")["mask_centroid_px"][frame])


def test_sheet_mask_and_plate_agree(drop):
    mask = drop.sheet_mask
    gray = cv2.cvtColor(drop.plate, cv2.COLOR_BGR2GRAY)
    assert gray[mask].mean() > 200 and gray[~mask].mean() < 120


def test_slow_motion_export_retimes_the_file():
    gt = clip(system="free_fall", size=(160, 90), duration_s=0.5, capture_fps=240, export_fps=30)
    pts = _pts(gt.video_path)
    assert gt.time_factor == pytest.approx(0.125)
    assert np.allclose(np.diff(pts), 1 / 30, atol=1e-4)
    assert np.allclose(np.diff(gt.t_real), 1 / 240)


def test_jitter_and_mixed_timing_are_in_the_file():
    jit = clip(system="free_fall", size=(160, 90), duration_s=0.8, pts_jitter_frac=0.1)
    assert 0.02 < np.diff(_pts(jit.video_path)).std() / (1 / 60) < 0.2
    mixed = clip(
        system="free_fall", size=(160, 90), duration_s=0.5, capture_fps=240, export_fps=30,
        mixed_ramps=(20, 20, 240.0),
    )  # fmt: skip
    dt = np.diff(_pts(mixed.video_path))
    assert set(np.round(dt * 1000).astype(int)) == {4, 33}
    assert mixed.data["timing"]["usable_range"] == [20, 100]


def test_camera_shake_moves_the_sheet_and_the_object_together():
    gt = clip(system="free_fall", size=(160, 90), duration_s=0.5, shake_sigma_px=1.5, seed=3)
    shift = np.asarray(gt.data["camera_shift_px"])
    corners = gt.sheet_corners_px
    assert shift.std() > 0.5
    assert np.allclose(corners - corners[0], (shift - shift[0])[:, None, :], atol=1e-3)


@pytest.mark.skipif(not have_encoder("libx265"), reason="needs ffmpeg with libx265")
def test_hdr_and_rotation_tags_are_written():
    gt = clip(system="free_fall", size=(90, 160), duration_s=0.4, hdr=True, rotation_tag=90)
    import json
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-select_streams", "v:0", "-of", "json",
         str(gt.video_path)],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    stream = json.loads(out)["streams"][0]
    assert stream["codec_name"] == "hevc" and stream["color_transfer"] == "arib-std-b67"
    assert stream["side_data_list"][0]["rotation"] == 90
    assert (stream["width"], stream["height"]) == (160, 90)  # stored landscape, shown portrait
