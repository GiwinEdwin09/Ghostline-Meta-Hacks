"""The ghost sprite and the overlay renderer (build plan Part 16)."""

import time

import av
import cv2
import numpy as np
import pytest

from ghostline.ghost.render import (
    Annotator,
    RenderPlan,
    _text_layer,
    delta_text,
    render_ghost_video,
    render_stills,
)
from ghostline.ghost.sprite import (
    CYAN_BGR,
    GLOW_OPACITY,
    OPACITY,
    OUTLINE_BGR,
    build_sprite,
    disc_sprite,
    draw_sprite,
    pick_frame,
    rank_frames,
)

ORANGE = (40, 120, 250)  # BGR


def disc_image(radius=14, size=(120, 160), colour=ORANGE, centre=(80, 60)):
    image = np.full((*size, 3), 70, np.uint8)
    mask = np.zeros(size, np.uint8)
    cv2.circle(image, centre, radius, colour, -1)
    cv2.circle(mask, centre, radius, 1, -1)
    return image, mask


# --- the sprite ----------------------------------------------------------------------------
def test_the_sprite_is_feathered_tinted_halfway_to_cyan_and_translucent():
    image, mask = disc_image()
    sprite = build_sprite(image, mask, source_frame=7)
    assert sprite is not None and sprite.source_frame == 7
    # opacity 0.55 at the heart of the object, fading to nothing over the 2 px feather
    cy, cx = (int(round(v)) for v in sprite.centre[::-1])
    assert sprite.alpha[cy, cx] == pytest.approx(OPACITY, abs=0.01)
    edge = sprite.alpha[cy, cx + 14]  # on the mask's edge: about half
    assert 0.2 * OPACITY < edge < 0.8 * OPACITY
    assert sprite.alpha[cy, cx + 24] == pytest.approx(0.0, abs=0.01)  # well outside
    # colour: half the object's own colour, half cyan
    expected = 0.5 * np.array(ORANGE, float) + 0.5 * CYAN_BGR
    assert sprite.color[cy, cx] == pytest.approx(expected, abs=1.0)
    assert sprite.radius_px == pytest.approx(14, abs=0.5)
    # the crop starts 14 px (the glow margin) before the disc's left and top edges
    assert sprite.centre == pytest.approx((80 - 66 + 14, 60 - 46 + 14), abs=0.6)


def test_the_glow_is_a_dilated_blurred_alpha_at_a_quarter_opacity_and_reaches_past_the_object():
    image, mask = disc_image()
    sprite = build_sprite(image, mask, source_frame=0)
    assert sprite is not None
    cy, cx = (int(round(v)) for v in sprite.centre[::-1])
    assert sprite.glow.max() == pytest.approx(GLOW_OPACITY, abs=0.01)
    outside = cx + 14 + 4  # four pixels beyond the mask's edge, where the sprite itself is gone
    assert sprite.alpha[cy, outside] < 0.02 < sprite.glow[cy, outside]


def test_an_empty_mask_gives_no_sprite_and_a_disc_stands_in_when_there_is_no_mask():
    assert (
        build_sprite(np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20), np.uint8), source_frame=0)
        is None
    )
    disc = disc_sprite(9.0, (10.0, 200.0, 10.0))
    assert disc is not None and disc.radius_px == pytest.approx(9.0, abs=0.7)


class FakeTrack:
    """The parts of vision.Track the sprite picker reads."""

    def __init__(self, conf, speed, border=None):
        n = len(conf)
        self.frame = np.arange(n)
        self.cx = np.cumsum(speed)
        self.cy = np.zeros(n)
        self.area = np.full(n, 600.0)
        self.conf = np.asarray(conf, float)
        self.touches_border = np.zeros(n, bool) if border is None else np.asarray(border)

    def mask(self, frame):
        return np.ones((8, 8), np.uint8)


def test_the_picker_prefers_high_confidence_then_the_slowest_frame():
    conf = [0.90, 0.99, 0.99, 0.99, 0.70, 0.99]
    speed = [3.0, 9.0, 0.5, 4.0, 0.1, 6.0]
    track = FakeTrack(conf, speed)
    ranked = rank_frames(track)
    assert ranked[0] == 2  # confident and slowest of the confident
    assert 4 not in ranked[:4]  # slowest of all, but the least confident of the best group
    # a frame touching the border (object cut off) never wins
    cut = FakeTrack(conf, speed, border=[False, False, True, False, False, False])
    assert rank_frames(cut)[0] != 2
    # restricted to a window of frames
    assert set(rank_frames(track, range(3, 6))) <= {3, 4, 5}


def test_the_picker_takes_the_sharpest_of_the_best_frames():
    track = FakeTrack([0.99, 0.99, 0.99], [1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2)
    images = {0: blurred, 1: sharp, 2: blurred}
    assert pick_frame(track, images, [0, 1, 2]) == 1
    assert pick_frame(track, {}, [0, 1, 2]) is None


# --- drawing it ----------------------------------------------------------------------------
def test_the_ghost_is_composited_translucently_with_an_outline_in_the_status_colour():
    image, mask = disc_image(radius=14)
    sprite = build_sprite(image, mask, source_frame=0)
    frame = np.full((200, 300, 3), 60, np.uint8)
    before = frame.copy()
    draw_sprite(frame, sprite, (150.0, 100.0), outline="amber")
    # inside the disc: the background shows through at 45%, the tinted object at 55%
    expect = 0.45 * 60 + 0.55 * (0.5 * np.array(ORANGE) + 0.5 * CYAN_BGR)
    assert frame[100, 150] == pytest.approx(expect, abs=6)
    # the outline ring is amber (BGR) just outside the body
    ring = frame[100, 150 + 14]
    assert np.abs(ring.astype(float) - OUTLINE_BGR["amber"]).max() < 60
    far = frame[10:20, 10:20]
    assert (far == before[10:20, 10:20]).all()  # nothing drawn away from the ghost


def test_a_spinner_sprite_turns_by_the_predicted_angle():
    mask = np.zeros((80, 80), np.uint8)
    cv2.rectangle(mask, (20, 36), (60, 44), 1, -1)  # a horizontal bar
    image = np.full((80, 80, 3), 120, np.uint8)
    sprite = build_sprite(image, mask, source_frame=0)

    def tilt(angle):
        """Direction of the bar's long axis on screen, in degrees (image y points down)."""
        frame = np.zeros((160, 160, 3), np.uint8)
        draw_sprite(frame, sprite, (80.0, 80.0), angle)
        ys, xs = np.nonzero(frame.sum(axis=2) > 200)
        pts = np.column_stack([xs, ys]).astype(float)
        pts -= pts.mean(axis=0)
        major = np.linalg.eigh(pts.T @ pts)[1][:, -1]
        deg = np.degrees(np.arctan2(major[1], major[0]))
        return (deg + 90) % 180 - 90

    assert abs(tilt(0.0)) < 5
    assert abs(abs(tilt(90.0)) - 90) < 8  # counter-clockwise quarter turn: vertical
    assert abs(tilt(30.0) + 30) < 8  # counter-clockwise on screen: y grows downward


def test_the_delta_label_and_its_glyph():
    assert delta_text(4.2) == "Δ 4.2 mm"
    assert delta_text(18.4) == "Δ 18 mm"
    wide = _text_layer("Δ 4 mm", (255, 255, 255), 20)[1]
    narrow = _text_layer(" 4 mm", (255, 255, 255), 20)[1]
    assert wide.shape[1] > narrow.shape[1] + 5 and wide.max() == 1.0  # the delta is drawn


# --- the overlay on a frame ----------------------------------------------------------------
def make_plan(n=90, fps=30.0, launch_end=1.0, slow_motion=False):
    t = np.arange(n) / fps
    px = np.column_stack([100 + 3.0 * np.arange(n), 120 + 0.0 * t])
    image, mask = disc_image(radius=8)
    sprite = build_sprite(image, mask, source_frame=0)
    error = np.linspace(1.0, 30.0, n)
    status = ["green" if e < 10 else "amber" if e < 20 else "red" for e in error]
    return RenderPlan(
        frames=np.arange(10, 10 + n), t_rel=t, predicted_px=px + [0, 6], actual_px=px,
        error_mm=error, status=status, sprite=sprite, launch_end_rel=launch_end,
        label="Ghost v1 · P1", anchor_px=np.array([200.0, 20.0]), slow_motion=slow_motion,
    )  # fmt: skip


def hud_pixels(frame, bgr):
    """Count pixels in the top-left HUD that are close to a colour."""
    region = frame[:70, :260].astype(int)
    return int((np.abs(region - np.array(bgr)).sum(axis=2) < 40).sum())


def test_the_overlay_draws_the_trail_the_dotted_path_and_the_hud():
    plan = make_plan()
    ann = Annotator(plan, 640, 360)
    frame = ann.draw(np.full((360, 640, 3), 40, np.uint8), 60)
    # a thin white trail of the last half second, ending at the real object
    trail = frame[plan.actual_px[60, 1].astype(int) - 1 : plan.actual_px[60, 1].astype(int) + 2]
    assert (trail.min(axis=2) > 200).any()
    # dotted cyan path ahead of the ghost (blue and green high, red low)
    cyan = (frame[..., 0] > 200) & (frame[..., 1] > 200) & (frame[..., 2] < 80)
    assert cyan.sum() > 20
    # "predicting 2.0 s ahead" once past the launch window, and no launch label
    assert hud_pixels(frame, (255, 255, 255)) > 50
    assert hud_pixels(frame, (250, 139, 167)) == 0


def test_the_launch_window_label_shows_over_the_first_frames_only():
    plan = make_plan(launch_end=1.0)
    ann = Annotator(plan, 640, 360)
    early = ann.draw(np.full((360, 640, 3), 40, np.uint8), 5)  # t = 0.17 s
    late = ann.draw(np.full((360, 640, 3), 40, np.uint8), 60)  # t = 2 s
    violet = (250, 139, 167)
    assert hud_pixels(early, violet) > 100
    assert hud_pixels(late, violet) == 0


# --- the video -----------------------------------------------------------------------------
def plain_video(path, n, size=(640, 360), fps=30.0, pts=None):
    w, h = size
    container = av.open(str(path), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    stream.codec_context.options = {"crf": "20", "preset": "ultrafast", "bf": "0"}
    for i in range(n):
        frame = np.full((h, w, 3), 50, np.uint8)
        cv2.circle(frame, (100 + 3 * i % (w - 200), 120), 8, (40, 120, 250), -1)
        out = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in stream.encode(out):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_a_slow_motion_clip_also_gets_a_30_fps_slow_playback_version(tmp_path):
    master = tmp_path / "master.mp4"
    plain_video(master, 120, fps=30.0)
    # the clip was captured at 240 fps: real time runs 8x faster than the file's clock
    plan = make_plan(n=100, fps=240.0, slow_motion=True)
    plan.frames = np.arange(5, 105)
    stats = render_ghost_video(master, plan, tmp_path / "v1.mp4", tmp_path / "v1_slow.mp4")
    assert stats.n_frames == 100
    with av.open(str(tmp_path / "v1_slow.mp4")) as c:
        s = c.streams.video[0]
        assert float(s.average_rate) == 30 and sum(1 for _ in c.decode(s)) == 100
    with av.open(str(tmp_path / "v1.mp4")) as c:
        s = c.streams.video[0]
        times = [float(f.pts * s.time_base) for f in c.decode(s)]
    assert times[-1] == pytest.approx(99 / 240.0, abs=1e-4)  # real seconds, not file seconds


def test_stills_come_back_annotated_for_the_frames_asked_for(tmp_path):
    master = tmp_path / "master.mp4"
    plain_video(master, 60)
    plan = make_plan(n=40)
    stills = render_stills(master, plan, [3, 20])
    assert set(stills) == {3, 20} and stills[3].shape == (360, 640, 3)
    assert not (stills[3] == stills[20]).all()


def test_ten_seconds_of_720p_video_renders_well_inside_the_60_second_budget(tmp_path):
    master = tmp_path / "master.mp4"
    plain_video(master, 600, size=(1280, 720), fps=60.0)
    n = 600
    plan = make_plan(n=n, fps=60.0)
    plan.frames = np.arange(n)
    plan.launch_end_rel = 1.0
    started = time.monotonic()
    stats = render_ghost_video(master, plan, tmp_path / "v1.mp4")
    assert stats.n_frames == 600 and time.monotonic() - started < 60.0
