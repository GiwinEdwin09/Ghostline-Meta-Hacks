import hashlib
import json
import subprocess

import cv2
import numpy as np
import pytest
from synth_helpers import HAVE_FFMPEG, clip, have_encoder, tmp_dir

from ghostline.ingest import (
    DeclaredCapture,
    IngestError,
    assemble_chunks,
    ingest_trial,
    median_image,
    probe_file,
    save_stills,
    trim_master,
)
from ghostline.ingest.models import ProbeInfo
from ghostline.ingest.normalize import master_filters

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg and ffprobe")

BALL = {"radius": 0.035}
SMALL = {"size": (320, 180), "params": BALL}


@pytest.fixture(scope="module")
def plain():
    gt = clip(system="free_fall", duration_s=1.5, **SMALL)
    return gt, ingest_trial(gt.video_path, tmp_dir("plain"), DeclaredCapture(mode="normal", fps=60))


def test_normal_clip_gives_an_identical_master_and_real_timestamps(plain):
    gt, res = plain
    assert res.normalize.n_frames == res.probe.nb_frames == gt.n_frames
    assert res.timing.classification == "cfr" and res.timing.time_factor == 1.0
    assert np.allclose(res.timestamps, gt.t_real - gt.t_real[0], atol=1e-5)
    assert res.normalize.max_pts_error_s < 1e-5
    master = probe_file(res.master).info
    assert (master.display_width, master.display_height, master.rotation) == (320, 180, 0)
    assert master.hdr is None and master.pix_fmt == "yuv420p"
    assert res.sha256 == hashlib.sha256((res.trial_dir / "original.mp4").read_bytes()).hexdigest()


def test_outputs_are_all_written(plain):
    _, res = plain
    names = ("original.mp4", "master.mp4", "playback_720.mp4", "wonder_480.mp4", "poster.jpg")
    names += ("probe.json", "timing.json", "timestamps.npy", "trim.json", "ingest.json")
    for name in names:
        assert (res.trial_dir / name).stat().st_size > 0, name
    assert not (res.trial_dir / "audio.wav").exists()  # the clip has no audio
    assert json.loads((res.trial_dir / "timing.json").read_text())["time_factor"] == 1.0
    assert np.load(res.trial_dir / "timestamps.npy").shape == (90,)
    wonder = probe_file(res.trial_dir / "wonder_480.mp4")
    assert wonder.info.nb_frames <= 16  # 1.5 s at 10 fps
    assert min(wonder.info.display_width, wonder.info.display_height) <= 480


def test_trim_suggestion_brackets_the_motion(plain):
    gt, res = plain
    release, stop = (round(gt.events[k] * 60) for k in ("release_t_s", "stop_t_s"))
    assert res.trim.reason == "motion_window"
    assert res.trim.start_frame <= release and res.trim.stop_frame >= stop
    assert res.trim.stop_frame - res.trim.start_frame < gt.n_frames


def test_trimmed_master_has_exactly_the_requested_frames(plain):
    _, res = plain
    out = trim_master(res.master, res.trial_dir / "trimmed.mp4", 10, 40)
    assert out == 30 == probe_file(res.trial_dir / "trimmed.mp4").info.nb_frames


def test_median_image_removes_the_moving_ball(plain):
    gt, res = plain
    med = median_image(res.master, res.normalize.n_frames, 15)
    x, y = np.round(gt.track("object")["centroid_px"][45]).astype(int)
    b, g, r = med[y, x].astype(int)
    assert not (r > 180 and b < 120)  # not orange: the ball is gone
    stills = save_stills(res.master, [0, 45], res.trial_dir / "stills")
    assert [p.name for p in stills] == ["still_00000.jpg", "still_00045.jpg"]


def test_retimed_slow_motion_is_given_the_right_time_base():
    gt = clip(system="free_fall", duration_s=1.2, capture_fps=240, export_fps=30, **SMALL)
    res = ingest_trial(gt.video_path, tmp_dir("slowmo"), DeclaredCapture(mode="slow_mo", fps=240))
    tm = res.timing
    assert tm.retimed and tm.needs_confirmation and res.needs_confirmation
    assert tm.time_factor == pytest.approx(0.125) and tm.real_fps == pytest.approx(240)
    assert np.allclose(res.timestamps, gt.t_real, atol=1e-4)
    assert res.normalize.n_frames == gt.n_frames == 288
    release, stop = (round(gt.events[k] * 240) for k in ("release_t_s", "stop_t_s"))
    assert res.trim.start_frame <= release and res.trim.stop_frame >= stop
    assert "retimed" in {w.code for w in res.warnings}


def test_vfr_clip_keeps_its_timestamps():
    gt = clip(system="free_fall", duration_s=1.5, pts_jitter_frac=0.04, **SMALL)
    res = ingest_trial(gt.video_path, tmp_dir("vfr"), DeclaredCapture(mode="normal", fps=60))
    assert res.timing.classification == "vfr_jitter"
    assert np.allclose(res.timestamps, gt.pts - gt.pts[0], atol=1e-4)
    assert res.normalize.max_pts_error_s < 1e-4


def test_mixed_rate_clip_is_restricted_to_the_uniform_section():
    gt = clip(
        system="free_fall", duration_s=1.0, capture_fps=240, export_fps=30,
        mixed_ramps=(40, 40, 240.0), **SMALL,
    )  # fmt: skip
    res = ingest_trial(gt.video_path, tmp_dir("mixed"), DeclaredCapture(mode="slow_mo", fps=240))
    tm = res.timing
    assert tm.classification == "mixed" and tm.needs_confirmation
    assert tm.usable_start == 40 and tm.usable_stop in (200, 201)
    assert tm.time_factor == pytest.approx(0.125)
    assert tm.usable_start <= res.trim.start_frame < res.trim.stop_frame <= tm.usable_stop
    mid = slice(40, 200)
    assert np.allclose(np.diff(res.timestamps[mid]), 1 / 240, atol=1e-4)


@pytest.mark.skipif(not have_encoder("libx265"), reason="needs ffmpeg with libx265")
def test_hdr_portrait_clip_becomes_upright_sdr():
    gt = clip(
        system="free_fall", duration_s=1.5, size=(180, 320), params=BALL, hdr=True, rotation_tag=90
    )
    res = ingest_trial(gt.video_path, tmp_dir("hdr"), DeclaredCapture(mode="normal", fps=60))
    p = res.probe
    assert p.hdr == "hlg" and p.rotation == 90 and p.codec == "hevc"
    assert (p.width, p.height, p.display_width, p.display_height) == (320, 180, 180, 320)
    master = probe_file(res.master).info
    assert (master.display_width, master.display_height) == (180, 320)
    assert master.rotation == 0 and master.hdr is None and master.pix_fmt == "yuv420p"
    assert res.normalize.n_frames == gt.n_frames
    assert res.normalize.tonemap in ("zscale_hable", "colorspace_fallback")
    # upright: the bright sheet sits where the truth says
    med = cv2.cvtColor(median_image(res.master, gt.n_frames, 9), cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(med > 150)
    truth = gt.sheet_corners_px[0].mean(axis=0)
    assert np.allclose([xs.mean(), ys.mean()], truth, atol=4)
    assert "hdr_tonemapped" in {w.code for w in res.warnings}


def test_a_very_dark_clip_is_warned_about():
    gt = clip(system="free_fall", duration_s=1.0, **SMALL)
    dark = tmp_dir("dark_src") / "dark.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(gt.video_path), "-vf",
         "eq=brightness=-0.5", str(dark)], check=True,
    )  # fmt: skip
    res = ingest_trial(dark, tmp_dir("dark"))
    assert "dark_clip" in {w.code for w in res.warnings}


def test_a_file_without_video_is_rejected(tmp_path):
    bad = tmp_path / "notes.mp4"
    bad.write_text("not a video")
    with pytest.raises(IngestError):
        ingest_trial(bad, tmp_path / "trial")


def test_hdr_chain_uses_zscale_when_available_and_a_fallback_otherwise():
    probe = ProbeInfo(
        codec="hevc", width=3840, height=2160, display_width=3840, display_height=2160,
        nb_frames=10, hdr="hlg", color_transfer="arib-std-b67",
    )  # fmt: skip
    with_z, kind = master_filters(probe, zscale=True)
    assert (
        kind == "zscale_hable" and "zscale=t=linear" in with_z and "tonemap=tonemap=hable" in with_z
    )
    without, kind = master_filters(probe, zscale=False)
    assert kind == "colorspace_fallback" and "zscale" not in without and "colorspace=" in without
    assert "min(1920" in with_z  # long side capped
    sdr, kind = master_filters(probe.model_copy(update={"hdr": None}), zscale=True)
    assert kind == "none" and "zscale" not in sdr


def test_chunks_are_assembled_in_order_with_their_hash(tmp_path):
    parts = []
    for i, data in enumerate([b"abc", b"defg", b"h"]):
        p = tmp_path / f"{i}.part"
        p.write_bytes(data)
        parts.append(p)
    digest = assemble_chunks(parts, tmp_path / "original.mp4")
    assert (tmp_path / "original.mp4").read_bytes() == b"abcdefgh"
    assert digest == hashlib.sha256(b"abcdefgh").hexdigest()
