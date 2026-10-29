import numpy as np
import pytest

from ghostline.ingest.models import DeclaredCapture, ProbeInfo
from ghostline.ingest.timing import build_timing, real_timestamps


def cfr(fps, n=200):
    return np.arange(n) / fps


def codes(timing):
    return {w.code for w in timing.warnings}


def probe(**kw):
    base = dict(
        codec="h264", width=1920, height=1080, display_width=1920, display_height=1080, nb_frames=0
    )
    return ProbeInfo(**(base | kw))


def test_constant_rate_normal_video():
    tm = build_timing(cfr(60), declared=DeclaredCapture(mode="normal", fps=60))
    assert tm.classification == "cfr"
    assert tm.time_factor == 1.0 and not tm.retimed and not tm.needs_confirmation
    assert tm.real_fps == pytest.approx(60)
    assert (tm.usable_start, tm.usable_stop) == (0, 200)


def test_ntsc_rates_count_as_matching():
    tm = build_timing(
        np.arange(200) * 1001 / 60000, declared=DeclaredCapture(mode="normal", fps=60)
    )
    assert tm.time_factor == 1.0 and "declared_fps_mismatch" not in codes(tm)


def test_vfr_jitter_keeps_real_timestamps():
    rng = np.random.default_rng(0)
    pts = np.sort(cfr(60) + rng.uniform(-0.002, 0.002, 200))
    tm = build_timing(pts, declared=DeclaredCapture(mode="normal", fps=60))
    assert tm.classification == "vfr_jitter"
    assert 0.02 < tm.jitter_rel < 0.3
    assert tm.time_factor == 1.0
    assert real_timestamps(pts, tm)[5] == pytest.approx(pts[5] - pts[0])


def test_original_slow_mo_file_is_real_time():
    tm = build_timing(cfr(240, 480), declared=DeclaredCapture(mode="slow_mo", fps=240))
    assert tm.time_factor == 1.0 and not tm.retimed and not tm.needs_confirmation
    assert tm.real_fps == pytest.approx(240)


@pytest.mark.parametrize("playback", [30, 25, 60])
def test_retimed_slow_mo_gets_a_factor_and_a_warning(playback):
    tm = build_timing(cfr(playback, 300), declared=DeclaredCapture(mode="slow_mo", fps=240))
    assert tm.retimed and tm.needs_confirmation
    assert tm.time_factor == pytest.approx(playback / 240)
    assert tm.real_fps == pytest.approx(240)
    assert "retimed" in codes(tm)
    t = real_timestamps(cfr(playback, 300), tm)
    assert np.diff(t) == pytest.approx(np.full(299, 1 / 240))


def test_android_capture_rate_tag_overrides_the_container_rate():
    tm = build_timing(cfr(30, 300), probe(capture_fps_tag=240.0))
    assert tm.time_factor == pytest.approx(30 / 240)
    assert tm.time_factor_source == "metadata_tag" and tm.retimed


def test_declared_rate_beats_the_tag_with_a_warning():
    tm = build_timing(
        cfr(30, 300), probe(capture_fps_tag=120.0), DeclaredCapture(mode="slow_mo", fps=240)
    )
    assert tm.time_factor == pytest.approx(30 / 240)
    assert tm.time_factor_source == "declared" and "capture_rate_conflict" in codes(tm)


def test_normal_declaration_against_a_slow_mo_tag_is_flagged():
    tm = build_timing(cfr(30, 300), probe(capture_fps_tag=240.0), DeclaredCapture(mode="normal"))
    assert tm.time_factor == pytest.approx(0.125) and tm.needs_confirmation
    assert "capture_rate_conflict" in codes(tm)


def test_declared_fps_that_disagrees_with_the_file_keeps_real_time():
    tm = build_timing(cfr(30), declared=DeclaredCapture(mode="normal", fps=60))
    assert tm.time_factor == 1.0 and "declared_fps_mismatch" in codes(tm)
    tm = build_timing(cfr(240), declared=DeclaredCapture(mode="slow_mo", fps=60))
    assert tm.time_factor == 1.0 and "declared_fps_mismatch" in codes(tm)


def test_slow_mo_without_a_rate_asks_for_one():
    tm = build_timing(cfr(30), declared=DeclaredCapture(mode="slow_mo"))
    assert tm.needs_confirmation and "slow_mo_fps_unknown" in codes(tm)
    assert (
        build_timing(cfr(240), declared=DeclaredCapture(mode="slow_mo")).needs_confirmation is False
    )


def test_mixed_rate_uses_the_longest_uniform_run():
    dt = np.full(299, 1 / 30)
    dt[:50] = 1 / 240
    dt[-50:] = 1 / 240
    pts = np.concatenate([[0], np.cumsum(dt)])
    tm = build_timing(pts, declared=DeclaredCapture(mode="slow_mo", fps=240))
    assert tm.classification == "mixed" and tm.needs_confirmation
    assert (tm.usable_start, tm.usable_stop) == (50, 250)
    assert tm.time_factor == pytest.approx(30 / 240)
    assert "mixed_frame_rate" in codes(tm)
    assert len(tm.clusters) == 2


def test_sporadic_dropped_frames_are_not_a_mixed_rate():
    dt = np.full(299, 1 / 60)
    dt[::25] *= 2
    pts = np.concatenate([[0], np.cumsum(dt)])
    tm = build_timing(pts, declared=DeclaredCapture(mode="normal", fps=60))
    assert tm.classification != "mixed"
    assert tm.n_gaps == 12 and "frame_gaps" in codes(tm)
    assert (tm.usable_start, tm.usable_stop) == (0, 300)


def test_uncertainty_is_larger_for_a_nominal_rate_and_for_jitter():
    exact = build_timing(cfr(60), declared=DeclaredCapture(mode="normal", fps=60))
    retimed = build_timing(cfr(30), declared=DeclaredCapture(mode="slow_mo", fps=240))
    assert retimed.sigma_time_rel > exact.sigma_time_rel
    jittery = build_timing(
        np.sort(cfr(60) + np.random.default_rng(1).uniform(-0.003, 0.003, 200)),
        declared=DeclaredCapture(mode="normal", fps=60),
    )
    assert jittery.sigma_time_rel > exact.sigma_time_rel


def test_too_few_frames_is_an_error():
    with pytest.raises(ValueError):
        build_timing(np.array([0.0, 0.1]))
