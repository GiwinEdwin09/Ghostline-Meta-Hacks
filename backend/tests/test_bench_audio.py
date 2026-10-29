"""Bounce restitution from audio clicks (build plan Part 25): synthetic recordings with a known e,
white noise and mains hum."""

import json

import numpy as np
import pytest
from audio_helpers import click_track, drops_recording, impact_times, write_wav
from bench.audio_bounce import (
    AudioBounceError,
    analyze_bounces,
    analyze_file,
    detect_onsets,
    find_audio,
    main,
    read_wav,
    restitution_from_onsets,
    split_drops,
    truth_from_recordings,
)


# --- accuracy: the Done-when ---------------------------------------------------------------
@pytest.mark.parametrize("e", [0.5, 0.65, 0.78, 0.9])
def test_recovers_e_within_001_through_noise_and_hum(e):
    times = impact_times(e, n_clicks=9)
    x = click_track(times, e=e, noise=0.02, hum=0.5, seed=3)
    [drop] = analyze_bounces(x, 48_000)
    assert drop.e == pytest.approx(e, abs=0.01)
    assert drop.e_fit == pytest.approx(e, abs=0.02)
    assert drop.n_clicks >= 5


@pytest.mark.parametrize("seed", range(8))
def test_error_stays_inside_001_across_noise_realisations(seed):
    e = 0.75
    x = click_track(impact_times(e, 10), e=e, noise=0.03, hum=0.8, seed=seed)
    [drop] = analyze_bounces(x, 48_000)
    assert abs(drop.e - e) < 0.01


@pytest.mark.parametrize(
    ("rate", "hum_hz"), [(44_100, 50.0), (48_000, 60.0), (32_000, 60.0), (16_000, 50.0)]
)
def test_other_sample_rates_and_mains_frequencies(rate, hum_hz):
    e = 0.7
    x = click_track(impact_times(e, 9), e=e, rate=rate, hum_hz=hum_hz, noise=0.02, seed=5)
    [drop] = analyze_bounces(x, rate)
    assert drop.e == pytest.approx(e, abs=0.01)


def test_a_hum_far_louder_than_the_clicks_is_filtered_out():
    e = 0.8
    x = click_track(impact_times(e, 9), e=e, hum=6.0, rumble=4.0, noise=0.02, seed=1)
    [drop] = analyze_bounces(x, 48_000)
    assert drop.e == pytest.approx(e, abs=0.01)


def test_onset_times_match_the_true_impacts_up_to_a_constant_delay():
    times = impact_times(0.8, 9)
    x = click_track(times, e=0.8, noise=0.01, seed=2)
    found = detect_onsets(x, 48_000)
    assert len(found) == len(times)
    delay = found - times
    assert delay.mean() == pytest.approx(0.0005, abs=0.0015)  # a fraction of the smoothing window
    assert delay.std() < 0.0004  # spread under 0.4 ms (the later, quieter clicks are the worst)


def test_clicks_that_vanish_into_the_noise_end_the_series_without_biasing_it():
    e = 0.6  # the 8th click is e^7 = 2.8% of the first: below the noise
    x = click_track(impact_times(e, 12), e=e, noise=0.05, seed=4)
    [drop] = analyze_bounces(x, 48_000)
    assert 4 <= drop.n_clicks < 12
    assert drop.e == pytest.approx(e, abs=0.01)


# --- structure of a recording --------------------------------------------------------------------
def test_several_drops_in_one_recording_are_split_by_the_silence():
    es = [0.7, 0.75, 0.8]
    drops = analyze_bounces(drops_recording(es, seed=6), 48_000)
    assert len(drops) == 3
    for drop, e in zip(drops, es, strict=True):
        assert drop.e == pytest.approx(e, abs=0.01)


def test_split_drops_groups_onsets_at_long_gaps():
    onsets = np.array([0.1, 0.5, 0.8, 4.0, 4.4, 9.0])
    groups = split_drops(onsets, gap_s=2.0)
    assert [len(g) for g in groups] == [3, 2, 1]
    assert split_drops(np.array([])) == []


def test_a_release_click_before_the_first_impact_is_skipped_on_request():
    e = 0.7
    times = impact_times(e, 9)
    first_fall = 0.45  # releasing the ball makes a click one fall time before the first impact
    with_release = np.concatenate([[times[0] - first_fall], times])
    x = click_track(with_release, amplitudes=np.r_[0.5, e ** np.arange(len(times))], seed=7)
    [bad] = analyze_bounces(x, 48_000)
    [fixed] = analyze_bounces(x, 48_000, skip_first=1)
    assert fixed.e == pytest.approx(e, abs=0.01)
    # Left in, the release click's ratio (here 0.9 e / 0.45 = 1.4 e) is not a restitution; the
    # physical-range filter must keep it from being averaged in.
    assert abs(bad.e - e) < 0.05
    assert len(bad.rejected_ratios) + len(bad.ratios) == len(fixed.ratios) + 1


def test_start_and_end_windows_ignore_unwanted_audio():
    e = 0.8
    times = impact_times(e, 9, start=3.0)
    x = click_track(times, e=e, seed=8)
    chatter = round(2.8 * 48_000)  # talking before the drop, loud enough to be read as clicks
    x[:chatter] += np.random.default_rng(0).normal(0, 0.5, chatter)
    [drop] = analyze_bounces(x, 48_000, start_s=2.9)
    assert drop.e == pytest.approx(e, abs=0.01)
    [short] = analyze_bounces(x, 48_000, start_s=2.9, end_s=float(times[5]) + 0.2)
    assert short.n_clicks == 6


def test_the_rattle_at_the_end_is_not_read_as_flights():
    e = 0.7
    times = list(impact_times(e, 9))
    last = times[-1]
    times += [last + 0.012, last + 0.021, last + 0.027]  # the ball settles: merging clicks
    x = click_track(times, e=e, seed=9)
    [drop] = analyze_bounces(x, 48_000)
    assert drop.e == pytest.approx(e, abs=0.01)
    assert drop.n_clicks <= 10


def test_ratio_outliers_are_dropped_not_averaged():
    base = impact_times(0.8, 10)
    onsets = np.array(base)
    onsets[5:] += 0.05  # one flight suddenly 50 ms longer: a spurious late click
    result = restitution_from_onsets(onsets)
    assert result.rejected_ratios
    assert result.e == pytest.approx(0.8, abs=0.06)


# --- failure modes --------------------------------------------------------------------------------
def test_too_few_clicks_is_a_clear_error():
    x = click_track([0.5, 1.2], e=0.7, seed=1)
    with pytest.raises(AudioBounceError, match="found 2 clicks"):
        analyze_bounces(x, 48_000)


def test_noise_and_hum_alone_have_no_bounces():
    x = click_track([0.5], amplitudes=[0.0], e=0.7, duration=4.0, hum=2.0, noise=0.05, seed=1)
    with pytest.raises(AudioBounceError, match="could not measure"):
        analyze_bounces(x, 48_000)


def test_a_rate_too_low_for_the_band_is_refused():
    with pytest.raises(AudioBounceError, match="can't hold"):
        detect_onsets(np.zeros(4000), 2000)


def test_impossible_ratios_everywhere_is_an_error_not_an_average():
    with pytest.raises(AudioBounceError, match="physical"):
        restitution_from_onsets(np.array([0.0, 0.1, 0.5, 2.0]))  # flights growing: 4x, 3.75x


# --- WAV files ------------------------------------------------------------------------------------
@pytest.mark.parametrize(("width", "channels"), [(2, 1), (2, 2), (3, 1)])
def test_wav_formats_round_trip_to_the_same_answer(tmp_path, width, channels):
    e = 0.78
    x = click_track(impact_times(e, 9), e=e, seed=11)
    path = write_wav(tmp_path / "drop.wav", x, 48_000, width=width, channels=channels)
    samples, rate = read_wav(path)
    assert rate == 48_000 and len(samples) == len(x)
    [drop] = analyze_file(path)
    assert drop.e == pytest.approx(e, abs=0.01)


def test_float_wav_files_are_read(tmp_path):
    from scipy.io import wavfile

    e = 0.78
    x = click_track(impact_times(e, 9), e=e, seed=12)
    path = tmp_path / "float.wav"
    wavfile.write(path, 48_000, (x / np.abs(x).max()).astype(np.float32))
    [drop] = analyze_file(path)
    assert drop.e == pytest.approx(e, abs=0.01)


# --- the truth for a clip -------------------------------------------------------------------------
def test_truth_averages_five_drops_and_carries_their_spread(tmp_path):
    es = [0.72, 0.74, 0.76, 0.78, 0.8]
    paths = [
        write_wav(tmp_path / f"d{i}.wav", click_track(impact_times(e, 9), e=e, seed=20 + i))
        for i, e in enumerate(es)
    ]
    truth = truth_from_recordings(paths)
    assert truth.value == pytest.approx(np.mean(es), abs=0.01)
    spread = float(np.std(es, ddof=1))
    assert truth.sigma == pytest.approx(np.hypot(spread, 0.005), abs=0.006)
    assert "5 drop" in truth.method


def test_truth_from_one_recording_holding_all_the_drops(tmp_path):
    es = [0.7, 0.72, 0.74, 0.76, 0.78]
    path = write_wav(tmp_path / "all.wav", drops_recording(es, seed=30))
    truth = truth_from_recordings([path])
    assert truth.value == pytest.approx(np.mean(es), abs=0.01)
    assert "5 drop" in truth.method


def test_manifest_audio_truth_resolves_from_the_clips_folder(tmp_path):
    from bench.truth import ManifestRow, resolve_truth

    e = 0.7
    write_wav(tmp_path / "a.wav", click_track(impact_times(e, 9), e=e, seed=31))
    row = ManifestRow(clip_id="BO", system="bounce", quantity="e", method="audio_clicks")
    assert resolve_truth(row, clips_dir=tmp_path) is None  # no recordings named yet
    row = row.model_copy(update={"audio_files": "a.wav"})
    assert resolve_truth(row, clips_dir=tmp_path).value == pytest.approx(e, abs=0.01)


# --- command line ---------------------------------------------------------------------------------
def test_find_audio_takes_a_file_or_walks_a_folder(tmp_path):
    trial = tmp_path / "experiments" / "exp1" / "trials" / "t1"
    trial.mkdir(parents=True)
    wav = write_wav(trial / "audio.wav", click_track(impact_times(0.7, 9), e=0.7, seed=1))
    assert find_audio(wav) == [wav]
    assert find_audio(tmp_path / "experiments" / "exp1") == [wav]
    with pytest.raises(FileNotFoundError):
        find_audio(tmp_path / "nothing.wav")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no audio.wav"):
        find_audio(tmp_path / "empty")


def test_cli_prints_a_summary_and_exits_zero(tmp_path, capsys):
    e = 0.74
    write_wav(tmp_path / "audio.wav", click_track(impact_times(e, 9), e=e, seed=40))
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "drop 1:" in out and "e = 0.7" in out


def test_cli_json_output_and_failure_exit_code(tmp_path, capsys):
    e = 0.74
    good = write_wav(tmp_path / "good.wav", click_track(impact_times(e, 9), e=e, seed=41))
    assert main([str(good), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["e"] == pytest.approx(e, abs=0.01)
    assert payload["summary"]["n_drops"] == 1

    bad = write_wav(tmp_path / "bad.wav", click_track([0.5, 1.0], e=0.7, seed=1))
    assert main([str(good), str(bad)]) == 1  # one recording could not be measured
    out = capsys.readouterr().out
    assert "bad.wav" in out and "could not measure" in out
