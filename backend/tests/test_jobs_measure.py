"""The pieces around the measurement stages: upload validation (Part 28), the analysis window and
Files API bookkeeping, the Postgres-backed spend store, and how model errors become job errors."""

import dataclasses
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ghostline.jobs import media
from ghostline.jobs.clients import RepoSpendStore, llm_errors
from ghostline.jobs.errors import StageError, TransientStageError
from ghostline.jobs.measure import _choose, _f35_from_tags, _is_sheet, file_key
from ghostline.jobs.validate import validate_upload
from ghostline.llm import (
    CallRecord,
    DailyBudgetExceeded,
    ModelNotFoundError,
    PermanentLLMError,
    ReplayMissError,
    StructuredOutputError,
    TransientLLMError,
)
from ghostline.store.core_schemas import SceneAnalysis, TrackedObject
from ghostline.store.records import Trial, Trim

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    """Small files of every kind a stranger might upload."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    root = tmp_path_factory.mktemp("uploads")
    video = root / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=1", "-pix_fmt", "yuv420p",
           str(video))  # fmt: skip
    image = root / "still.jpg"
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120", "-frames:v", "1", str(image))
    sound = root / "sound.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(sound))
    text = root / "notes.mp4"
    text.write_text("this is not a video, whatever the name says")
    subs = root / "subs.srt"
    subs.write_text("1\n00:00:00,000 --> 00:00:00,900\nhello\n")
    captioned = root / "captioned.mkv"
    ffmpeg("-i", str(video), "-i", str(subs), "-c:v", "copy", "-c:s", "srt", str(captioned))
    with_audio = root / "with_audio.mp4"
    ffmpeg("-i", str(video), "-f", "lavfi", "-i", "sine=frequency=300:duration=1", "-c:v", "copy",
           "-shortest", str(with_audio))  # fmt: skip
    return {
        "video": video,
        "image": image,
        "sound": sound,
        "text": text,
        "captioned": captioned,
        "with_audio": with_audio,
    }


# --- Part 28: upload validation -----------------------------------------------------------------
@needs_ffmpeg
def test_a_plain_video_passes_and_audio_is_allowed(files, settings):
    probe = validate_upload(files["video"], settings)
    assert probe.nb_frames == 30 and probe.codec == "h264"
    assert validate_upload(files["with_audio"], settings).has_audio


@needs_ffmpeg
@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("text", "not_a_video"),  # ffprobe can't read it
        ("image", "not_a_video"),  # one frame
        ("sound", "not_a_video"),  # no video stream
        ("captioned", "unsupported_streams"),  # video plus a subtitle stream
    ],
)
def test_files_that_are_not_plain_videos_are_refused_for_good(files, settings, name, code):
    with pytest.raises(StageError) as refused:
        validate_upload(files[name], settings)
    assert refused.value.code == code and refused.value.retryable is False
    assert refused.value.message  # says what is wrong, in words


@needs_ffmpeg
def test_the_frame_and_size_limits(files, settings):
    with pytest.raises(StageError) as many:
        validate_upload(files["video"], dataclasses.replace(settings, max_frames=20))
    assert many.value.code == "too_many_frames" and "30" in many.value.message
    with pytest.raises(StageError) as big:
        validate_upload(files["video"], dataclasses.replace(settings, max_upload_bytes=1000))
    assert big.value.code == "too_large"


# --- the analysis window --------------------------------------------------------------------------
def trial_with(trim, timing=None) -> Trial:
    return Trial(
        id="11111111-1111-1111-1111-111111111111",
        experiment_id="22222222-2222-2222-2222-222222222222",
        role="fit",
        status="ready",
        sha256="ab" * 32,
        trim=trim,
        timing=timing,
        created_at=datetime.now(UTC),
    )


T = np.arange(300) / 60  # five seconds at 60 fps


def test_the_trim_picks_the_frames_inside_it(settings):
    window = media.analysis_window(trial_with(Trim(start_s=1.0, end_s=3.0)), T, settings)
    assert (window.start, window.stop, window.n) == (60, 180, 120)
    assert window.key == "master@60-180" and not window.covers(300)


def test_a_trim_edge_on_a_frame_time_keeps_it_at_the_start_and_drops_it_at_the_end(settings):
    window = media.analysis_window(trial_with(Trim(start_s=T[10], end_s=T[50])), T, settings)
    assert (window.start, window.stop) == (10, 50)
    whole = media.analysis_window(trial_with(Trim(start_s=0.0, end_s=T[-1] + 1 / 60)), T, settings)
    assert whole.covers(300)


def test_the_usable_range_of_a_mixed_rate_clip_limits_the_window(settings):
    timing = {"usable_start": 40, "usable_stop": 200}
    window = media.analysis_window(trial_with(Trim(start_s=0.0, end_s=5.0), timing), T, settings)
    assert (window.start, window.stop) == (40, 200)


def test_a_window_that_is_too_short_or_too_long_is_refused(settings):
    with pytest.raises(StageError) as short:
        media.analysis_window(trial_with(Trim(start_s=1.0, end_s=1.05)), T, settings)
    assert short.value.code == "window_too_short" and not short.value.retryable
    with pytest.raises(StageError) as long:
        media.analysis_window(
            trial_with(Trim(start_s=0.0, end_s=5.0)),
            T,
            dataclasses.replace(settings, max_frames=100),
        )
    assert long.value.code == "too_many_frames"


def test_the_cache_identity_follows_the_original_not_the_encoded_bytes():
    trial = trial_with(None)
    same = trial.model_copy()
    other = trial.model_copy(update={"sha256": "cd" * 32})
    assert media.cache_identity(trial, "wonder_480") == media.cache_identity(same, "wonder_480")
    assert media.cache_identity(trial, "wonder_480") != media.cache_identity(trial, "master@0-9")
    assert media.cache_identity(trial, "wonder_480") != media.cache_identity(other, "wonder_480")
    assert len(media.cache_identity(trial, "x")) == 64


# --- small helpers --------------------------------------------------------------------------------
def test_object_keys_from_a_model_become_safe_file_names():
    assert file_key("falling ball") == "falling_ball"
    assert file_key("../../etc/passwd") == "etc_passwd"
    assert file_key("///") == "object"
    assert len(file_key("x" * 300)) == 64


def test_the_sheet_is_not_tracked_as_a_body():
    sheet = TrackedObject(
        key="scale_sheet", sam_prompt="white rectangular paper sheet", role="moving_body", why=""
    )
    ball = TrackedObject(key="ball", sam_prompt="orange spherical ball", role="moving_body", why="")
    paper_ball = TrackedObject(
        key="b", sam_prompt="crumpled paper ball", role="moving_body", why=""
    )
    assert _is_sheet(sheet, "white sheet of paper")
    assert _is_sheet(
        ball, "Orange Spherical Ball"
    )  # the same prompt as the sheet's, however spelled
    assert not _is_sheet(ball, "white sheet of paper")
    assert not _is_sheet(paper_ball, "white sheet of paper")


def scene_with(system, *objects) -> SceneAnalysis:
    return SceneAnalysis(
        system_type=system,
        description="",
        tracked_objects=[
            TrackedObject(key=k, sam_prompt=k, role=role, why="") for k, role in objects
        ],
        calibration={"sam_prompt": "white sheet of paper"},
        object_properties={},
        initial_hypotheses=[],
        quality_issues=[],
        proceed=True,
    )


def test_the_trajectory_follows_the_first_moving_body_except_for_a_spinner():
    swing = scene_with("pendulum", ("bob", "moving_body"), ("nail", "pivot"))
    assert _choose(swing, {"bob", "nail"}) == ("bob", "nail")
    assert _choose(swing, {"bob"}) == ("bob", None)  # a pivot that wasn't found is optional
    spin = scene_with("spin_down", ("spinner", "moving_body"), ("dot", "marker"), ("hub", "pivot"))
    assert _choose(spin, {"spinner", "dot", "hub"}) == ("dot", "hub")
    with pytest.raises(StageError):
        _choose(swing, {"nail"})


def test_the_focal_length_is_read_from_phone_tags_when_it_is_there():
    tags = {"com.apple.quicktime.camera.focal_length.35mm_equivalent": "26.0"}
    assert _f35_from_tags(tags) == 26.0
    assert _f35_from_tags({"title": "x"}) is None
    assert _f35_from_tags({"focal_length.35mm": "oops"}) is None
    assert _f35_from_tags({"focal_length.35mm": "5000"}) is None  # not a real lens


# --- spend in Postgres ----------------------------------------------------------------------------
def test_every_call_is_recorded_and_only_paid_calls_reach_the_ledger(repo):
    exp = repo.create_experiment(title="spend")
    store = RepoSpendStore(repo)
    paid = CallRecord(
        provider="meta", kind="sam_video", model="sam-3.1", cost_usd=0.02, request_hash="a" * 8,
        experiment_id=str(exp.id), phase="track", frames=100,
    )  # fmt: skip
    cached = dataclasses.replace(paid, cost_usd=0.0, cache_hit=True, request_hash="b" * 8)
    wonder = dataclasses.replace(
        paid, kind="respond", model="muse-wonder-1.0", cost_usd=0.05, request_hash="c" * 8
    )
    for record in (paid, cached, wonder):
        store.add(record)

    assert store.experiment_total(str(exp.id)) == pytest.approx(0.07)
    assert repo.get_experiment(exp.id).cost_usd == pytest.approx(0.07)
    assert store.day_total(datetime.now(UTC).date()) == pytest.approx(0.07)
    assert store.day_total((datetime.now(UTC) - timedelta(days=2)).date()) == 0.0
    stats = repo.llm_stats_today()
    assert stats == {"calls": 3, "cache_hits": 1}
    categories = {r["ref"]: r["category"] for r in repo.db.fetch_all("select * from spend_ledger")}
    assert categories == {"a" * 8: "sam", "c" * 8: "llm"}


def test_a_call_for_an_unknown_experiment_id_is_still_recorded(repo):
    store = RepoSpendStore(repo)
    store.add(
        CallRecord(
            provider="meta",
            kind="respond",
            model="m",
            cost_usd=0.01,
            request_hash="x" * 8,
            experiment_id="not-a-uuid",
        )  # fmt: skip
    )
    assert repo.llm_stats_today()["calls"] == 1 and repo.spend_today() == pytest.approx(0.01)
    assert store.experiment_total("not-a-uuid") == 0.0


def test_the_guard_reads_spend_across_workers(repo, settings):
    """The budget guard sums the ledger, so two workers share one cap."""
    from ghostline.llm import BudgetGuard, ExperimentBudgetExceeded

    exp = repo.create_experiment(title="cap")
    guard = BudgetGuard(RepoSpendStore(repo), daily_usd=5.0, per_experiment_usd=0.10)
    guard.check_call(str(exp.id))
    repo.add_spend(amount_usd=0.12, provider="meta", experiment_id=exp.id)
    with pytest.raises(ExperimentBudgetExceeded):
        guard.check_call(str(exp.id))
    guard.check_new_analysis()
    repo.add_spend(amount_usd=5.0, provider="meta")
    with pytest.raises(DailyBudgetExceeded):
        guard.check_new_analysis()


# --- model errors become job errors ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "kind", "code", "retryable"),
    [
        (DailyBudgetExceeded(5, 5), StageError, "daily_budget_reached", False),
        (ModelNotFoundError("no such model"), StageError, "model_not_found", False),
        (PermanentLLMError("bad request"), StageError, "model_error", False),
        (ReplayMissError("respond", "ab" * 32, "m"), StageError, "replay_miss", False),
        (StructuredOutputError("junk"), StageError, "bad_model_reply", True),
        (TransientLLMError("429"), TransientStageError, "model_busy", True),
    ],
)
def test_llm_failures_say_what_happened(error, kind, code, retryable):
    with pytest.raises(kind) as caught, llm_errors():
        raise error
    assert caught.value.code == code and caught.value.retryable is retryable
    assert caught.value.message


def test_other_exceptions_pass_through_untouched():
    with pytest.raises(ZeroDivisionError), llm_errors():
        _ = 1 / 0
