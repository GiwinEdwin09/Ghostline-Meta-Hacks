"""The SAM retry ladder and the sheet, with a stand-in Meta client and recorded SAM output."""

import json
from dataclasses import replace

import cv2
import numpy as np
import pytest
from fixtures import scenarios
from fixtures.synthetic_clip import write_video

from ghostline.llm import SamResult
from ghostline.vision import (
    BoxHint,
    RescueHooks,
    TrackingFailed,
    box_prompt,
    count_frames,
    median_frame,
    read_frames,
    segment_sheet,
    track_object,
    video_size,
)
from ghostline.vision.frames import spread_indices

BALL = "orange ball"


@pytest.fixture
def recorded(monkeypatch):
    meta, _, _ = scenarios.replay_clients(monkeypatch)
    # Present the recordings as live results, so costs are counted.
    return {
        "ball": replace(scenarios.sam_video(meta, scenarios.BALL_PROMPT), cache_hit=False),
        "sheet": replace(scenarios.sam_sheet(meta), cache_hit=False),
    }


def nothing():
    return SamResult(events=[], output_text="", usage={}, frames_processed=90, cost_usd=0.018)


class FakeMeta:
    """Answers `sam_video` from a table of prompt text to SamResult, and records the requests."""

    def __init__(self, answers, sheet=None):
        self.answers = answers
        self.sheet = sheet
        self.video_calls = []
        self.image_calls = []

    def sam_video(self, file_id, prompt, *, metadata=None, **kwargs):
        self.video_calls.append(
            {"file_id": file_id, "prompt": prompt, "metadata": metadata, **kwargs}
        )
        return self.answers.get(prompt, nothing())

    def sam_image(self, image, prompt, **kwargs):
        self.image_calls.append({"image": image, "prompt": prompt, **kwargs})
        return self.sheet


def track(meta, **kwargs):
    return track_object(
        meta,
        file_id="file-1",
        key="ball",
        prompt="orange foam ball",
        frame_size=(854, 480),
        **kwargs,
    )


def test_the_first_prompt_finding_the_object_is_enough(recorded, tmp_path):
    meta = FakeMeta({"orange foam ball": recorded["ball"]})

    result = track(meta, events_dir=tmp_path / "sam", media_sha256="abc", n_frames=90)

    assert len(meta.video_calls) == 1
    assert meta.video_calls[0]["media_sha256"] == "abc" and meta.video_calls[0]["frames_hint"] == 90
    assert result.track.key == "ball" and len(result.track) == 90
    assert result.attempts == ["prompt 'orange foam ball': found"]
    assert result.cost_usd == pytest.approx(0.018)
    saved = result.events_path.read_text().splitlines()
    assert result.events_path.name == "ball.events.jsonl"
    assert len(saved) == len(recorded["ball"].events)
    assert json.loads(saved[0])["type"] == "response.created"


def test_a_failed_prompt_is_rephrased(recorded):
    meta = FakeMeta({BALL: recorded["ball"]})
    asked = []

    def rephrase(previous, attempt):
        asked.append((previous, attempt))
        return BALL

    result = track(meta, rescue=RescueHooks(rephrase=rephrase))

    assert asked == [("orange foam ball", 1)]
    assert [c["prompt"] for c in meta.video_calls] == ["orange foam ball", BALL]
    assert result.prompt == BALL and len(result.track) == 90
    assert result.attempts == [
        "prompt 'orange foam ball': nothing usable",
        f"rephrased '{BALL}': found",
    ]
    assert result.cost_usd == pytest.approx(0.036)  # both requests are paid for


def test_at_most_two_rephrasings_before_the_box(recorded):
    meta = FakeMeta({})
    asked = []

    def rephrase(previous, attempt):
        asked.append(attempt)
        return f"red cube {attempt}"

    with pytest.raises(TrackingFailed):
        track(meta, rescue=RescueHooks(rephrase=rephrase))

    assert asked == [1, 2] and len(meta.video_calls) == 3


def test_repeating_the_same_prompt_ends_the_rephrasing():
    meta = FakeMeta({})

    with pytest.raises(TrackingFailed):
        track(meta, rescue=RescueHooks(rephrase=lambda previous, attempt: previous))

    assert len(meta.video_calls) == 1


def test_the_box_rung_sends_a_box_prompt_that_propagates_both_ways(recorded):
    box_text = box_prompt(BoxHint(30, 100, 110, 140, 150), (854, 480))
    meta = FakeMeta({box_text: recorded["ball"]})
    hooks = RescueHooks(
        rephrase=lambda p, a: None, locate_box=lambda: BoxHint(30, 100, 110, 140, 150)
    )

    result = track(meta, rescue=hooks)

    assert box_text.startswith("<30f>1<|box;x1=100;y1=110;x2=139;y2=149;w=854;h=480|>")
    assert meta.video_calls[-1]["prompt"] == box_text
    assert meta.video_calls[-1]["metadata"] == {"propagation_direction": "both"}
    assert result.attempts[-1] == "box on frame 30: found" and len(result.track) == 90


def test_the_ladder_ends_in_a_clear_failure_with_filming_tips():
    meta = FakeMeta({})
    hooks = RescueHooks(
        rephrase=lambda p, a: f"thing {a}", locate_box=lambda: BoxHint(0, 1, 1, 9, 9)
    )

    with pytest.raises(TrackingFailed) as caught:
        track(meta, rescue=hooks)

    message = str(caught.value)
    assert "couldn't find the 'ball'" in message and "tripod" in message
    assert len(caught.value.attempts) == 4  # the prompt, two rephrasings, the box


def test_a_caller_can_reject_a_track_that_found_the_wrong_thing(recorded):
    meta = FakeMeta({"orange foam ball": recorded["ball"], BALL: recorded["ball"]})
    seen = []

    def accept(candidate):
        seen.append(len(candidate))
        return len(seen) > 1  # reject the first, accept the second

    result = track(meta, rescue=RescueHooks(rephrase=lambda p, a: BALL), accept=accept)

    assert result.prompt == BALL and len(meta.video_calls) == 2


def test_the_frame_size_comes_from_the_output_when_not_given(recorded):
    meta = FakeMeta({"orange foam ball": recorded["ball"]})

    result = track_object(meta, file_id="f", key="ball", prompt="orange foam ball")

    assert result.track.frame_size == (854, 480)


def test_a_box_hint_without_a_known_frame_size_is_skipped_not_crashed():
    meta = FakeMeta({})

    with pytest.raises(TrackingFailed) as caught:
        track_object(
            meta,
            file_id="f",
            key="ball",
            prompt="x",
            rescue=RescueHooks(locate_box=lambda: BoxHint(0, 0, 0, 5, 5)),
        )

    assert "frame size is unknown" in caught.value.attempts[-1]


def moving_square_video(path, n=20, size=(854, 480)):
    """A grey scene with a bright square that moves; its median over time is the plain scene."""
    width, height = size

    def frames():
        for i in range(n):
            image = np.full((height, width, 3), 90, np.uint8)
            x = 40 + i * 30
            cv2.rectangle(image, (x, 200), (x + 40, 240), (255, 255, 255), -1)
            yield image

    write_video(path, frames(), fps=30)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    moving_square_video(path)
    return path


def test_frame_helpers(clip):

    assert count_frames(clip) == 20 and video_size(clip) == (854, 480)
    frames = read_frames(clip, [0, 19])
    assert sorted(frames) == [0, 19] and frames[0].shape == (480, 854, 3)
    assert spread_indices(20, 3) == [0, 10, 19] and spread_indices(5, 15) == [0, 1, 2, 3, 4]


def test_the_median_image_removes_the_moving_object(clip):

    median = median_frame(clip, count=15)

    assert np.abs(median.astype(int) - 90).max() <= 4  # no trace of the square is left


def test_the_sheet_is_segmented_from_the_median_and_three_probe_frames(recorded, clip):
    meta = FakeMeta({}, sheet=recorded["sheet"])

    sheet = segment_sheet(meta, clip, prompt="white sheet of paper")

    assert len(meta.image_calls) == 4
    assert {c["prompt"] for c in meta.image_calls} == {"white sheet of paper"}
    assert [m.frame for m in [sheet.median, *sheet.probes]] == [None, 0, 10, 19]
    assert sheet.median.mask.shape == (480, 854) and sheet.frame_size == (854, 480)
    assert sheet.median.mask.sum() == sheet.median.area
    assert sheet.median.box == (141, 46, 705, 440)
    assert sheet.median.confidence == pytest.approx(0.92, abs=0.01)
    assert sheet.cost_usd == pytest.approx(4 * 0.0025)
    # The first image sent is the median, which shows no moving square.
    first = cv2.imdecode(np.frombuffer(meta.image_calls[0]["image"], np.uint8), cv2.IMREAD_COLOR)
    assert np.abs(first.astype(int) - 90).max() <= 4


def test_the_sheet_result_is_written_next_to_the_tracks(recorded, clip, tmp_path):
    sheet = segment_sheet(FakeMeta({}, sheet=recorded["sheet"]), clip)

    summary = sheet.write(tmp_path / "trial")

    saved = json.loads(summary.read_text())
    assert saved["frame_size"] == [854, 480] and len(saved["probes"]) == 3
    assert saved["median"]["box"] == [141, 46, 705, 440]
    mask = cv2.imread(str(tmp_path / "trial" / "sheet_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (480, 854) and (mask > 0).sum() == sheet.median.area
    assert (tmp_path / "trial" / "median.png").exists()


def test_no_sheet_found_is_a_clear_failure(clip):
    empty = SamResult(events=[], output_text="", usage={}, frames_processed=0, cost_usd=0.0025)

    with pytest.raises(TrackingFailed, match="calibration sheet"):
        segment_sheet(FakeMeta({}, sheet=empty), clip)
