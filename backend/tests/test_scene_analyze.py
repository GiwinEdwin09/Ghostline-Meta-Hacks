import json
import re
from types import SimpleNamespace

import pytest
from fixtures import scenarios
from fixtures.fakes import FakeSDK, response_dict, status_error, stream_events
from fixtures.synthetic_clip import ClipTruth
from test_scene_schema import valid_scene

from ghostline.config import get_settings
from ghostline.llm import InMemorySpendStore, MetaClient, ModelNotFoundError, Runtime
from ghostline.scene import (
    BoxLocation,
    PromptSuggestion,
    SceneAnalysis,
    SceneInput,
    SceneNotMeasurable,
    analyze_scene,
    locate_box,
    make_rescue_hooks,
    render_prompt,
    rephrase_prompt,
    require_measurable,
    sam_prompt_problems,
    setup_tips,
)
from ghostline.scene.analyze import MAX_IMAGES, build_user_content, load_prompt
from ghostline.vision import BoxHint


@pytest.fixture
def replayed(monkeypatch):
    return scenarios.replay_clients(monkeypatch)[0]


def live_meta(replies, tmp_path):
    runtime = Runtime.create(
        get_settings(),
        mode="live",
        cache_dir=tmp_path,
        store=InMemorySpendStore(),
        sleep=lambda s: None,
    )
    sdk = FakeSDK([stream_events(response_dict(r)) if isinstance(r, str) else r for r in replies])
    return MetaClient(runtime, sdk=sdk)


def test_the_prompt_is_filled_in_from_the_input():
    scene = SceneInput(
        duration_s=8.5, capture="240 fps slow-mo", hint="a dropped ball", stills=[b"x"]
    )

    text = render_prompt(scene)

    assert "real duration is 8.5s" in text
    assert "captured at 240 fps slow-mo" in text and 'The user says: "a dropped ball"' in text
    assert not re.search(r"\{(duration|capture|hint)\}", text)
    assert "SceneAnalysis" in text


def test_an_empty_hint_is_not_left_blank():
    assert 'The user says: "nothing"' in render_prompt(SceneInput(1.0, "60 fps", hint="  "))


def test_the_prompt_has_a_version_that_changes_with_its_text():
    text, version = load_prompt()

    assert re.fullmatch(r"[0-9a-f]{12}", version) and "SAM prompt" in text


def test_the_user_message_carries_the_video_and_labelled_stills():
    scene = SceneInput(
        duration_s=10,
        capture="30 fps",
        description="a ball bounces",
        video_file_id="file-1",
        stills=[b"\xff\xd8a", b"\xff\xd8b", b"\xff\xd8c"],
        still_times_s=[0.0, 5.0, 9.97],
    )

    parts = build_user_content(scene)

    kinds = [p["type"] for p in parts]
    assert kinds == ["input_text", "input_text", "input_video"] + ["input_text", "input_image"] * 3
    assert "a ball bounces" in parts[0]["text"]
    assert "one frame per second" in parts[1]["text"]
    assert parts[2] == {"type": "input_video", "file_id": "file-1"}
    assert "t = 5.00 s" in parts[5]["text"]
    assert parts[4]["image_url"].startswith("data:image/jpeg;base64,")


def test_without_a_video_the_stills_are_sampled_frames():
    scene = SceneInput(duration_s=10, capture="30 fps", stills=[b"a"] * 8)

    parts = build_user_content(scene)

    assert "input_video" not in [p["type"] for p in parts]
    assert "Frame 1 of 8" in parts[1]["text"] and "Frame 8 of 8" in parts[-2]["text"]


def test_a_request_carries_at_most_fifty_images():
    scene = SceneInput(duration_s=10, capture="30 fps", stills=[b"a"] * 80)

    images = [p for p in build_user_content(scene) if p["type"] == "input_image"]

    assert len(images) == MAX_IMAGES == 50


def test_wonder_needs_something_to_look_at(replayed):
    with pytest.raises(ValueError):
        analyze_scene(replayed, SceneInput(duration_s=5, capture="60 fps"))


def test_the_recorded_scene_analysis_replays_and_follows_the_rules(replayed):
    result = scenarios.scene(replayed)

    scene = result.analysis
    assert isinstance(scene, SceneAnalysis) and scene.proceed
    assert scene.system_type in {"projectile", "bounce", "free_fall"}
    assert scene.moving_bodies and 2 <= len(scene.initial_hypotheses) <= 4
    for prompt in [o.sam_prompt for o in scene.tracked_objects] + [scene.calibration.sam_prompt]:
        assert sam_prompt_problems(prompt) == []
    assert any("orange" in o.sam_prompt for o in scene.moving_bodies)
    assert scene.calibration.found
    assert result.response.cache_hit and result.response.repairs == 0
    assert result.cost_usd == pytest.approx(0.0538, abs=0.001)
    assert result.prompt_version == load_prompt()[1]
    assert require_measurable(result) is scene


def test_the_scene_request_uses_wonder_with_medium_reasoning_and_the_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("SCENE_MODEL", "muse-wonder-1.0")
    meta = live_meta([json.dumps(valid_scene())], tmp_path)
    scene_input = SceneInput(
        duration_s=10, capture="30 fps normal video", hint="something bouncing", video_file_id="f-1"
    )

    result = analyze_scene(meta, scene_input, experiment_id="exp1")

    (call,) = meta.sdk.responses.calls
    assert call["model"] == "muse-wonder-1.0" and call["stream"] is True
    assert call["reasoning"] == {"effort": "medium"} and call["max_output_tokens"] >= 8000
    assert call["text"]["format"]["name"] == "SceneAnalysis"
    assert "captured at 30 fps normal video" in call["instructions"]
    assert result.analysis.system_type == "bounce"


def test_a_reply_that_breaks_the_rules_gets_one_repair(tmp_path):
    bad = valid_scene()
    bad["tracked_objects"][0]["sam_prompt"] = "orange ball and blue dot"
    meta = live_meta([json.dumps(bad), json.dumps(valid_scene())], tmp_path)

    result = analyze_scene(meta, SceneInput(10, "30 fps", stills=[b"x"]))

    assert result.response.repairs == 1 and len(meta.sdk.responses.calls) == 2
    repair_text = meta.sdk.responses.calls[1]["input"][-1]["content"][0]["text"]
    assert "tracked_objects.0.sam_prompt" in repair_text and "lists several things" in repair_text


def test_a_missing_model_is_a_clear_error(tmp_path):
    meta = live_meta([status_error(404, code="model_not_found")], tmp_path)

    with pytest.raises(ModelNotFoundError, match="preview"):
        analyze_scene(meta, SceneInput(10, "30 fps", stills=[b"x"]))


def test_when_wonder_says_to_stop_the_reason_and_setup_tips_are_shown(tmp_path):
    stop = valid_scene(
        proceed=False,
        tracked_objects=[],
        initial_hypotheses=[],
        reason_if_not="The sheet is cut off and the camera shakes.",
        quality_issues=[
            {"issue": "Handheld camera shake", "severity": "major", "note": "the view moves"},
            {"issue": "The A4 sheet is partly out of frame", "severity": "major", "note": ""},
        ],
    )
    meta = live_meta([json.dumps(stop)], tmp_path)
    result = analyze_scene(meta, SceneInput(10, "30 fps", stills=[b"x"]))

    assert not result.analysis.proceed
    with pytest.raises(SceneNotMeasurable) as caught:
        require_measurable(result)

    message = str(caught.value)
    assert "The sheet is cut off" in message
    assert any("tripod" in tip for tip in caught.value.tips)
    assert any("A4 sheet" in tip for tip in caught.value.tips)


def test_setup_tips_fall_back_to_the_checklist():
    data = valid_scene(proceed=False, tracked_objects=[], initial_hypotheses=[])
    data.update(reason_if_not="unclear", quality_issues=[])

    tips = setup_tips(SceneAnalysis.model_validate(data))

    assert len(tips) == 1 and "checklist" in tips[0]


# Wonder's help for the SAM retry ladder


def test_the_recorded_rephrase_is_a_valid_new_prompt(replayed):
    new = scenarios.rephrase(replayed)

    assert new and new != "red cube" and sam_prompt_problems(new) == []


def test_the_recorded_box_contains_the_ball(replayed):
    hint = scenarios.box(replayed)
    truth = ClipTruth.from_json(scenarios.TRUTH.read_text())
    cx, cy = truth.centers[150]

    assert isinstance(hint, BoxHint) and hint.frame == 150
    assert hint.left <= cx < hint.right and hint.top <= cy < hint.bottom


class StubMeta:
    """Returns canned parsed objects from `respond`."""

    settings = SimpleNamespace(scene_model="muse-wonder-1.0")

    def __init__(self, *parsed):
        self.parsed = list(parsed)
        self.calls = []

    def respond(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed=self.parsed.pop(0))


def test_repeating_a_failed_prompt_means_give_up():
    meta = StubMeta(PromptSuggestion(sam_prompt="Red Cube", reasoning="same"))

    assert rephrase_prompt(meta, failed=["red cube"], description="ball", frame=b"x") is None


def test_the_rephrase_request_shows_a_frame_and_lists_failures():
    meta = StubMeta(PromptSuggestion(sam_prompt="orange ball", reasoning="color"))

    new = rephrase_prompt(
        meta, failed=["red cube", "ball"], description="the bouncing ball", frame=b"\xff"
    )

    (call,) = meta.calls
    text = call["input"][0]["content"][0]["text"]
    assert new == "orange ball" and call["text_format"] is PromptSuggestion
    assert '"red cube", "ball"' in text and "the bouncing ball" in text
    assert call["input"][0]["content"][1]["type"] == "input_image"
    assert call["reasoning_effort"] == "low" and call["model"] == "muse-wonder-1.0"


def test_a_suggested_prompt_must_follow_the_sam_rules():
    with pytest.raises(ValueError):
        PromptSuggestion(sam_prompt="the ball on the left", reasoning="")


def test_a_box_is_clipped_to_the_frame_and_none_when_not_found():
    meta = StubMeta(
        BoxLocation(found=True, x1=-5, y1=10, x2=900, y2=50),
        BoxLocation(found=False, x1=0, y1=0, x2=0, y2=0),
        BoxLocation(found=True, x1=860, y1=10, x2=900, y2=50),
    )
    ask = {"description": "ball", "frame": b"x", "frame_index": 7, "frame_size": (854, 480)}

    assert locate_box(meta, **ask) == BoxHint(7, 0, 10, 854, 50)
    assert locate_box(meta, **ask) is None
    assert locate_box(meta, **ask) is None  # entirely outside the frame
    assert "854 x 480" in meta.calls[0]["input"][0]["content"][0]["text"]


def test_a_box_must_have_positive_size():
    with pytest.raises(ValueError, match="x2 > x1"):
        BoxLocation(found=True, x1=10, y1=10, x2=10, y2=20)


def test_rescue_hooks_pass_failures_along_and_show_the_first_frame():
    meta = StubMeta(
        PromptSuggestion(sam_prompt="orange ball", reasoning=""),
        PromptSuggestion(sam_prompt="orange sphere", reasoning=""),
        BoxLocation(found=True, x1=10, y1=10, x2=30, y2=30),
    )
    hooks = make_rescue_hooks(
        meta, description="the ball", frames={40: b"late", 12: b"early"}, frame_size=(854, 480)
    )

    first = hooks.rephrase("orange foam ball", 1)
    second = hooks.rephrase("orange ball", 2)
    box = hooks.locate_box()

    assert (first, second) == ("orange ball", "orange sphere")
    assert "red cube" not in meta.calls[1]["input"][0]["content"][0]["text"]
    assert '"orange foam ball", "orange ball"' in meta.calls[1]["input"][0]["content"][0]["text"]
    assert box == BoxHint(12, 10, 10, 30, 30)  # the earliest frame was shown
