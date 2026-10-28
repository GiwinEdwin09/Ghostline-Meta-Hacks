import json
from typing import Literal

import pytest
from pydantic import BaseModel, ValidationError

from ghostline.llm import strict_schema
from ghostline.scene import SceneAnalysis, TrackedObject, sam_prompt_problems


def valid_scene(**overrides):
    data = {
        "system_type": "bounce",
        "description": "An orange ball bounces on a sheet.",
        "tracked_objects": [
            {
                "key": "ball",
                "sam_prompt": "orange foam ball",
                "role": "moving_body",
                "why": "the bouncing body",
            },
            {
                "key": "dot",
                "sam_prompt": "blue dot sticker",
                "role": "marker",
                "why": "a fixed reference",
            },
        ],
        "calibration": {
            "sam_prompt": "white sheet of paper",
            "found": True,
            "fully_visible": True,
            "in_motion_plane": True,
            "note": "all four corners are visible",
        },
        "object_properties": {"shape": "sphere", "material_guess": "foam", "approx_size_cm": 6.5},
        "initial_hypotheses": [
            {
                "name": "drag-free bounce",
                "mechanism": "elastic loss",
                "observable_signature": "peaks shrink",
            },
            {
                "name": "quadratic drag",
                "mechanism": "air drag",
                "observable_signature": "asymmetric arcs",
            },
        ],
        "quality_issues": [{"issue": "blur", "severity": "minor", "note": "slight"}],
        "proceed": True,
        "reason_if_not": None,
    }
    return {**data, **overrides}


def walk(node):
    """Every dict in a JSON schema."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def test_the_scene_schema_is_valid_strict_structured_output():
    schema = strict_schema(SceneAnalysis)

    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and "anyOf" not in text
    for node in walk(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
        assert "title" not in node and "default" not in node
    properties = schema["properties"]
    assert set(properties["system_type"]["enum"]) == {
        "free_fall", "projectile", "pendulum", "ramp", "bounce", "spin_down", "oscillator", "other",
    }  # fmt: skip
    assert properties["object_properties"]["properties"]["approx_size_cm"]["type"] == [
        "number",
        "null",
    ]
    assert properties["reason_if_not"]["type"] == ["string", "null"]
    assert properties["tracked_objects"]["items"]["properties"]["role"]["enum"] == [
        "moving_body", "marker", "pivot", "surface",
    ]  # fmt: skip


def test_strict_schema_keeps_fields_named_like_schema_keywords():
    class Odd(BaseModel):
        title: str
        default: int
        format: Literal["a", "b"] = "a"
        nested: list[TrackedObject]

    schema = strict_schema(Odd)

    assert list(schema["properties"]) == ["title", "default", "format", "nested"]
    assert schema["required"] == ["title", "default", "format", "nested"]
    assert schema["properties"]["format"]["enum"] == ["a", "b"]
    assert schema["properties"]["nested"]["items"]["properties"]["key"] == {"type": "string"}


def test_a_valid_analysis_round_trips():
    scene = SceneAnalysis.model_validate(valid_scene())

    assert SceneAnalysis.model_validate_json(scene.model_dump_json()) == scene
    assert [o.key for o in scene.moving_bodies] == ["ball"]


@pytest.mark.parametrize(
    "prompt",
    [
        "orange foam ball",
        "blue dot sticker",
        "white sheet of paper",
        "steel ball bearing",
        "A4 sheet",
    ],
)
def test_good_sam_prompts(prompt):
    assert sam_prompt_problems(prompt) == []


@pytest.mark.parametrize(
    ("prompt", "problem"),
    [
        ("orange ball, blue dot", "lists"),
        ("ball and sheet", "lists"),
        ("orange ball + white sheet", "lists"),
        ("the ball that is falling", "verb"),
        ("moving orange ball", "verb"),
        ("find the orange ball", "verb"),
        ("ball not on the sheet", "negation"),
        ("orange ball without a string", "negation"),
        ("ball on the left", "position"),
        ("the ball above the sheet", "position"),
        ("where is the orange ball?", "question"),
        ("a small round orange foam ball sitting there", "words"),
        ("", "empty"),
    ],
)
def test_bad_sam_prompts_are_named(prompt, problem):
    problems = sam_prompt_problems(prompt)

    assert problems and any(problem in p for p in problems)


def test_a_bad_prompt_fails_validation_with_a_message_the_repair_call_can_use():
    data = valid_scene()
    data["tracked_objects"][0]["sam_prompt"] = "the ball moving left, not the dot"

    with pytest.raises(ValidationError) as caught:
        SceneAnalysis.model_validate(data)

    message = str(caught.value)
    assert "short noun phrase" in message and "negation" in message


def test_the_sheet_prompt_follows_the_same_rules():
    data = valid_scene()
    data["calibration"]["sam_prompt"] = "the white sheet behind the ball"

    with pytest.raises(ValidationError, match="position word"):
        SceneAnalysis.model_validate(data)


def test_proceeding_needs_a_moving_body_and_two_to_four_hypotheses():
    data = valid_scene()
    data["tracked_objects"] = [data["tracked_objects"][1]]
    with pytest.raises(ValidationError, match="moving_body"):
        SceneAnalysis.model_validate(data)

    one = valid_scene(initial_hypotheses=valid_scene()["initial_hypotheses"][:1])
    with pytest.raises(ValidationError, match="2 to 4"):
        SceneAnalysis.model_validate(one)

    five = valid_scene(initial_hypotheses=valid_scene()["initial_hypotheses"] * 3)
    with pytest.raises(ValidationError, match="2 to 4"):
        SceneAnalysis.model_validate(five)


def test_object_keys_are_unique():
    data = valid_scene()
    data["tracked_objects"][1]["key"] = "ball"

    with pytest.raises(ValidationError, match="unique"):
        SceneAnalysis.model_validate(data)


def test_stopping_needs_a_reason_but_no_objects():
    stop = valid_scene(proceed=False, tracked_objects=[], initial_hypotheses=[])

    with pytest.raises(ValidationError, match="reason_if_not"):
        SceneAnalysis.model_validate(stop)
    ok = SceneAnalysis.model_validate({**stop, "reason_if_not": "the sheet is cut off"})
    assert not ok.proceed


def test_unknown_fields_and_values_are_rejected():
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(valid_scene(extra_field=1))
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(valid_scene(system_type="tennis"))
