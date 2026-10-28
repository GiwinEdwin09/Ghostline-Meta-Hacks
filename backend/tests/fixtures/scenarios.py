"""Recorded model-call scenarios.

Each function makes the same calls whether it runs in `record` mode (`record.py`, which spends
money) or `replay` mode (the tests, which don't): the request, and so its cache key, is built in
one place. The responses live in `cache/`, with auth headers and keys never stored.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from ghostline.config import get_settings
from ghostline.llm import (
    MetaClient,
    ModelResponse,
    ReviewerClient,
    Runtime,
    SamResult,
    data_url,
    function_call_output,
    function_tool,
    input_image,
    user_message,
)
from ghostline.scene import SceneInput
from ghostline.scene.analyze import SceneResult, analyze_scene
from ghostline.scene.rescue import locate_box, rephrase_prompt
from ghostline.vision import BoxHint

FIXTURES = Path(__file__).parent
CACHE_DIR = FIXTURES / "cache"
STILLS = [FIXTURES / "stills" / f"still_{i}.jpg" for i in range(3)]
SHEET_FRAME = FIXTURES / "stills" / "sheet_frame.jpg"
MEDIA_INFO = FIXTURES / "media.json"
TRUTH = FIXTURES / "truth.json"

WONDER_FILE_ID = "file-fixture-wonder480"
SAM_FILE_ID = "file-fixture-sam3s"
FRAME_SIZE = (854, 480)
SAM_FRAMES = 90

BALL_PROMPT = "orange ball"
DOT_PROMPT = "blue dot sticker"
SHEET_PROMPT = "white sheet of paper"


def media_info() -> dict:
    """Hashes of the media the recorded requests refer to, and the model names recorded."""
    return json.loads(MEDIA_INFO.read_text())


class GravityAnswer(BaseModel):
    planet: str
    g_m_s2: float


class PlotRequest(BaseModel):
    name: str


def gravity(meta: MetaClient) -> ModelResponse:
    """Wonder, text only, structured output (Part 1, check 1)."""
    return meta.respond(
        input="What is the surface gravity on Mars, in m/s^2?",
        text_format=GravityAnswer,
        reasoning_effort="low",
        max_output_tokens=3000,
        experiment_id="fixture",
        phase="gravity",
    )


def tool_roundtrip(meta: MetaClient) -> tuple[ModelResponse, ModelResponse]:
    """Wonder calls a tool, gets an image back, and answers (Part 1, check 3)."""
    tool = function_tool(
        "get_plot", "Return a named plot of the experiment as an image.", PlotRequest
    )
    question = user_message(
        "Call get_plot with name 'frame'. Then describe the main colors in it in one sentence."
    )
    first = meta.respond(
        input=[question],
        tools=[tool],
        reasoning_effort="low",
        max_output_tokens=3000,
        experiment_id="fixture",
        phase="tools",
    )
    call = first.function_calls[0]
    image = input_image(STILLS[0].read_bytes(), "image/jpeg")
    second = meta.respond(
        input=[question, *first.output, function_call_output(call.call_id, [image])],
        tools=[tool],
        reasoning_effort="low",
        max_output_tokens=3000,
        experiment_id="fixture",
        phase="tools",
    )
    return first, second


def scene_input(file_id: str = WONDER_FILE_ID) -> SceneInput:
    return SceneInput(
        duration_s=10.0,
        capture="30 fps normal video",
        hint="something bouncing",
        video_file_id=file_id,
        stills=[p.read_bytes() for p in STILLS],
        still_times_s=[0.0, 5.0, 299 / 30],
    )


def scene(meta: MetaClient, file_id: str = WONDER_FILE_ID) -> SceneResult:
    """Scene analysis of the synthetic clip (Part 6). `file_id` is the uploaded wonder_480.mp4;
    the default is the recorded one, whose hash is taken from media.json."""
    meta.runtime.file_hashes.setdefault(file_id, media_info()["wonder_480_sha256"])
    return analyze_scene(meta, scene_input(file_id), experiment_id="fixture")


def sam_video(meta: MetaClient, prompt: str, file_id: str = SAM_FILE_ID) -> SamResult:
    """SAM on the 3 s clip. The key uses the clip's hash, so a re-upload doesn't change it."""
    return meta.sam_video(
        file_id,
        prompt,
        media_sha256=media_info()["sam_3s_sha256"],
        frames_hint=SAM_FRAMES,
        experiment_id="fixture",
    )


def sam_sheet(meta: MetaClient) -> SamResult:
    return meta.sam_image(
        SHEET_FRAME.read_bytes(), SHEET_PROMPT, mime="image/jpeg", experiment_id="fixture"
    )


def rephrase(meta: MetaClient) -> str | None:
    return rephrase_prompt(
        meta,
        failed=["red cube"],
        description="the ball that bounces",
        frame=STILLS[1].read_bytes(),
        experiment_id="fixture",
    )


def box(meta: MetaClient) -> BoxHint | None:
    return locate_box(
        meta,
        description="the orange ball",
        frame=STILLS[1].read_bytes(),
        frame_index=150,
        frame_size=FRAME_SIZE,
        experiment_id="fixture",
    )


class Reading(BaseModel):
    number: int
    shape: str


def reviewer_reading(reviewer: ReviewerClient):
    """Spark Open reads an image with a JSON-schema response (Part 1, check 6)."""
    image = {
        "type": "image_url",
        "image_url": {"url": data_url(STILLS[1].read_bytes(), "image/jpeg")},
    }
    question = "Count the blue dots in this image, and name the shape of the big object."
    messages = [{"role": "user", "content": [{"type": "text", "text": question}, image]}]
    return reviewer.chat(messages, experiment_id="fixture", schema=Reading, max_tokens=2000)


class NoNetwork:
    """An SDK stand-in that fails the test if a replayed call reaches for the network."""

    def __getattr__(self, name: str):
        raise AssertionError(f"replay mode touched the network (sdk.{name})")


def replay_clients(monkeypatch, cache_dir: Path = CACHE_DIR):
    """(meta, reviewer, runtime) in replay mode on the recorded cache, with no keys and an SDK
    that fails on use. The model names are set to the ones recorded in media.json."""
    models = media_info()["models"]
    for name, key in [
        ("PHYSICIST_MODEL", "physicist"),
        ("SCENE_MODEL", "scene"),
        ("SAM_MODEL", "sam"),
        ("REVIEWER_MODEL", "reviewer"),
    ]:
        monkeypatch.setenv(name, models[key])
    monkeypatch.setenv("LLM_MODE", "replay")
    for name in ("MODEL_API_KEY", "REVIEWER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    runtime = Runtime.create(get_settings(), cache_dir=cache_dir)
    return MetaClient(runtime, sdk=NoNetwork()), ReviewerClient(runtime, sdk=NoNetwork()), runtime
