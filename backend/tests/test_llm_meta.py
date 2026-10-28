import json

import pytest
from fixtures.fakes import (
    FakeSDK,
    function_call_item,
    response_dict,
    sam_stream,
    status_error,
    stream_events,
)
from pydantic import BaseModel

from ghostline.config import get_settings
from ghostline.llm import (
    InMemorySpendStore,
    MetaClient,
    PermanentLLMError,
    ReplayMissError,
    Runtime,
    StructuredOutputError,
    TransientLLMError,
    function_call_output,
    function_tool,
    input_image,
    user_message,
)
from ghostline.llm.pricing import WONDER


class Gravity(BaseModel):
    planet: str
    g_m_s2: float


GOOD = json.dumps({"planet": "Mars", "g_m_s2": 3.71})
BAD = json.dumps({"planet": "Mars"})


def make(sdk, tmp_path, mode="live"):
    runtime = Runtime.create(
        get_settings(),
        mode=mode,
        cache_dir=tmp_path,
        store=InMemorySpendStore(),
        sleep=lambda s: None,
    )
    return MetaClient(runtime, sdk=sdk)


def test_a_streamed_call_returns_text_usage_and_calls_back(tmp_path):
    reply = response_dict("hello", input_tokens=40, output_tokens=12, reasoning=5)
    sdk = FakeSDK([stream_events(reply)])
    meta = make(sdk, tmp_path)
    deltas = []

    response = meta.respond(
        input="say hello",
        instructions="be brief",
        reasoning_effort="low",
        max_output_tokens=900,
        on_text=deltas.append,
    )

    assert response.output_text == "hello" and deltas == ["hello"]
    assert (response.usage.input_tokens, response.usage.reasoning_tokens) == (40, 5)
    (call,) = sdk.responses.calls
    assert call["stream"] is True
    assert call["model"] == "muse-wonder-1.0"
    assert call["reasoning"] == {"effort": "low"}
    assert call["max_output_tokens"] == 900 and call["instructions"] == "be brief"
    assert call["input"] == [user_message("say hello")]


def test_a_non_streamed_call(tmp_path):
    sdk = FakeSDK([response_dict("plain")])
    meta = make(sdk, tmp_path)

    response = meta.respond(input="q", stream=False)

    assert response.output_text == "plain"
    assert "stream" not in sdk.responses.calls[0]


def test_physicist_and_scene_models_come_from_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSICIST_MODEL", "muse-spark-1.3")
    sdk = FakeSDK([stream_events(response_dict("a")), stream_events(response_dict("b"))])
    meta = make(sdk, tmp_path)

    meta.respond(input="q")
    meta.respond(input="q", model="muse-wonder-1.0")

    assert [c["model"] for c in sdk.responses.calls] == ["muse-spark-1.3", "muse-wonder-1.0"]


def test_tool_calls_and_history_for_the_agent_loop(tmp_path):
    class Plot(BaseModel):
        name: str

    tool = function_tool("get_plot", "Return a plot.", Plot)
    first_reply = response_dict(output=[function_call_item("get_plot", {"name": "residuals"})])
    sdk = FakeSDK([stream_events(first_reply), stream_events(response_dict("it is flat"))])
    meta = make(sdk, tmp_path)
    question = user_message("show me residuals")

    first = meta.respond(input=[question], tools=[tool])
    (call,) = first.function_calls
    assert (call.name, call.call_id, call.args()) == ("get_plot", "call_1", {"name": "residuals"})
    image = input_image(b"\x89PNG", "image/png")
    meta.respond(
        input=[question, *first.output, function_call_output(call.call_id, [image])], tools=[tool]
    )

    sent = sdk.responses.calls[1]["input"]
    assert sent[1]["type"] == "function_call"  # the output items went back as history
    assert sent[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": [{"type": "input_image", "image_url": image["image_url"]}],
    }
    assert sdk.responses.calls[0]["tools"][0]["strict"] is True


def test_truncation_is_reported(tmp_path):
    reply = response_dict("{ cut", status="incomplete")
    reply["incomplete_details"] = {"reason": "max_output_tokens"}
    meta = make(FakeSDK([stream_events(reply, "response.incomplete")]), tmp_path)

    response = meta.respond(input="q")

    assert response.truncated and response.incomplete_reason == "max_output_tokens"


def test_structured_output_is_validated_and_parsed(tmp_path):
    sdk = FakeSDK([stream_events(response_dict(GOOD))])
    meta = make(sdk, tmp_path)

    response = meta.respond(input="q", text_format=Gravity)

    assert response.parsed == Gravity(planet="Mars", g_m_s2=3.71) and response.repairs == 0
    fmt = sdk.responses.calls[0]["text"]["format"]
    assert (fmt["type"], fmt["name"], fmt["strict"]) == ("json_schema", "Gravity", True)
    assert fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["required"] == ["planet", "g_m_s2"]


def test_an_invalid_reply_gets_exactly_one_repair_call(tmp_path):
    sdk = FakeSDK([stream_events(response_dict(BAD)), stream_events(response_dict(GOOD))])
    meta = make(sdk, tmp_path)

    response = meta.respond(input="q", text_format=Gravity)

    assert response.parsed.g_m_s2 == 3.71 and response.repairs == 1
    assert len(sdk.responses.calls) == 2
    repair_input = sdk.responses.calls[1]["input"]
    assert repair_input[1]["type"] == "message"  # the bad reply is shown back
    assert "g_m_s2" in repair_input[-1]["content"][0]["text"]  # with the validation error
    assert response.usage.input_tokens == 200  # both calls' tokens are counted
    assert response.cost_usd == pytest.approx(2 * WONDER.cost(100, 0, 50))


def test_a_reply_that_is_still_invalid_raises(tmp_path):
    sdk = FakeSDK([stream_events(response_dict(BAD)), stream_events(response_dict("not json"))])
    meta = make(sdk, tmp_path)

    with pytest.raises(StructuredOutputError):
        meta.respond(input="q", text_format=Gravity)

    assert len(sdk.responses.calls) == 2  # never a second repair


def test_a_fenced_reply_is_accepted(tmp_path):
    meta = make(FakeSDK([stream_events(response_dict(f"```json\n{GOOD}\n```"))]), tmp_path)

    assert meta.respond(input="q", text_format=Gravity).parsed.planet == "Mars"


def test_transient_errors_are_retried_and_a_400_is_not(tmp_path):
    sdk = FakeSDK([status_error(429), status_error(503), stream_events(response_dict("ok"))])
    assert make(sdk, tmp_path).respond(input="q").output_text == "ok"
    assert len(sdk.responses.calls) == 3

    sdk = FakeSDK([status_error(400), stream_events(response_dict("ok"))])
    with pytest.raises(PermanentLLMError):
        make(sdk, tmp_path).respond(input="q")
    assert len(sdk.responses.calls) == 1


def test_four_failures_give_up(tmp_path):
    sdk = FakeSDK([status_error(503)] * 4)

    with pytest.raises(TransientLLMError):
        make(sdk, tmp_path).respond(input="q")


def test_a_stream_without_a_terminal_event_is_retried(tmp_path):
    cut = stream_events(response_dict("ok"))[:-1]
    sdk = FakeSDK([cut, stream_events(response_dict("ok"))])

    assert make(sdk, tmp_path).respond(input="q").output_text == "ok"
    assert len(sdk.responses.calls) == 2


def test_a_failed_response_event_with_a_permanent_code_raises(tmp_path):
    failed = response_dict("")
    failed["error"] = {"code": "content_policy_violation", "message": "no"}
    sdk = FakeSDK([stream_events(failed, "response.failed")])

    with pytest.raises(PermanentLLMError, match="content_policy_violation"):
        make(sdk, tmp_path).respond(input="q")


def test_record_then_replay_is_byte_identical(tmp_path):
    reply = response_dict(GOOD, input_tokens=77, cached=10, output_tokens=33, reasoning=20)
    recorder = make(FakeSDK([stream_events(reply)]), tmp_path, mode="record")
    recorded = recorder.respond(input="q", text_format=Gravity, reasoning_effort="low")

    replayer = make(FakeSDK([]), tmp_path, mode="replay")  # a script with no steps: no calls
    replayed = replayer.respond(input="q", text_format=Gravity, reasoning_effort="low")

    assert replayed.cache_hit and not recorded.cache_hit
    assert replayed.model_dump_json(exclude={"cache_hit"}) == recorded.model_dump_json(
        exclude={"cache_hit"}
    )
    assert replayed.parsed == recorded.parsed


def test_live_mode_writes_no_cache(tmp_path):
    meta = make(FakeSDK([stream_events(response_dict("x"))]), tmp_path, mode="live")

    meta.respond(input="q")

    assert not list(tmp_path.rglob("*.json"))


def test_replay_without_a_recording_says_how_to_record(tmp_path):
    meta = make(FakeSDK([]), tmp_path, mode="replay")

    with pytest.raises(ReplayMissError, match="LLM_MODE=record"):
        meta.respond(input="never recorded")


def test_a_different_request_is_a_different_recording(tmp_path):
    recorder = make(FakeSDK([stream_events(response_dict("a"))]), tmp_path, mode="record")
    recorder.respond(input="q", reasoning_effort="low")
    replayer = make(FakeSDK([]), tmp_path, mode="replay")

    with pytest.raises(ReplayMissError):
        replayer.respond(input="q", reasoning_effort="high")


def test_recorded_requests_hold_no_keys_or_image_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-super-secret")
    recorder = make(FakeSDK([stream_events(response_dict("a"))]), tmp_path, mode="record")
    image = input_image(b"\x89PNGsecretpixels", "image/png")

    recorder.respond(input=[user_message("look", image)])

    text = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert "sk-super-secret" not in text and "secretpixels" not in text
    assert "sha256=" in text


def test_upload_uses_user_data_and_a_seven_day_expiry(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    sdk = FakeSDK([])
    meta = make(sdk, tmp_path)

    uploaded = meta.upload_file(clip)

    (call,) = sdk.files.created
    assert call["purpose"] == "user_data"
    assert call["expires_after"] == {"anchor": "created_at", "seconds": 604800}
    assert uploaded.id == "file-live-1" and len(uploaded.sha256) == 64
    assert meta.runtime.file_hashes[uploaded.id] == uploaded.sha256
    meta.delete_file(uploaded.id)
    assert sdk.files.deleted == ["file-live-1"]


def test_upload_in_replay_mode_touches_no_network_and_is_content_addressed(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"same bytes")
    other = tmp_path / "copy.mp4"
    other.write_bytes(b"same bytes")
    meta = make(None, tmp_path, mode="replay")

    first, second = meta.upload_file(clip), meta.upload_file(other)

    assert first.id == second.id and first.id.startswith("file-replay-")
    meta.delete_file(first.id)  # a no-op


def test_a_reupload_of_the_same_clip_hits_the_same_recording(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"master video bytes")
    events = sam_stream(["<0f>0"], frames=90)
    recorder = make(FakeSDK([events]), tmp_path, mode="record")
    first_id = recorder.upload_file(clip).id
    recorded = recorder.sam_video(
        first_id, "orange ball", media_sha256=recorder.runtime.file_hashes[first_id]
    )

    replayer = make(None, tmp_path, mode="replay")
    replayed = replayer.sam_video(replayer.upload_file(clip).id, "orange ball")

    assert first_id != replayer.upload_file(clip).id
    assert replayed.cache_hit and replayed.output_text == recorded.output_text


def test_sam_video_request_and_result(tmp_path):
    lines = ["<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|><|mask;x=0;y=0;data=2,2,~xx|>"]
    sdk = FakeSDK([sam_stream(lines, frames=90)])
    meta = make(sdk, tmp_path)

    result = meta.sam_video(
        "file-9", "orange foam ball", metadata={"propagation_direction": "both"}
    )

    (call,) = sdk.responses.calls
    assert call["model"] == "sam-3.1" and call["stream"] is True
    assert call["metadata"] == {"include_confidence": "true", "propagation_direction": "both"}
    parts = call["input"][0]["content"]
    assert parts == [
        {"type": "input_text", "text": "orange foam ball"},
        {"type": "input_video", "file_id": "file-9"},
    ]
    assert result.output_text == lines[0] + "\n"
    assert result.frames_processed == 90
    assert [e["type"] for e in result.events][0] == "response.created"


def test_sam_events_are_written_as_jsonl(tmp_path):
    meta = make(FakeSDK([sam_stream(["<0f>0", "<1f>0"])]), tmp_path)
    result = meta.sam_video("file-9", "orange ball")

    path = result.write_events_jsonl(tmp_path / "sam" / "ball.events.jsonl")

    lines = path.read_text().splitlines()
    assert len(lines) == len(result.events)
    assert json.loads(lines[1])["type"] == "response.output_text.delta"


def test_sam_pricing_falls_back_to_the_frame_hint(tmp_path):
    events = sam_stream(["<0f>0"], frames=None)
    for event in events:
        if event["type"] == "response.completed":
            event["response"]["usage"].pop("video_frames_processed", None)
    meta = make(FakeSDK([events]), tmp_path)

    result = meta.sam_video("file-9", "orange ball", frames_hint=300)

    assert result.frames_processed == 300 and result.cost_usd == pytest.approx(0.06)
