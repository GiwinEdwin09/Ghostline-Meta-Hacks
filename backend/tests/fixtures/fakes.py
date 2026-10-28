"""Offline stand-ins for the OpenAI SDK client, shaped like the parts `ghostline.llm` uses."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx2
import openai


class Obj:
    """A dict with attribute access and `model_dump`, like the SDK's pydantic models."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        return _wrap(value)

    def model_dump(self, mode: str = "python", exclude_none: bool = False) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Obj(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def message_item(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def response_dict(
    text: str = "",
    *,
    output: list[dict[str, Any]] | None = None,
    input_tokens: int = 100,
    cached: int = 0,
    output_tokens: int = 50,
    reasoning: int = 20,
    status: str = "completed",
    model: str = "muse-wonder-1.0",
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "model": model,
        "status": status,
        "output": output if output is not None else [message_item(text)],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning},
            "total_tokens": input_tokens + output_tokens,
        },
    }


def function_call_item(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }


def stream_events(response: dict[str, Any], terminal: str = "response.completed") -> list:
    """The events of a streamed response: created, one text delta per text part, terminal."""
    events: list[dict[str, Any]] = [{"type": "response.created", "response": {"id": "resp_1"}}]
    for item in response["output"]:
        for part in item.get("content", []) if item["type"] == "message" else []:
            events.append({"type": "response.output_text.delta", "delta": part["text"]})
    events.append({"type": terminal, "response": response})
    return events


def sam_stream(lines: list[str], frames: int | None = 90, kind: str = "video") -> list:
    """The events of a SAM response whose output is `lines` (one per frame)."""
    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    usage["video_frames_processed" if kind == "video" else "images_processed"] = frames
    text = "".join(line + "\n" for line in lines)
    events: list[dict[str, Any]] = [{"type": "response.created", "response": {"id": "resp_s"}}]
    events += [{"type": "response.output_text.delta", "delta": line + "\n"} for line in lines]
    events.append({"type": "response.output_text.done", "text": text})
    events.append({"type": "response.completed", "response": {"id": "resp_s", "usage": usage}})
    return events


def status_error(status: int, headers: dict[str, str] | None = None, code: str | None = None):
    """An `openai.APIStatusError` for an HTTP status."""
    request = httpx2.Request("POST", "https://api.meta.ai/v1/responses")
    body = {"error": {"message": f"error {status}", "type": "x", "param": None, "code": code}}
    response = httpx2.Response(status, request=request, headers=headers or {}, json=body)
    error_class = {
        400: openai.BadRequestError,
        404: openai.NotFoundError,
        429: openai.RateLimitError,
    }.get(status, openai.APIStatusError if status < 500 else openai.InternalServerError)
    return error_class(f"error {status}", response=response, body=body)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://api.meta.ai/v1"))


class FakeResponses:
    """`sdk.responses`: each call pops the next scripted step. A step is an exception to raise,
    a list of events (for `stream=True`), or a response dict."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if kwargs.get("stream"):
            return iter(Obj(event) for event in step)
        return Obj(step)


class FakeCompletions:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return Obj(step)


def chat_dict(content: str, prompt: int = 200, completion: int = 80, cached: int = 0) -> dict:
    return {
        "model": "spark-open-test",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


class FakeFiles:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append({k: v for k, v in kwargs.items() if k != "file"})
        return Obj({"id": f"file-live-{len(self.created)}"})

    def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)


class FakeSDK:
    def __init__(self, responses: list[Any] | None = None, chats: list[Any] | None = None) -> None:
        self.responses = FakeResponses(responses or [])
        self.chat = type("Chat", (), {"completions": FakeCompletions(chats or [])})()
        self.files = FakeFiles()
