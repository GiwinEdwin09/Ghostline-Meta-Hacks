"""Builders for Responses API input items, tools and tool outputs.

All of these are plain dicts, which is what the agent loop passes back in as history.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .structured import strict_schema


def data_url(data: bytes | str | Path, mime: str | None = None) -> str:
    """A base64 data URL for image bytes or an image file."""
    if isinstance(data, str | Path):
        mime = mime or mimetypes.guess_type(str(data))[0]
        data = Path(data).read_bytes()
    return f"data:{mime or 'image/png'};base64,{base64.b64encode(data).decode()}"


def input_text(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}


def input_image(image: bytes | str | Path, mime: str | None = None) -> dict[str, Any]:
    """An image part. A string starting with `data:` or `http` is used as the URL as it is."""
    if isinstance(image, str) and image.startswith(("data:", "http")):
        return {"type": "input_image", "image_url": image}
    return {"type": "input_image", "image_url": data_url(image, mime)}


def input_video(file_id: str) -> dict[str, Any]:
    return {"type": "input_video", "file_id": file_id}


def user_message(text: str, *parts: dict[str, Any]) -> dict[str, Any]:
    """A user message with text followed by any media parts."""
    return {"type": "message", "role": "user", "content": [input_text(text), *parts]}


def function_tool(name: str, description: str, params: type[BaseModel]) -> dict[str, Any]:
    """A strict function tool whose arguments are validated by `params`."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": strict_schema(params),
        "strict": True,
    }


def function_call_output(call_id: str, output: str | list[dict[str, Any]]) -> dict[str, Any]:
    """A tool result. `output` is text, or a list of parts such as `input_image` (tool outputs
    can carry images; Part 1, check 3)."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def as_items(value: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A plain string becomes one user message."""
    return [user_message(value)] if isinstance(value, str) else list(value)
