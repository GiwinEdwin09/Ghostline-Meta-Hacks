"""One call to the referee: Muse Spark Open on a Modal Shared Endpoint (build plan Part 14).

`ReviewerClient.chat` does the transport: Chat Completions with a `response_format` JSON schema
(the endpoint enforces it, Day 0 check 6), and the `Modal-Routing-Affinity-Key` header set to the
experiment id, so the packet stays in one container's prompt cache across rounds. This module adds
the prompt, the message layout and the outage handling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import ghostline
from ghostline.llm import (
    BudgetExceeded,
    PermanentLLMError,
    ReviewerClient,
    StructuredOutputError,
    TransientLLMError,
)
from ghostline.llm.reviewer import ChatResult
from ghostline.store.core_schemas import Review

from .packet import Packet

PROMPT_PATH = Path(ghostline.__file__).parent / "prompts" / "reviewer.md"
MAX_TOKENS = 4000


class ReviewerUnavailable(Exception):
    """The endpoint is down, unconfigured, over budget or returned unusable output."""


def load_prompt() -> tuple[str, str]:
    """(prompt text, version): the referee prompt and a short hash of it."""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


def opening_messages(packet: Packet) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": load_prompt()[0]},
        {"role": "user", "content": packet.content()},
    ]


def next_round(
    messages: list[dict[str, Any]], previous: ChatResult, packet: Packet
) -> list[dict[str, Any]]:
    """The history plus the reviewer's own last answer and the author's reply: round 2."""
    return [
        *messages,
        {"role": "assistant", "content": previous.content},
        {"role": "user", "content": packet.content()},
    ]


def review_round(
    client: ReviewerClient, messages: list[dict[str, Any]], experiment_id: str
) -> tuple[Review, ChatResult]:
    try:
        result = client.chat(
            messages, experiment_id=experiment_id, schema=Review, max_tokens=MAX_TOKENS
        )
    except (PermanentLLMError, TransientLLMError, StructuredOutputError, BudgetExceeded) as exc:
        raise ReviewerUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return result.parsed, result
