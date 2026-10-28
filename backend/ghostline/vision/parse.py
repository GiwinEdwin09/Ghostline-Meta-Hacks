"""Glue between recorded SAM event streams and `meta-sam-parser`.

The browser replays the same events into `@meta-sam/parser`; here the Python parser reads them
synchronously, so a recorded stream parses offline, identically on every run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from meta_sam_parser import (
    CompletedOutcome,
    IncompleteOutcome,
    SegmentationResult,
    image_segmentation_format,
    video_segmentation_format,
)

_FRAME_SIZE = re.compile(r"w=(\d+);h=(\d+)")


class SamStreamError(Exception):
    """The SAM stream reported a failure, or its output could not be parsed."""


def read_events_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a raw event stream saved by `SamResult.write_events_jsonl`."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def output_text_of(events: Iterable[dict[str, Any]]) -> tuple[str, str]:
    """(output text, terminal status) of a raw event stream. The text is the concatenated
    deltas, or the finalized text part when no deltas were sent. Status is `completed`,
    `incomplete` or `eof`; a failed response raises `SamStreamError`."""
    deltas: list[str] = []
    final_text = ""
    status = "eof"
    for event in events:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            deltas.append(event.get("delta", ""))
        elif kind == "response.output_text.done":
            final_text = event.get("text", final_text)
        elif kind == "response.content_part.done":
            part = event.get("part") or {}
            if part.get("type") == "output_text":
                final_text = final_text or part.get("text", "")
        elif kind == "response.completed":
            status = "completed"
        elif kind == "response.incomplete":
            status = "incomplete"
        elif kind in ("response.failed", "error"):
            error = (event.get("response") or {}).get("error") or event
            raise SamStreamError(f"SAM stream failed: {error.get('message', error)}")
    return ("".join(deltas) or final_text), status


def parse_sam_output(
    text: str, status: str = "completed", media: str = "video"
) -> SegmentationResult:
    """Parse SAM output text (one line per frame for video) into the parser's final result."""
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text)
    outcome = (
        CompletedOutcome()
        if status == "completed"
        else IncompleteOutcome(reason="response" if status == "incomplete" else "eof")
    )
    return parser.finish(outcome).result


def parse_sam_events(events: Iterable[dict[str, Any]], media: str = "video") -> SegmentationResult:
    """Parse a raw event stream, as recorded in a cache entry or an events JSONL file."""
    text, status = output_text_of(events)
    return parse_sam_output(text, status, media)


def frame_size_of(text: str) -> tuple[int, int] | None:
    """(width, height) the stream reports in its first box token (`w=..;h=..`)."""
    match = _FRAME_SIZE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None
