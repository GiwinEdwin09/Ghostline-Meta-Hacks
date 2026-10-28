"""Wonder's help for the SAM retry ladder (build plan Part 7): a rephrased concept prompt, then a
bounding box. These are the callables `ghostline.vision.RescueHooks` takes.

    hooks = make_rescue_hooks(meta, description="an orange ball", frame_png=..., frames={...},
                              frame_size=(1920, 1080))
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ghostline.llm import MetaClient, input_image, user_message
from ghostline.vision.tracker import BoxHint, RescueHooks

from .schema import checked_sam_prompt

REPHRASE_PROMPT = """\
SAM 3.1 could not find "{prompt}" in this video. It found nothing, or the wrong object. Look at
the frame and write a different SAM prompt for the same object: a short noun phrase naming one
concrete object, with a distinguishing color or material. No verbs, questions, lists, negation
or position words. Do not repeat a prompt that already failed: {failed}.
Object to track: {description}"""

BOX_PROMPT = """\
Draw a bounding box around the object to track on the frame shown: {description}.
The frame is {width} x {height} pixels. Give pixel coordinates in that frame: x1, y1 is the
top-left corner and x2, y2 the bottom-right corner, with x2 > x1 and y2 > y1. Set found to false
if the object is not visible on this frame."""


class PromptSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sam_prompt: str
    reasoning: str

    check_sam_prompt = field_validator("sam_prompt")(checked_sam_prompt)


class BoxLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    x1: int
    y1: int
    x2: int
    y2: int

    @model_validator(mode="after")
    def _ordered(self) -> BoxLocation:
        if self.found and not (self.x2 > self.x1 and self.y2 > self.y1):
            raise ValueError("the box needs x2 > x1 and y2 > y1")
        return self


def rephrase_prompt(
    meta: MetaClient,
    *,
    failed: list[str],
    description: str,
    frame: bytes | Path,
    experiment_id: str | None = None,
) -> str | None:
    """Ask Wonder for a different concept prompt, showing it one frame. None if it repeats a
    failed prompt."""
    text = REPHRASE_PROMPT.format(
        prompt=failed[-1], failed=", ".join(f'"{p}"' for p in failed), description=description
    )
    reply = meta.respond(
        model=meta.settings.scene_model,
        input=[user_message(text, input_image(frame))],
        text_format=PromptSuggestion,
        reasoning_effort="low",
        experiment_id=experiment_id,
        phase="rescue",
    )
    suggestion = reply.parsed.sam_prompt
    return None if suggestion.lower() in {p.lower() for p in failed} else suggestion


def locate_box(
    meta: MetaClient,
    *,
    description: str,
    frame: bytes | Path,
    frame_index: int,
    frame_size: tuple[int, int],
    experiment_id: str | None = None,
) -> BoxHint | None:
    """Ask Wonder for a box around the object on one frame, in the master's pixel coordinates.
    None if Wonder can't see it or the box falls outside the frame."""
    width, height = frame_size
    text = BOX_PROMPT.format(description=description, width=width, height=height)
    reply = meta.respond(
        model=meta.settings.scene_model,
        input=[user_message(text, input_image(frame))],
        text_format=BoxLocation,
        reasoning_effort="low",
        experiment_id=experiment_id,
        phase="rescue",
    )
    box = reply.parsed
    if not box.found:
        return None
    left, top = max(box.x1, 0), max(box.y1, 0)
    right, bottom = min(box.x2, width), min(box.y2, height)
    if right <= left or bottom <= top:
        return None
    return BoxHint(frame_index, left, top, right, bottom)


def make_rescue_hooks(
    meta: MetaClient,
    *,
    description: str,
    frames: dict[int, bytes],
    frame_size: tuple[int, int],
    experiment_id: str | None = None,
) -> RescueHooks:
    """Hooks for `track_object`. `frames` maps master frame indices to encoded stills; the first
    is shown to Wonder for the rephrase and the box request."""
    index = min(frames)
    failed: list[str] = []

    def rephrase(previous: str, attempt: int) -> str | None:
        if previous not in failed:
            failed.append(previous)
        return rephrase_prompt(
            meta,
            failed=failed,
            description=description,
            frame=frames[index],
            experiment_id=experiment_id,
        )

    def box() -> BoxHint | None:
        return locate_box(
            meta,
            description=description,
            frame=frames[index],
            frame_index=index,
            frame_size=frame_size,
            experiment_id=experiment_id,
        )

    return RescueHooks(rephrase=rephrase, locate_box=box)
