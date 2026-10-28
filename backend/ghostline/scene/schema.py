"""The `SceneAnalysis` schema (build plan Appendix C) with the SAM prompt rules of Part 6.

Appendix C abbreviates some fields as plain dicts. They are typed here, with the keys its
comments list, because strict structured output can't describe a free-form object.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SystemType = Literal[
    "free_fall", "projectile", "pendulum", "ramp", "bounce", "spin_down", "oscillator", "other"
]

# SAM wants a short noun phrase naming one concrete object (Meta's guidance, Part 6).
_MAX_PROMPT_WORDS = 6
_LIST_MARKS = re.compile(r"[,;+&/]|\b(and|or)\b", re.IGNORECASE)
_NEGATION = {"not", "no", "without", "except", "non", "excluding"}
_POSITION = {
    "left", "right", "top", "bottom", "above", "below", "behind", "beside", "between", "next",
    "near", "inside", "outside", "under", "over", "upper", "lower", "front", "back",
}  # fmt: skip
_MOTION_VERBS = {
    "moving", "falling", "swinging", "rolling", "spinning", "bouncing", "sliding", "hanging",
    "flying", "dropped", "thrown", "find", "segment", "show", "track", "select", "detect",
}  # fmt: skip


def sam_prompt_problems(prompt: str) -> list[str]:
    """Why `prompt` breaks the SAM prompt rules (empty when it follows them): one concrete noun
    phrase, no verbs, questions, lists, negation or position words."""
    words = re.findall(r"[a-z0-9'-]+", prompt.lower())
    problems = []
    if not words:
        problems.append("it is empty")
    if len(words) > _MAX_PROMPT_WORDS:
        problems.append(f"it has {len(words)} words (keep it to {_MAX_PROMPT_WORDS} or fewer)")
    if "?" in prompt:
        problems.append("it is a question")
    if _LIST_MARKS.search(prompt):
        problems.append("it lists several things (one concept per prompt)")
    if _NEGATION & set(words):
        problems.append("it uses negation")
    if _POSITION & set(words):
        problems.append("it uses a position word")
    if _MOTION_VERBS & set(words):
        problems.append("it contains a verb (name the object, not what it does)")
    return problems


def checked_sam_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    problems = sam_prompt_problems(prompt)
    if problems:
        raise ValueError(
            f"SAM prompt {prompt!r} must be a short noun phrase naming one concrete object: "
            + "; ".join(problems)
        )
    return prompt


class TrackedObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    sam_prompt: str
    role: Literal["moving_body", "marker", "pivot", "surface"]
    why: str

    check_sam_prompt = field_validator("sam_prompt")(checked_sam_prompt)


class SheetInfo(BaseModel):
    """The A4 calibration sheet (`calibration` in Appendix C)."""

    model_config = ConfigDict(extra="forbid")

    sam_prompt: str
    found: bool
    fully_visible: bool
    in_motion_plane: bool
    note: str

    check_sam_prompt = field_validator("sam_prompt")(checked_sam_prompt)


class ObjectProperties(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: str
    material_guess: str
    approx_size_cm: float | None


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    mechanism: str
    observable_signature: str


class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue: str
    severity: Literal["minor", "major"]
    note: str


class SceneAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_type: SystemType
    description: str
    tracked_objects: list[TrackedObject]
    calibration: SheetInfo
    object_properties: ObjectProperties
    initial_hypotheses: list[Hypothesis]
    quality_issues: list[QualityIssue]
    proceed: bool
    reason_if_not: str | None

    @model_validator(mode="after")
    def _consistent(self) -> SceneAnalysis:
        keys = [o.key for o in self.tracked_objects]
        if len(set(keys)) != len(keys):
            raise ValueError("tracked_objects must have unique keys")
        if self.proceed:
            if not any(o.role == "moving_body" for o in self.tracked_objects):
                raise ValueError("when proceed is true, track at least one moving_body")
            if not 2 <= len(self.initial_hypotheses) <= 4:
                raise ValueError("list 2 to 4 initial_hypotheses when proceed is true")
        elif not (self.reason_if_not or "").strip():
            raise ValueError("when proceed is false, reason_if_not must say why")
        return self

    @property
    def moving_bodies(self) -> list[TrackedObject]:
        return [o for o in self.tracked_objects if o.role == "moving_body"]
