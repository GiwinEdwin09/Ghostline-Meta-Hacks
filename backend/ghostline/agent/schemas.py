"""Arguments of Wonder's seven tools (build plan Part 13), as Pydantic models.

`llm.function_tool` turns each into a strict function tool, so the API enforces the shape and the
orchestrator validates again on receipt (strict mode drops numeric bounds). Strict mode can't
describe free-form objects, so a parameter spec is a list of named entries, and optional values
are written as empty lists or `null` rather than missing keys.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ListLibraryArgs(BaseModel):
    system: str = Field(
        description="free_fall, projectile, pendulum, ramp, bounce, spin_down or oscillator"
    )


class TrainingDataArgs(BaseModel):
    t_start: float | None = Field(description="start time in seconds, or null for the start")
    t_end: float | None = Field(description="end time in seconds, or null for the end of training")
    max_points: int = Field(description="at most 400; 60 is plenty to read a shape")


class ViewFramesArgs(BaseModel):
    times: list[float] = Field(description="clip times in seconds, at most six")
    overlay: Literal["none", "mask", "trail"] = Field(
        description="mask outlines the tracked object; trail draws where it has been"
    )


class ParamArg(BaseModel):
    name: str
    init: float
    lower: float
    upper: float
    unit: str
    kind: Literal["physical", "initial_condition", "nuisance"]


class ProposedModel(BaseModel):
    id: str = Field(description="short id; for a library model use the library id (P1, S2, ...)")
    name: str
    mechanism: str = Field(description="one sentence a student understands")
    equations_latex: str
    library_ref: str | None = Field(description="library id, or null for a model you write")
    code: str | None = Field(
        description="Python source for a custom model, or null for a library one"
    )
    params: list[ParamArg] = Field(
        description="parameter spec of a custom model; empty for a library one"
    )
    observables: list[str] = Field(description="custom models: observable names; else empty")
    state: list[str] = Field(description="custom models: state vector names; else empty")
    constants: list[str] = Field(description="custom models: constants read from p; else empty")
    prior_rank: int = Field(description="1 is your favorite, 2 the next, and so on")
    refuting_signature: str = Field(
        description="the result that would refute this model, stated before any fit"
    )


class ProposeArgs(BaseModel):
    favorite_id: str = Field(description="id of the model you expect to win")
    favorite_why: str = Field(description="why you expect it, in your own words")
    models: list[ProposedModel]


class RunFitsArgs(BaseModel):
    model_ids: list[str]
    split: Literal["selection", "final"] = Field(
        description="selection: fit on the first 65% and score the rest; final: refit the winner"
    )


class ViewPlotArgs(BaseModel):
    plot_id: str


class RejectedArg(BaseModel):
    id: str
    reason: str
    evidence: str = Field(
        description="the held-out number or residual pattern, quoted from the harness"
    )


class QuantityArg(BaseModel):
    name: str
    value: float
    ci_low: float
    ci_high: float
    unit: str


class IssueResponse(BaseModel):
    issue_id: str
    stance: Literal["concede", "rebut", "partial"]
    response: str


class ConclusionArgs(BaseModel):
    winner_id: str
    quantities: list[QuantityArg]
    rejected: list[RejectedArg]
    confidence: Literal["high", "medium", "low"]
    caveats: list[str]
    narrative_md: str
    issue_responses: list[IssueResponse] = Field(
        description="answers to the reviewer's issues; an empty list when there is no review yet"
    )
