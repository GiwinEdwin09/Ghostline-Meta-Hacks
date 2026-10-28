"""Appendix C core schemas.

The API serves these shapes, so they appear in the OpenAPI document and in the web app's generated
types. If the agent, scene or export modules define their own copies, make those modules import
these instead so there is one definition.
"""

from typing import Any, Literal

from pydantic import BaseModel

Confidence = Literal["high", "medium", "low"]


class TrackedObject(BaseModel):
    key: str
    sam_prompt: str
    role: Literal["moving_body", "marker", "pivot", "surface"]
    why: str


class SceneAnalysis(BaseModel):
    system_type: Literal[
        "free_fall", "projectile", "pendulum", "ramp", "bounce", "spin_down", "oscillator", "other"
    ]
    description: str
    tracked_objects: list[TrackedObject]
    calibration: dict[str, Any]  # sam_prompt, found, fully_visible, in_motion_plane, note
    object_properties: dict[str, Any]  # shape, material_guess, approx_size_cm
    initial_hypotheses: list[dict[str, Any]]  # name, mechanism, observable_signature
    quality_issues: list[dict[str, Any]]  # issue, severity, note
    proceed: bool
    reason_if_not: str | None = None


class ParamSpec(BaseModel):
    init: float
    lower: float
    upper: float
    unit: str
    kind: Literal["physical", "initial_condition", "nuisance"]


class ModelProposal(BaseModel):
    id: str
    name: str
    mechanism: str
    equations_latex: str
    library_ref: str | None = None
    code: str | None = None
    params: dict[str, ParamSpec] | None = None
    prior_rank: int
    refuting_signature: str


class Quantity(BaseModel):
    name: str
    value: float
    ci_low: float
    ci_high: float
    unit: str


class Conclusion(BaseModel):
    winner_id: str
    quantities: list[Quantity]
    rejected: list[dict[str, Any]]  # id, reason, evidence (must reference harness numbers)
    confidence: Confidence
    caveats: list[str]
    narrative_md: str


class ReviewIssue(BaseModel):
    id: str
    severity: Literal["minor", "major"]
    claim: str
    evidence: str
    requested_action: Literal["run_model", "refit", "lower_confidence", "add_caveat", "none"]
    detail: str


class Review(BaseModel):
    verdict: Literal["accept", "accept_with_caveats", "challenge"]
    issues: list[ReviewIssue]
    summary: str


class ExportedParam(BaseModel):
    name: str
    value: float
    ci95: tuple[float, float]
    unit: str
    definition: str
    from_model: str


class SimExport(BaseModel):
    schema_version: Literal["ghostline.sim/1"]
    experiment_url: str
    object: str
    surface: str | None = None
    confidence: Confidence
    mass_kg: float | None = None
    parameters: list[ExportedParam]
    randomization: dict[str, tuple[float, float]]  # from the 95% intervals
    valid_for: dict[str, Any]  # measured speed, angle and amplitude ranges
    replay_check: dict[str, Any] | None = None  # simulator, quantity, simulated, measured, passed
