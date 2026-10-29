"""Request and response models for the routes (build plan Part 21). The web app's TypeScript types
are generated from these, so field names are the contract. Row-shaped responses (Experiment, Trial,
Job, Event, ...) are defined in ghostline.store.records and Appendix C shapes in
ghostline.store.core_schemas."""

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ghostline.api.views import EventView, ViewStatus
from ghostline.jobs.stage_types import SimulateResult
from ghostline.store.records import (
    DeclaredCapture,
    ExperimentStatus,
    Job,
    Setup,
    SystemHint,
    TrialRole,
    Upload,
)

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _clean_text(value: str, *, multiline: bool = False) -> str:
    pattern = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]") if multiline else _CONTROL_CHARS
    return pattern.sub("", value).strip()


# --- experiments ---------------------------------------------------------------------------
class ExperimentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    system_hint: SystemHint = "auto"
    description: str = Field(default="", max_length=2000)
    declared_capture: DeclaredCapture = Field(default_factory=DeclaredCapture)
    setup: Setup = Field(default_factory=Setup)

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        value = _clean_text(value)
        if not value:
            raise ValueError("title must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def _description(cls, value: str) -> str:
        return _clean_text(value, multiline=True)


class ExperimentPatch(BaseModel):
    """Admin edits: publish an experiment as an example, rename it."""

    title: str | None = Field(default=None, min_length=1, max_length=120)
    is_example: bool | None = None
    visibility: Literal["private", "public"] | None = None

    @field_validator("title")
    @classmethod
    def _title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _clean_text(value)
        if not value:
            raise ValueError("title must not be blank")
        return value


class EventsView(BaseModel):
    """GET /experiments/{id}/events: new notebook events plus enough state to drive the stepper
    from one poll."""

    events: list[EventView]
    last_seq: int
    status: ViewStatus
    job: Job | None = None  # the active job, else the most recent one


# --- uploads -------------------------------------------------------------------------------
class UploadCreate(BaseModel):
    experiment_id: UUID
    role: TrialRole = "fit"
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, description="File size in bytes")

    @field_validator("filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        return _clean_text(value) or "upload"


class UploadInfo(BaseModel):
    upload_id: UUID
    experiment_id: UUID
    role: TrialRole
    filename: str
    size: int
    chunk_size: int
    n_chunks: int
    received_chunks: list[int]
    status: Literal["open", "complete"]
    trial_id: UUID | None = None

    @classmethod
    def from_record(cls, upload: Upload) -> "UploadInfo":
        return cls(
            upload_id=upload.id,
            experiment_id=upload.experiment_id,
            role=upload.role,
            filename=upload.filename,
            size=upload.size_bytes,
            chunk_size=upload.chunk_size,
            n_chunks=upload.n_chunks,
            received_chunks=upload.received_chunks,
            status=upload.status,
            trial_id=upload.trial_id,
        )


class ChunkReceipt(BaseModel):
    upload_id: UUID
    n: int
    size: int
    received_count: int
    n_chunks: int


class UploadComplete(BaseModel):
    trial_id: UUID
    job_id: UUID
    sha256: str
    size: int
    job: Job


# --- trials and jobs -----------------------------------------------------------------------
class TrimRequest(BaseModel):
    """Omit a field to keep the auto-suggested value."""

    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, gt=0)


class AnalyzeRequest(BaseModel):
    trial_id: UUID | None = None  # default: the newest ready fit trial


class ProveRequest(BaseModel):
    trial_id: UUID | None = None  # default: the newest ready prove trial


class ReviseRequest(BaseModel):
    ghost_id: UUID | None = None  # default: the newest ghost


class RerunRequest(BaseModel):
    split_pct: float = Field(default=0.65, ge=0.3, le=0.9)
    initial_guesses: dict[str, float] = Field(default_factory=dict, max_length=50)


# --- misc ----------------------------------------------------------------------------------
class AuthCheck(BaseModel):
    ok: Literal[True] = True


class DeleteResult(BaseModel):
    deleted: Literal[True] = True
    meta_files_deleted: int
    meta_files_failed: int


class AdminBudget(BaseModel):
    daily_usd: float
    spent_today_usd: float
    per_experiment_usd: float


class AdminCache(BaseModel):
    hit_rate: float  # of today's model calls, the share answered from the cache
    calls_24h: int


class AdminJob(Job):
    experiment_title: str


class AdminExperiment(BaseModel):
    id: UUID
    title: str
    cost_usd: float
    status: ExperimentStatus
    llm_mode: str


class AdminOverview(BaseModel):
    """GET /admin/overview: what the /admin page shows."""

    budget: AdminBudget
    cache: AdminCache
    jobs: list[AdminJob]
    experiments: list[AdminExperiment]


# --- simulate -----------------------------------------------------------------------------
class SimulateBody(BaseModel):
    """POST /experiments/{id}/simulate. `model_id` defaults to the winner; `initial` overrides
    initial-condition parameters and `params` the rest, both by parameter name."""

    model_id: str | None = None
    params: dict[str, float] = Field(default_factory=dict, max_length=50)
    initial: dict[str, float] = Field(default_factory=dict, max_length=50)
    t_end: float = Field(default=10.0, gt=0, le=120, description="Seconds to simulate")
    dt: float = Field(default=1 / 60, gt=0, description="Output spacing, seconds")


class SimulateResponse(SimulateResult):
    observables: list[list[float]]  # the series in observable order, for the chart
