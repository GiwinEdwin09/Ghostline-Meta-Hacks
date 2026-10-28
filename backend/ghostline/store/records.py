"""Row models for the Postgres tables (build plan Part 22).

They double as API response models, so field names are the contract with the web app. Paths are
relative to the experiment folder on the Volume; the media URL is /api/media/{experiment_id}/{path}.
"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ghostline.store.core_schemas import (
    Conclusion,
    Confidence,
    ParamSpec,
    ReviewIssue,
    SceneAnalysis,
)

SystemHint = Literal["free_fall", "pendulum", "ramp", "bounce", "spin_down", "auto"]
ExperimentStatus = Literal["draft", "ingesting", "ready", "analyzing", "complete", "failed"]
Visibility = Literal["private", "public"]
TrialRole = Literal["fit", "prove"]
TrialStatus = Literal["uploaded", "ingesting", "ready", "failed"]
UploadStatus = Literal["open", "complete"]
JobKind = Literal[
    "ingest_trial", "analyze_experiment", "prove_trial", "revise", "rerun", "export_verify"
]
JobStatus = Literal["queued", "running", "succeeded", "failed"]
Phase = Literal[
    "normalizing",
    "watching",
    "tracking",
    "calibrating",
    "hypothesizing",
    "fitting",
    "peer_review",
    "diagnosing",
    "predicting",
    "rendering",
    "verifying",
]
Actor = Literal["wonder", "spark_open", "harness", "sam", "system", "user"]
EventKind = Literal[
    "observation",
    "hypothesis",
    "prediction",
    "fit",
    "rejection",
    "acceptance",
    "review",
    "response",
    "revision",
    "ghost",
    "whatif",
    "warning",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="ignore")


class DeclaredCapture(BaseModel):
    mode: Literal["normal", "slow_mo"] = "normal"
    fps: float | None = Field(default=None, gt=0, le=1000)

    @model_validator(mode="after")
    def _slow_mo_needs_fps(self) -> "DeclaredCapture":
        if self.mode == "slow_mo" and self.fps is None:
            raise ValueError("fps is required for slow-motion clips")
        return self


class Setup(BaseModel):
    plane_offset_cm: float = Field(default=2.0, ge=0, le=50)
    paper: Literal["a4", "letter"] = "a4"
    pendulum_length_cm: float | None = Field(default=None, gt=0, le=1000)


class Experiment(Record):
    id: UUID
    created_at: datetime
    title: str
    description: str = ""
    system_type: str
    status: ExperimentStatus
    declared_capture: DeclaredCapture
    setup: Setup
    scene: SceneAnalysis | None = None
    winner_model_id: str | None = None
    conclusion: Conclusion | None = None
    confidence: Confidence | None = None
    review_status: str | None = None
    is_example: bool = False
    is_benchmark: bool = False
    visibility: Visibility = "private"
    cost_usd: float | None = None  # only filled in for requests that carry the passcode
    error: str | None = None
    replay_check: dict[str, Any] | None = None


class Trim(BaseModel):
    start_s: float
    end_s: float
    suggested_start_s: float | None = None
    suggested_end_s: float | None = None
    source: Literal["auto", "user"] = "auto"


class Trial(Record):
    id: UUID
    experiment_id: UUID
    role: TrialRole
    status: TrialStatus
    original_filename: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    paths: dict[str, str] = Field(default_factory=dict)
    meta_file_ids: dict[str, str] = Field(default_factory=dict)
    probe: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    trim: Trim | None = None
    n_frames: int | None = None
    duration_s: float | None = None
    calibration: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime


class TrackRecord(Record):
    trial_id: UUID
    object_key: str
    sam_prompt: str | None = None
    cache_key: str | None = None
    parquet_path: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(Record):
    trial_id: UUID
    parquet_path: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class ModelRecord(Record):
    experiment_id: UUID
    id: str
    round: int = 0
    source: Literal["library", "wonder"] = "library"
    library_ref: str | None = None
    code: str | None = None
    name: str
    mechanism: str = ""
    equations_latex: str = ""
    params_spec: dict[str, ParamSpec] | None = None
    prior_rank: int | None = None
    refuting_signature: str = ""
    status: str = "proposed"
    rejection_reason: str | None = None


class FitParam(BaseModel):
    name: str
    value: float
    ci_low: float
    ci_high: float
    unit: str


class FitRecord(Record):
    id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    model_id: str
    trial_id: UUID
    split: dict[str, Any] = Field(default_factory=dict)
    params: list[FitParam] = Field(default_factory=list)
    train_rmse: float | None = None
    heldout_rmse: float | None = None
    heldout_ci: list[float] | None = None
    aic: float | None = None
    bic: float | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    plot_paths: dict[str, str] = Field(default_factory=dict)
    created_at: datetime | None = None


class ReviewRecord(Record):
    id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    round: int = 1
    verdict: Literal["accept", "accept_with_caveats", "challenge"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    summary: str = ""
    response: dict[str, Any] | None = None
    created_at: datetime | None = None


class GhostRender(Record):
    id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    trial_id: UUID
    model_id: str | None = None
    version: int = 0  # assigned by the repo on insert
    video_path: str | None = None
    track_path: str | None = None
    slow_video_path: str | None = None
    divergence_onset_s: float | None = None
    max_error_mm: float | None = None
    created_at: datetime | None = None


class Attachment(BaseModel):
    path: str
    kind: Literal["image", "video", "json", "file"] = "image"
    caption: str | None = None


class Ref(BaseModel):
    kind: str  # event, model, fit, review, ghost, trial
    id: str


class Event(Record):
    id: int
    experiment_id: UUID
    seq: int
    ts: datetime
    phase: str | None = None
    actor: Actor
    kind: EventKind
    title: str
    body_md: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    attachments: list[Attachment] = Field(default_factory=list)
    refs: list[Ref] = Field(default_factory=list)


class Job(Record):
    id: UUID
    experiment_id: UUID
    kind: JobKind
    status: JobStatus
    phase: Phase | None = None
    progress: float = 0.0
    params: dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    retry_of: UUID | None = None
    created_at: datetime
    started_at: datetime | None = None
    heartbeat_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
    error_code: str | None = None
    retryable: bool = True


class Upload(Record):
    id: UUID
    experiment_id: UUID
    role: TrialRole
    filename: str
    size_bytes: int
    chunk_size: int
    n_chunks: int
    received_chunks: list[int] = Field(default_factory=list)
    status: UploadStatus
    trial_id: UUID | None = None
    created_at: datetime
    completed_at: datetime | None = None


class BenchmarkEntry(BaseModel):
    system: str
    quantity: str
    truth_value: float
    truth_sigma: float | None = None
    truth_method: str = ""
    notes: str = ""


class BenchmarkRecord(BenchmarkEntry):
    experiment_id: UUID


class LlmCall(BaseModel):
    experiment_id: UUID | None = None
    phase: str | None = None
    provider: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    frames: int = 0
    cost_usd: float = 0.0
    latency_ms: int | None = None
    cache_hit: bool = False
    request_hash: str | None = None
