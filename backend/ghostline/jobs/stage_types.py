"""Inputs and outputs of the pipeline stages in jobs/stages.py.

Stages return these; the job handlers persist them. All paths are relative to the experiment
folder on the Volume (see store/volume.py), so they can be stored as they are.
"""

from typing import Any

from pydantic import BaseModel, Field

from ghostline.store.core_schemas import Conclusion
from ghostline.store.records import ModelRecord, TrackRecord, Trim


class IngestResult(BaseModel):
    """Part 5. `paths` keys: master, playback_720, wonder_480, audio, poster, timestamps, probe,
    timing (audio is optional). `trim` is the auto-suggested window."""

    probe: dict[str, Any]
    timing: dict[str, Any]
    n_frames: int
    duration_s: float
    paths: dict[str, str]
    meta_file_ids: dict[str, str] = Field(default_factory=dict)  # master, wonder_480
    trim: Trim
    warnings: list[str] = Field(default_factory=list)


class TrackingResult(BaseModel):
    """Part 7. One TrackRecord per tracked object (the sheet mask is tracked too)."""

    tracks: list[TrackRecord]


class CalibrationResult(BaseModel):
    """Part 8. `calibration` is the contents of calibration.json."""

    calibration: dict[str, Any]
    paths: dict[str, str] = Field(default_factory=dict)  # calib_overlay, rectified


class TrajectoryResult(BaseModel):
    """Part 9."""

    parquet_path: str
    summary: dict[str, Any]


class ReviewOutcome(BaseModel):
    """Part 14, including the respond rounds. `conclusion` is the final one, which may have a
    lower confidence or extra caveats. `review_status` is one of accepted, accepted_with_caveats,
    contested or not_reviewed."""

    review_status: str
    conclusion: Conclusion


class GhostResult(BaseModel):
    """Part 16. `version` is the number the stage used in the file names (v{n}.mp4); get it from
    ctx.repo.next_ghost_version(experiment_id). Leave it out to let the repo pick the next one."""

    version: int | None = None
    model_id: str | None = None
    video_path: str
    track_path: str
    slow_video_path: str | None = None
    divergence_onset_s: float | None = None
    max_error_mm: float | None = None


class RevisionResult(BaseModel):
    """Part 17. `ghost` is set only when the revision was adopted."""

    diagnosis_md: str
    adopted: bool
    revised_model: ModelRecord | None = None
    ghost: GhostResult | None = None


class SimulateRequest(BaseModel):
    """POST /experiments/{id}/simulate (Part 18): run one model, library or Wonder-written."""

    model_id: str
    params: dict[str, float] = Field(default_factory=dict)  # overrides by parameter name
    t_end_s: float = Field(default=10.0, gt=0, le=120)
    n_points: int = Field(default=600, ge=2, le=5000)


class SimulateResult(BaseModel):
    model_id: str
    t: list[float]
    series: dict[str, list[float]]  # observable name -> values at each t
    derived: dict[str, float] = Field(default_factory=dict)  # period, time to stop, ...
