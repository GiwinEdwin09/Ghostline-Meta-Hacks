"""Pydantic schemas for the ingest outputs (probe.json, timing.json, trim.json, ingest.json)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "warn", "error"]


class Notice(BaseModel):
    """A warning or note for the user, with a stable code for the UI and the reviewer."""

    code: str
    severity: Severity = "warn"
    message: str


class ProbeInfo(BaseModel):
    codec: str
    width: int  # coded size
    height: int
    rotation: int = 0  # counter-clockwise degrees to apply for display: 0, 90, 180 or 270
    display_width: int  # size after rotation
    display_height: int
    pix_fmt: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    hdr: Literal["hlg", "pq"] | None = None
    r_frame_rate: float | None = None
    avg_frame_rate: float | None = None
    nb_frames: int  # video packets, one per frame
    duration_s: float | None = None
    capture_fps_tag: float | None = None  # from a capture-rate tag such as com.android.capture.fps
    has_audio: bool = False
    tags: dict[str, str] = Field(default_factory=dict)


class DeclaredCapture(BaseModel):
    """What the user said about the capture: normal or slow-mo, and the capture frame rate."""

    mode: Literal["normal", "slow_mo", "unknown"] = "unknown"
    fps: float | None = None


class Cluster(BaseModel):
    dt_s: float
    count: int
    start_frame: int  # first frame of the longest uniform run
    stop_frame: int  # one past its last frame


class TimingModel(BaseModel):
    classification: Literal["cfr", "vfr_jitter", "mixed"]
    n_frames: int
    file_fps: float  # of the usable range, from the median frame interval
    mean_dt_s: float  # median interval in file seconds
    jitter_rel: float  # std/mean of the intervals in the usable range
    time_factor: float  # real seconds per file second
    time_factor_source: Literal["metadata_tag", "declared", "assumed_realtime"]
    real_fps: float  # frames per real second in the usable range
    usable_start: int
    usable_stop: int  # exclusive
    sigma_time_rel: float  # relative uncertainty of the time base
    retimed: bool = False
    needs_confirmation: bool = False
    clusters: list[Cluster] = Field(default_factory=list)
    n_gaps: int = 0  # intervals much longer than the median (dropped or missing frames)
    warnings: list[Notice] = Field(default_factory=list)


class TrimSuggestion(BaseModel):
    start_frame: int
    stop_frame: int  # exclusive
    start_t: float  # real seconds
    stop_t: float
    reason: Literal["motion_window", "no_motion_detected"]
    threshold: float
    n_active: int


class NormalizeInfo(BaseModel):
    filters: str
    tonemap: Literal["none", "zscale_hable", "colorspace_fallback"]
    width: int
    height: int
    n_frames: int
    max_pts_error_s: float  # master vs source timestamps, after removing a constant shift
