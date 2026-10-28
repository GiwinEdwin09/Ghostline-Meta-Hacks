"""Settings read from environment variables (build plan Appendix A, Parts 21, 22, 27 and 28)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

LLMMode = Literal["live", "record", "replay"]

MIB = 1024 * 1024
GIB = 1024 * MIB


@dataclass(frozen=True)
class Settings:
    model_api_key: str
    meta_base_url: str
    physicist_model: str
    scene_model: str
    sam_model: str
    reviewer_base_url: str
    reviewer_model: str
    reviewer_api_key: str
    database_url: str
    app_passcode: str
    budget_usd_daily: float
    budget_usd_per_experiment: float
    llm_mode: LLMMode
    cors_origins: list[str]
    # Storage (Part 22). The Modal Volume is mounted here; locally it is any directory.
    volume_root: str = "/data"
    db_pool_min: int = 1
    db_pool_max: int = 5
    # Uploads and abuse limits (Parts 4 and 28).
    max_upload_bytes: int = GIB
    upload_chunk_bytes: int = 8 * MIB
    max_frames: int = 15_000
    max_open_uploads_per_experiment: int = 10
    json_body_limit_bytes: int = MIB
    analyses_per_ip_per_day: int = 10
    passcode_failures_per_10min: int = 20
    # Jobs (Part 21).
    modal_app_name: str = "ghostline"
    job_heartbeat_interval_s: float = 5.0
    job_heartbeat_timeout_s: float = 60.0
    job_queued_timeout_s: float = 180.0
    simulate_timeout_s: float = 20.0
    # Health check (Part 21): how long a Meta API reachability result is reused.
    meta_health_ttl_s: float = 60.0
    # Chunks of an upload that was started and never completed are removed after this long.
    stale_upload_hours: float = 24.0


@lru_cache
def get_settings() -> Settings:
    env = os.environ.get
    llm_mode = env("LLM_MODE", "live")
    if llm_mode not in ("live", "record", "replay"):
        raise ValueError(f"LLM_MODE must be live, record or replay, got {llm_mode!r}")
    return Settings(
        model_api_key=env("MODEL_API_KEY", ""),
        meta_base_url=env("META_BASE_URL", "https://api.meta.ai/v1"),
        physicist_model=env("PHYSICIST_MODEL", "muse-wonder-1.0"),
        scene_model=env("SCENE_MODEL", "muse-wonder-1.0"),
        sam_model=env("SAM_MODEL", "sam-3.1"),
        reviewer_base_url=env("REVIEWER_BASE_URL", ""),
        reviewer_model=env("REVIEWER_MODEL", ""),
        reviewer_api_key=env("REVIEWER_API_KEY", ""),
        database_url=env("DATABASE_URL", ""),
        app_passcode=env("APP_PASSCODE", ""),
        budget_usd_daily=float(env("BUDGET_USD_DAILY", "20")),
        budget_usd_per_experiment=float(env("BUDGET_USD_PER_EXPERIMENT", "3")),
        llm_mode=llm_mode,
        cors_origins=[
            origin.strip()
            for origin in env("CORS_ORIGINS", "http://localhost:3000").split(",")
            if origin.strip()
        ],
        volume_root=env("VOLUME_ROOT", "/data"),
        db_pool_min=int(env("DB_POOL_MIN", "1")),
        db_pool_max=int(env("DB_POOL_MAX", "5")),
        max_upload_bytes=int(env("MAX_UPLOAD_BYTES", str(GIB))),
        upload_chunk_bytes=int(env("UPLOAD_CHUNK_BYTES", str(8 * MIB))),
        max_frames=int(env("MAX_FRAMES", "15000")),
        max_open_uploads_per_experiment=int(env("MAX_OPEN_UPLOADS_PER_EXPERIMENT", "10")),
        json_body_limit_bytes=int(env("JSON_BODY_LIMIT_BYTES", str(MIB))),
        analyses_per_ip_per_day=int(env("ANALYSES_PER_IP_PER_DAY", "10")),
        passcode_failures_per_10min=int(env("PASSCODE_FAILURES_PER_10MIN", "20")),
        modal_app_name=env("MODAL_APP_NAME", "ghostline"),
        job_heartbeat_interval_s=float(env("JOB_HEARTBEAT_INTERVAL_S", "5")),
        job_heartbeat_timeout_s=float(env("JOB_HEARTBEAT_TIMEOUT_S", "60")),
        job_queued_timeout_s=float(env("JOB_QUEUED_TIMEOUT_S", "180")),
        simulate_timeout_s=float(env("SIMULATE_TIMEOUT_S", "20")),
        meta_health_ttl_s=float(env("META_HEALTH_TTL_S", "60")),
        stale_upload_hours=float(env("STALE_UPLOAD_HOURS", "24")),
    )
