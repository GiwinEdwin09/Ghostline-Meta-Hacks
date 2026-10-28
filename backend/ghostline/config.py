"""Settings read from environment variables (build plan Appendix A)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

LLMMode = Literal["live", "record", "replay"]


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
    )
