"""Health check: the database, the Volume, and the Meta API's reachability (build plan Part 21).

A component that isn't configured (no DATABASE_URL on a laptop) reports "unconfigured" and does
not make the service "degraded". The Meta API result is cached, so polling health stays cheap
and never spends money (it lists models, which is free).
"""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from ghostline import __version__
from ghostline.api.deps import ServicesDep
from ghostline.config import Settings

router = APIRouter()

Component = Literal["ok", "error", "unconfigured", "skipped"]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    llm_mode: str
    database: Component
    volume: Component
    meta_api: Component


def http_meta_ping(settings: Settings) -> Callable[[], bool] | None:
    """A ping that lists models on the Meta API, or None when there is no API key."""
    if not settings.model_api_key:
        return None
    url = settings.meta_base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.model_api_key}"}

    def ping() -> bool:
        try:
            return httpx.get(url, headers=headers, timeout=3.0).is_success
        except httpx.HTTPError:
            return False

    return ping


def _database(services) -> Component:
    if services.repo is None:
        return "unconfigured"
    try:
        services.repo.db.fetch_one("select 1 as ok")
        return "ok"
    except Exception:
        return "error"


def _volume(settings: Settings) -> Component:
    root = Path(settings.volume_root)
    if not root.is_dir():
        return "unconfigured"
    return "ok" if os.access(root, os.W_OK) else "error"


def _meta_api(services) -> Component:
    settings = services.settings
    if services.meta_ping is None:
        return "unconfigured"
    if settings.llm_mode == "replay":
        return "skipped"
    now = time.monotonic()
    cached = services.meta_health
    if cached and now - cached[0] < settings.meta_health_ttl_s:
        return cached[1]  # type: ignore[return-value]
    result: Component = "ok" if services.meta_ping() else "error"
    services.meta_health = (now, result)
    return result


@router.get("/health")
def health(services: ServicesDep) -> Health:
    database = _database(services)
    volume = _volume(services.settings)
    meta_api = _meta_api(services)
    degraded = "error" in (database, volume, meta_api)
    return Health(
        status="degraded" if degraded else "ok",
        version=__version__,
        llm_mode=services.settings.llm_mode,
        database=database,
        volume=volume,
        meta_api=meta_api,
    )
