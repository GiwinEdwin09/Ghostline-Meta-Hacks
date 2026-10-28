"""FastAPI application factory (build plan Part 21).

Everything the routes touch (database, Volume, job spawner, Meta file deleter) is passed in or
built from settings, so tests and local runs can swap it:

    create_app()                            # production: settings from the environment
    create_app(spawner=ModalSpawner(...))   # modal_app.py: spawn Modal Functions
    create_app(db=..., spawner=InlineSpawner(...), volume=NullVolume())   # tests
"""

import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic.json_schema import models_json_schema
from starlette.types import ASGIApp, Receive, Scope, Send

from ghostline import __version__
from ghostline.api import experiments, health, media, misc, uploads
from ghostline.api.deps import FailureLimiter, Services
from ghostline.api.errors import install_error_handlers
from ghostline.config import Settings, get_settings
from ghostline.jobs.log import configure_logging
from ghostline.jobs.runner import run_job
from ghostline.jobs.spawn import JobSpawner, ThreadSpawner
from ghostline.store.core_schemas import ModelProposal, Review, ReviewIssue
from ghostline.store.db import Database
from ghostline.store.meta_files import FileDeleter, HttpFileDeleter, NoFileDeleter
from ghostline.store.repo import Repo
from ghostline.store.volume import NullVolume, VolumePaths, VolumeSync

log = logging.getLogger("ghostline.api")

_CHUNK_PATH = re.compile(r"^/api/uploads/[^/]+/chunks/\d+$")


class BodyLimitMiddleware:
    """Refuse JSON requests that declare a body over the limit. Chunk uploads are exempt (their
    route checks the size itself). Only the declared Content-Length is checked."""

    def __init__(self, app: ASGIApp, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not _CHUNK_PATH.match(scope["path"]):
            for name, value in scope["headers"]:
                if name == b"content-length" and value.isdigit() and int(value) > self.limit:
                    body = json.dumps(
                        {"detail": "That request is too large.", "code": "body_too_large"}
                    ).encode()
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 413,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
        await self.app(scope, receive, send)


# Appendix C shapes that no route returns directly (the API returns their stored forms), still
# published in the OpenAPI document so the web app can generate types for them.
_EXTRA_SCHEMAS = (Review, ReviewIssue, ModelProposal)


def _publish_extra_schemas(app: FastAPI) -> None:
    default_openapi = app.openapi

    def openapi() -> dict:
        if app.openapi_schema:
            return app.openapi_schema
        schema = default_openapi()
        _, definitions = models_json_schema(
            [(m, "serialization") for m in _EXTRA_SCHEMAS],
            ref_template="#/components/schemas/{model}",
        )
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, definition in definitions.get("$defs", {}).items():
            components.setdefault(name, definition)
        return schema

    app.openapi = openapi  # type: ignore[method-assign]


def _operation_id(route: APIRoute) -> str:
    return route.name  # createExperiment-style ids for the generated TypeScript client


def build_services(
    settings: Settings,
    *,
    db: Database | None,
    spawner: JobSpawner | None,
    volume: VolumeSync | None,
    file_deleter: FileDeleter | None,
    meta_ping: Callable[[], bool] | None,
) -> Services:
    repo = Repo(db) if db is not None else None
    paths = VolumePaths(settings.volume_root)
    volume = volume or NullVolume()
    if spawner is None:
        # No Modal here (local development): run jobs in a thread of this process.
        def run(job_id):
            run_job(job_id, repo=repo, paths=paths, volume=volume, settings=settings)

        spawner = ThreadSpawner(run)
    if file_deleter is None:
        file_deleter = HttpFileDeleter(settings) if settings.model_api_key else NoFileDeleter()
    return Services(
        settings=settings,
        paths=paths,
        volume=volume,
        spawner=spawner,
        file_deleter=file_deleter,
        repo=repo,
        meta_ping=meta_ping,
        passcode_failures=FailureLimiter(settings.passcode_failures_per_10min),
    )


def create_app(
    *,
    settings: Settings | None = None,
    db: Database | None = None,
    spawner: JobSpawner | None = None,
    volume: VolumeSync | None = None,
    file_deleter: FileDeleter | None = None,
    meta_ping: Callable[[], bool] | None = None,
) -> FastAPI:
    """`meta_ping` checks Meta API reachability for /health; by default it lists models when
    MODEL_API_KEY is set."""
    settings = settings or get_settings()
    configure_logging()
    owns_db = db is None and bool(settings.database_url)
    if owns_db:
        db = Database(
            settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max
        )
    ping = meta_ping or health.http_meta_ping(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_db and db is not None:
            db.close()

    app = FastAPI(
        title="Ghostline API",
        version=__version__,
        lifespan=lifespan,
        generate_unique_id_function=_operation_id,
    )
    app.state.services = build_services(
        settings, db=db, spawner=spawner, volume=volume, file_deleter=file_deleter, meta_ping=ping
    )
    install_error_handlers(app)
    app.add_middleware(BodyLimitMiddleware, limit=settings.json_body_limit_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "ETag"],
    )
    for module in (health, experiments, uploads, media, misc):
        app.include_router(module.router, prefix="/api")
    _publish_extra_schemas(app)
    return app
