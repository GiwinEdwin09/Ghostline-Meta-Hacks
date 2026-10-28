"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ghostline import __version__
from ghostline.api import health
from ghostline.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Ghostline API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/api")
    return app
