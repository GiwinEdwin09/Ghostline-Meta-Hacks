"""One error shape for every failure the API reports on purpose.

    {"detail": "<message a person can read>", "code": "<stable machine code>", ...extra fields}

FastAPI's own 422 validation errors keep their default shape.
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ghostline.jobs.errors import StageError
from ghostline.store.db import DatabaseNotConfigured


class ErrorBody(BaseModel):
    detail: str
    code: str


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra

    def body(self) -> dict[str, Any]:
        return {"detail": self.message, "code": self.code, **self.extra}


# Declared on routers so the OpenAPI document (and the generated TypeScript) knows the shape.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorBody} for code in (400, 401, 404, 409, 413, 429, 501, 503, 504)
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status_code)

    @app.exception_handler(StageError)
    async def _stage_error(_: Request, exc: StageError) -> JSONResponse:
        return JSONResponse({"detail": exc.message, "code": exc.code}, status_code=422)

    @app.exception_handler(NotImplementedError)
    async def _not_implemented(_: Request, exc: NotImplementedError) -> JSONResponse:
        detail = str(exc) or "Not implemented yet."
        return JSONResponse({"detail": detail, "code": "not_implemented"}, status_code=501)

    @app.exception_handler(DatabaseNotConfigured)
    async def _no_database(_: Request, exc: DatabaseNotConfigured) -> JSONResponse:
        return JSONResponse(
            {"detail": "The database is not configured.", "code": "database_not_configured"},
            status_code=503,
        )
