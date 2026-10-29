"""GET /media/{exp}/{path}: files from the Volume with HTTP Range support (build plan Part 21).

Safari won't play video without 206 Partial Content. Verified on Starlette 1.10: FileResponse
answers single, open-ended and suffix ranges with 206, multi-range with multipart/byteranges,
unsatisfiable ranges with 416, and honors If-Range, so it is used as is. HEAD is added here, and
so is If-None-Match -> 304, which FileResponse leaves out.
"""

import mimetypes
import os
import re
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response

from ghostline.api.deps import RepoDep, ServicesDep
from ghostline.api.errors import ApiError
from ghostline.store.volume import UnsafePath, is_content_hashed

router = APIRouter()

_IMMUTABLE = "public, max-age=31536000, immutable"
# No max-age: the browser keeps a copy but revalidates it with the ETag on every use.
_REVALIDATE = "public, no-cache"

mimetypes.add_type("application/x-ndjson", ".jsonl")
mimetypes.add_type("application/octet-stream", ".parquet")
mimetypes.add_type("application/octet-stream", ".npy")


# What the browser never needs and the Volume holds anyway: the raw upload and the physicist's
# private session (Wonder's history and the held-out errors it must not see, Part 13).
_INTERNAL = re.compile(r"(^|/)(agent/|original\.[^/]*$)")


def is_internal(path: str) -> bool:
    return bool(_INTERNAL.search(path))


def _not_found() -> ApiError:
    return ApiError(404, "media_not_found", "That file doesn't exist.")


@router.get(
    "/media/{experiment_id}/{path:path}",
    response_class=FileResponse,
    responses={404: {"description": "No such file"}, 206: {"description": "Partial content"}},
)
def get_media(
    experiment_id: UUID, path: str, request: Request, services: ServicesDep, repo: RepoDep
) -> Response:
    """Serve a file from experiments/{exp}/{path}. Experiments are private by their unguessable
    id: whoever has it can read the media (a <video> tag can't send a passcode), an unknown id is
    a 404, and so is a path that is internal (the raw upload, the physicist's session)."""
    if repo.get_visibility(experiment_id) is None or is_internal(path):
        raise _not_found()
    try:
        target = services.paths.resolve_media(experiment_id, path)
    except UnsafePath:
        raise _not_found() from None
    if not target.is_file():
        # Another container may have written it since this one last looked.
        services.reload_volume()
        if not target.is_file():
            raise _not_found()

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    cache = _IMMUTABLE if is_content_hashed(path) else _REVALIDATE
    response = FileResponse(
        target,
        media_type=media_type,
        headers={"Cache-Control": cache},
        stat_result=os.stat(target),
    )
    etag = response.headers.get("etag")
    if etag and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache})
    return response


# HEAD answers with the same headers and no body. Starlette drops the body for HEAD requests.
router.add_api_route(
    "/media/{experiment_id}/{path:path}", get_media, methods=["HEAD"], include_in_schema=False
)
