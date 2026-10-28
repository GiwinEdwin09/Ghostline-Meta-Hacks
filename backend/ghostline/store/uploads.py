"""File side of chunked uploads (build plan Part 4): store chunks, assemble, hash.

Filenames from the client are never used as paths. Only a short extension from an allowlist
survives, and everything else is named by id.
"""

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from uuid import UUID

from ghostline.store.records import Upload
from ghostline.store.volume import VolumePaths

VIDEO_EXTENSIONS = {"mp4", "mov", "m4v", "webm", "mkv", "avi", "3gp", "3g2", "mts", "ts"}
_COPY_BUFFER = 1024 * 1024


class UploadIncomplete(Exception):
    def __init__(self, missing: list[int]):
        super().__init__(f"{len(missing)} chunk(s) missing")
        self.missing = missing


def extension_for(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext if ext in VIDEO_EXTENSIONS else "bin"


def n_chunks_for(size_bytes: int, chunk_size: int) -> int:
    return -(-size_bytes // chunk_size)


def expected_chunk_size(upload: Upload, n: int) -> int:
    """Every chunk is chunk_size bytes except the last, which holds the remainder."""
    if n < upload.n_chunks - 1:
        return upload.chunk_size
    return upload.size_bytes - upload.chunk_size * (upload.n_chunks - 1)


def write_chunk(paths: VolumePaths, upload_id: UUID, n: int, data: bytes) -> None:
    """Write chunk n atomically, so a retried or concurrent PUT never leaves a torn file."""
    target = paths.upload_part(upload_id, n)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{n}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def missing_chunks(paths: VolumePaths, upload: Upload) -> list[int]:
    """Chunks that are absent or the wrong size on disk."""
    missing = []
    for n in range(upload.n_chunks):
        part = paths.upload_part(upload.id, n)
        try:
            ok = part.stat().st_size == expected_chunk_size(upload, n)
        except FileNotFoundError:
            ok = False
        if not ok:
            missing.append(n)
    return missing


def assemble(paths: VolumePaths, upload: Upload, destination: Path) -> tuple[str, int]:
    """Concatenate the chunks into destination (written to a temp name, then renamed) and return
    (sha256 hex digest, size in bytes). Raises UploadIncomplete if any chunk is missing."""
    missing = missing_chunks(paths, upload)
    if missing:
        raise UploadIncomplete(missing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
    digest = hashlib.sha256()
    total = 0
    try:
        with tmp.open("wb") as out:
            for n in range(upload.n_chunks):
                with paths.upload_part(upload.id, n).open("rb") as part:
                    while block := part.read(_COPY_BUFFER):
                        digest.update(block)
                        out.write(block)
                        total += len(block)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    return digest.hexdigest(), total


def remove_upload_dir(paths: VolumePaths, upload_id: UUID) -> None:
    shutil.rmtree(paths.upload_dir(upload_id), ignore_errors=True)
