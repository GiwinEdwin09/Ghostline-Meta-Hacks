"""Chunked, resumable uploads and trial trimming (build plan Parts 4 and 21).

The client creates an upload, PUTs 8 MiB chunks (any order, any number in parallel, each safe to
repeat), then calls complete. Complete assembles the file, hashes it, creates the trial and
spawns ingest_trial. Chunks live on the Volume, and any API container may receive any request, so
every chunk is committed when it is written and complete reloads before it assembles.
"""

import logging
import shutil
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from ghostline.api.deps import (
    RepoDep,
    ServicesDep,
    require_passcode,
)
from ghostline.api.errors import ERROR_RESPONSES, ApiError
from ghostline.api.schemas import (
    ChunkReceipt,
    TrimRequest,
    UploadComplete,
    UploadCreate,
    UploadInfo,
)
from ghostline.jobs import lifecycle
from ghostline.jobs.spawn import dispatch
from ghostline.store import uploads as upload_files
from ghostline.store.records import Experiment, Trial, Trim, Upload
from ghostline.store.repo import ActiveJobError

log = logging.getLogger("ghostline.api")
router = APIRouter(responses=ERROR_RESPONSES)


def _load_upload(repo: RepoDep, upload_id: UUID) -> Upload:
    upload = repo.get_upload(upload_id)
    if upload is None:
        raise ApiError(404, "upload_not_found", "That upload doesn't exist.")
    return upload


@router.post(
    "/uploads", response_model=UploadInfo, status_code=201, dependencies=[Depends(require_passcode)]
)
def create_upload(body: UploadCreate, services: ServicesDep, repo: RepoDep) -> UploadInfo:
    settings = services.settings
    if body.size > settings.max_upload_bytes:
        limit_gib = settings.max_upload_bytes / 1024**3
        raise ApiError(
            413, "file_too_large", f"Clips can be at most {limit_gib:.0f} GiB. Trim it and retry."
        )
    experiment: Experiment | None = repo.get_experiment(body.experiment_id)
    if experiment is None:
        raise ApiError(404, "experiment_not_found", "That experiment doesn't exist.")
    if body.role == "fit" and experiment.status in ("analyzing", "complete"):
        raise ApiError(
            409, "experiment_locked", "This experiment already has a clip. Start a new experiment."
        )
    if body.role == "prove" and experiment.status != "complete":
        raise ApiError(409, "not_analyzed", "Analyze the first clip before uploading a proof clip.")
    if repo.count_uploads(experiment.id) >= settings.max_open_uploads_per_experiment:
        raise ApiError(429, "too_many_uploads", "Too many uploads for this experiment.")
    upload = repo.create_upload(
        experiment_id=experiment.id,
        role=body.role,
        filename=body.filename,
        size_bytes=body.size,
        chunk_size=settings.upload_chunk_bytes,
        n_chunks=upload_files.n_chunks_for(body.size, settings.upload_chunk_bytes),
    )
    return UploadInfo.from_record(upload)


@router.get("/uploads/{upload_id}", response_model=UploadInfo)
def get_upload(upload_id: UUID, repo: RepoDep) -> UploadInfo:
    """Which chunks the server has, for resuming after a reload."""
    return UploadInfo.from_record(_load_upload(repo, upload_id))


@router.put(
    "/uploads/{upload_id}/chunks/{n}",
    response_model=ChunkReceipt,
    dependencies=[Depends(require_passcode)],
)
async def put_chunk(
    upload_id: UUID, n: int, request: Request, services: ServicesDep, repo: RepoDep
) -> ChunkReceipt:
    """Store one raw chunk. Repeating a chunk replaces it, so retries are safe."""
    upload = await run_in_threadpool(_load_upload, repo, upload_id)
    if upload.status != "open":
        raise ApiError(409, "upload_complete", "This upload is already complete.")
    if not 0 <= n < upload.n_chunks:
        raise ApiError(
            400, "bad_chunk_number", f"Chunk numbers run from 0 to {upload.n_chunks - 1}."
        )
    expected = upload_files.expected_chunk_size(upload, n)

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > expected:
        raise ApiError(413, "chunk_too_large", f"Chunk {n} must be {expected} bytes.")
    buffer = bytearray()
    async for part in request.stream():
        buffer += part
        if len(buffer) > expected:
            raise ApiError(413, "chunk_too_large", f"Chunk {n} must be {expected} bytes.")
    if len(buffer) != expected:
        raise ApiError(
            400, "bad_chunk_size", f"Chunk {n} must be {expected} bytes, got {len(buffer)}."
        )

    await run_in_threadpool(upload_files.write_chunk, services.paths, upload.id, n, bytes(buffer))
    await run_in_threadpool(services.volume.commit)
    updated = await run_in_threadpool(repo.add_received_chunk, upload.id, n)
    if updated is None:
        raise ApiError(409, "upload_complete", "This upload is already complete.")
    return ChunkReceipt(
        upload_id=upload.id,
        n=n,
        size=expected,
        received_count=len(updated.received_chunks),
        n_chunks=upload.n_chunks,
    )


@router.post(
    "/uploads/{upload_id}/complete",
    response_model=UploadComplete,
    dependencies=[Depends(require_passcode)],
)
def complete_upload(upload_id: UUID, services: ServicesDep, repo: RepoDep) -> UploadComplete:
    """Assemble the chunks into original.<ext>, hash it, create the trial and start ingest.
    Calling it again after success returns the same trial."""
    upload = _load_upload(repo, upload_id)
    if upload.status == "complete" and upload.trial_id:
        return _existing_result(repo, upload)

    lifecycle.reap_stale_jobs(repo, services.settings, upload.experiment_id)
    active = repo.active_job(upload.experiment_id)
    if active:
        raise _job_active(active.id)

    services.reload_volume()
    trial_id = uuid4()
    ext = upload_files.extension_for(upload.filename)
    destination = services.paths.original(upload.experiment_id, trial_id, ext)
    try:
        sha256, size = upload_files.assemble(services.paths, upload, destination)
    except upload_files.UploadIncomplete as exc:
        raise ApiError(
            409,
            "chunks_missing",
            f"{len(exc.missing)} chunk(s) haven't arrived yet. Send them and try again.",
            missing=exc.missing,
        ) from exc
    trial_dir = services.paths.trial_dir(upload.experiment_id, trial_id)
    try:
        services.volume.commit()
        with repo.transaction() as tx:
            locked = tx.lock_upload(upload.id)
            if locked is None:
                raise ApiError(404, "upload_not_found", "That upload doesn't exist.")
            if locked.status == "complete" and locked.trial_id:
                # A concurrent complete won the race; its trial is the answer.
                shutil.rmtree(trial_dir, ignore_errors=True)
                return _existing_result(tx, locked)
            trial = tx.create_trial(
                trial_id=trial_id,
                experiment_id=upload.experiment_id,
                role=upload.role,
                original_filename=upload.filename,
                size_bytes=size,
                sha256=sha256,
                paths={"original": services.paths.rel(upload.experiment_id, destination)},
            )
            job = tx.create_job(
                experiment_id=upload.experiment_id,
                kind="ingest_trial",
                params={"trial_id": str(trial.id)},
            )
            tx.mark_upload_complete(upload.id, trial.id)
    except ActiveJobError as exc:
        shutil.rmtree(trial_dir, ignore_errors=True)
        raise _job_active(exc.job.id) from exc
    except Exception:
        shutil.rmtree(trial_dir, ignore_errors=True)
        raise

    upload_files.remove_upload_dir(services.paths, upload.id)
    try:
        services.volume.commit()
    except Exception:
        log.warning("volume commit after upload cleanup failed", exc_info=True)
    job = dispatch(repo, services.spawner, job)
    return UploadComplete(trial_id=trial.id, job_id=job.id, sha256=sha256, size=size, job=job)


def _job_active(job_id: UUID) -> ApiError:
    return ApiError(
        409,
        "job_active",
        "Another job is still running for this experiment. Wait for it to finish.",
        job_id=str(job_id),
    )


def _existing_result(repo, upload: Upload) -> UploadComplete:
    trial = repo.get_trial(upload.trial_id)
    job = repo.latest_job_for_trial(upload.trial_id)
    if trial is None or job is None:
        raise ApiError(409, "upload_complete", "This upload is already complete.")
    return UploadComplete(
        trial_id=trial.id,
        job_id=job.id,
        sha256=trial.sha256 or "",
        size=trial.size_bytes or 0,
        job=job,
    )


@router.post(
    "/trials/{trial_id}/trim", response_model=Trial, dependencies=[Depends(require_passcode)]
)
def set_trim(
    trial_id: UUID, services: ServicesDep, repo: RepoDep, body: TrimRequest | None = None
) -> Trial:
    """Set the analysis window. Anything left out keeps the auto-suggested value."""
    trial = repo.get_trial(trial_id)
    if trial is None:
        raise ApiError(404, "trial_not_found", "That clip doesn't exist.")
    if trial.status != "ready" or trial.trim is None:
        raise ApiError(409, "trial_not_ready", "The clip is still being prepared.")
    body = body or TrimRequest()
    current = trial.trim
    suggested_start = current.suggested_start_s
    suggested_end = current.suggested_end_s
    default_start = current.start_s if suggested_start is None else suggested_start
    default_end = current.end_s if suggested_end is None else suggested_end
    start = default_start if body.start_s is None else body.start_s
    end = default_end if body.end_s is None else body.end_s
    if not start < end:
        raise ApiError(400, "bad_trim", "The trim must end after it starts.")
    duration = trial.duration_s
    if duration is not None and end > duration + 1e-6:
        raise ApiError(400, "bad_trim", f"The clip is only {duration:.2f} s long.")
    if duration and trial.n_frames:
        frames = (end - start) * trial.n_frames / duration
        if frames > services.settings.max_frames:
            raise ApiError(
                400,
                "too_many_frames",
                f"That window holds about {frames:.0f} frames; the limit is "
                f"{services.settings.max_frames}. Trim it shorter.",
            )
    trim = Trim(
        start_s=start,
        end_s=end,
        suggested_start_s=suggested_start,
        suggested_end_s=suggested_end,
        source="user",
    )
    updated = repo.update_trial(trial.id, trim=trim)
    return updated or trial
