"""Run one job: claim it, heartbeat while it works, and end it succeeded or failed (Part 21)."""

import logging
import threading
from collections.abc import Callable, Mapping
from uuid import UUID

from ghostline.config import Settings
from ghostline.jobs import lifecycle
from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import JobAbandoned, StageError
from ghostline.jobs.handlers import HANDLERS, Handler
from ghostline.store.records import Job, JobKind
from ghostline.store.repo import Repo
from ghostline.store.volume import VolumePaths, VolumeSync

log = logging.getLogger("ghostline.jobs")


class Heartbeat:
    """Touches jobs.heartbeat_at every `interval_s` from a background thread, so a long blocking
    stage (a SAM call, a fit) still looks alive. If the job turns out to have been failed
    meanwhile, it calls on_lost and stops."""

    def __init__(self, repo: Repo, job_id: UUID, interval_s: float, on_lost: Callable[[], None]):
        self._repo = repo
        self._job_id = job_id
        self._interval = interval_s
        self._on_lost = on_lost
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval + 5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                alive = self._repo.heartbeat(self._job_id)
            except Exception:
                log.exception("heartbeat failed", extra={"job_id": str(self._job_id)})
                continue
            if not alive:
                self._on_lost()
                return


def run_job(
    job_id: UUID,
    *,
    repo: Repo,
    paths: VolumePaths,
    volume: VolumeSync,
    settings: Settings,
    handlers: Mapping[JobKind, Handler] | None = None,
) -> Job | None:
    """Execute a queued job to completion and return its final row. Never raises for a job
    failure: the failure is recorded on the job. Returns the current row without running anything
    if the job was already claimed or finished (a duplicate delivery)."""
    job = repo.claim_job(job_id)
    if job is None:
        log.warning("job not claimable", extra={"job_id": str(job_id)})
        return repo.get_job(job_id)

    extra = {"experiment_id": str(job.experiment_id), "job_id": str(job.id), "kind": job.kind}
    log.info("job started", extra=extra)
    ctx = JobContext(job=job, repo=repo, paths=paths, volume=volume, settings=settings)
    heartbeat = Heartbeat(repo, job.id, settings.job_heartbeat_interval_s, ctx.abandoned.set)
    heartbeat.start()
    try:
        lifecycle.job_started(repo, job)
        (handlers or HANDLERS)[job.kind](ctx)
        ctx.commit()
    except JobAbandoned:
        log.warning("job abandoned", extra=extra)
        return repo.get_job(job.id)
    except StageError as exc:
        lifecycle.fail_job(repo, job, error=exc.message, code=exc.code, retryable=exc.retryable)
    except NotImplementedError as exc:
        lifecycle.fail_job(repo, job, error=str(exc) or "Not implemented.", code="not_implemented")
    except Exception as exc:
        log.exception("job crashed", extra=extra)
        lifecycle.fail_job(
            repo, job, error=f"{type(exc).__name__}: {exc}", code="internal_error", retryable=True
        )
    except BaseException:
        # Modal cancelling or timing out the function: record it, then let it propagate.
        lifecycle.fail_job(repo, job, error="The worker was stopped.", code="interrupted")
        raise
    else:
        finished = repo.finish_job(job.id, status="succeeded")
        if finished is not None:
            lifecycle.job_succeeded(repo, finished)
            log.info("job succeeded", extra=extra)
    finally:
        heartbeat.stop()
    return repo.get_job(job.id)
