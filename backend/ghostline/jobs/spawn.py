"""Starting a job on a worker (build plan Part 21: Modal Functions started with .spawn())."""

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from uuid import UUID

from ghostline.jobs import lifecycle
from ghostline.store.records import Job, JobKind
from ghostline.store.repo import Repo

log = logging.getLogger("ghostline.jobs")

# Which Modal function runs which kind of job. The function names are the three in Part 21's
# sizing table; revise renders video so it shares prove_trial's resources, and the lighter rerun
# and export check share analyze_experiment's.
FUNCTION_FOR_KIND: dict[JobKind, str] = {
    "ingest_trial": "ingest_trial",
    "analyze_experiment": "analyze_experiment",
    "rerun": "analyze_experiment",
    "export_verify": "analyze_experiment",
    "prove_trial": "prove_trial",
    "revise": "prove_trial",
}


class JobSpawner(Protocol):
    def spawn(self, job: Job) -> str | None:
        """Start the job on a worker and return an id for the call, if there is one."""
        ...


class ModalSpawner:
    """Spawns the Modal Functions, given as {function name: modal Function} by modal_app.py."""

    def __init__(self, functions: Mapping[str, Any]):
        self._functions = functions

    def spawn(self, job: Job) -> str | None:
        call = self._functions[FUNCTION_FOR_KIND[job.kind]].spawn(str(job.id))
        return getattr(call, "object_id", None)


class ThreadSpawner:
    """Runs jobs in a background thread of this process. For local development without Modal."""

    def __init__(self, run: Callable[[UUID], Any]):
        self._run = run

    def spawn(self, job: Job) -> str | None:
        thread = threading.Thread(
            target=self._run, args=(job.id,), name=f"job-{job.id}", daemon=True
        )
        thread.start()
        return None


class InlineSpawner:
    """Runs the job to completion before returning. For tests."""

    def __init__(self, run: Callable[[UUID], Any]):
        self._run = run

    def spawn(self, job: Job) -> str | None:
        self._run(job.id)
        return None


def dispatch(repo: Repo, spawner: JobSpawner, job: Job) -> Job:
    """Spawn a job that was just created. If the spawn itself fails, fail the job so the user
    sees a Retry button instead of a job stuck in 'queued'. Returns the job's current row."""
    try:
        call_id = spawner.spawn(job)
    except Exception as exc:
        log.exception("spawn failed", extra={"job_id": str(job.id), "kind": job.kind})
        lifecycle.fail_job(
            repo, job, error=f"Could not start the job: {exc}", code="spawn_failed", retryable=True
        )
    else:
        if call_id:
            repo.set_job_call_id(job.id, call_id)
    return repo.get_job(job.id) or job
