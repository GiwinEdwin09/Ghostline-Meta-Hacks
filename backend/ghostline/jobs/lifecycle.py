"""Job state transitions and what they do to the experiment and trial rows (build plan Part 21).

    queued -> running -> succeeded | failed
    queued -> failed   (never started: spawn failed or no worker picked it up)
    running -> failed  (worker raised, or no heartbeat for JOB_HEARTBEAT_TIMEOUT_S)

fail_job is the only way a job fails. It is safe to call from the worker and from the reaper at
the same time: whichever gets there first wins, the other does nothing.
"""

import logging
from uuid import UUID

from ghostline.config import Settings
from ghostline.notebook import NotebookWriter
from ghostline.store.records import Job
from ghostline.store.repo import Repo

log = logging.getLogger("ghostline.jobs")

JOB_LABELS = {
    "ingest_trial": "Preparing the clip",
    "analyze_experiment": "Analysis",
    "prove_trial": "Prove it",
    "revise": "Revision",
    "rerun": "Re-run",
    "export_verify": "Export check",
}


def trial_id_of(job: Job) -> UUID | None:
    raw = job.params.get("trial_id")
    return UUID(str(raw)) if raw else None


def job_started(repo: Repo, job: Job) -> None:
    if job.kind == "analyze_experiment":
        repo.update_experiment(job.experiment_id, status="analyzing", error=None)
    elif job.kind == "ingest_trial":
        trial_id = trial_id_of(job)
        trial = repo.get_trial(trial_id) if trial_id else None
        if trial is None:
            return
        repo.update_trial(trial.id, status="ingesting", error=None)
        experiment = repo.get_experiment(job.experiment_id)
        if trial.role == "fit" and experiment and experiment.status in ("draft", "ready", "failed"):
            repo.update_experiment(job.experiment_id, status="ingesting")


def job_succeeded(repo: Repo, job: Job) -> None:
    experiment = repo.get_experiment(job.experiment_id)
    if experiment is None:
        return
    if job.kind == "analyze_experiment":
        repo.update_experiment(job.experiment_id, status="complete", error=None)
    elif job.kind == "ingest_trial" and experiment.status == "ingesting":
        repo.update_experiment(job.experiment_id, status="ready")


def job_failed(repo: Repo, job: Job) -> None:
    message = job.error or "The job failed."
    experiment = repo.get_experiment(job.experiment_id)
    if experiment is None:
        return
    if job.kind == "analyze_experiment":
        repo.update_experiment(job.experiment_id, status="failed", error=message)
    elif job.kind == "ingest_trial":
        trial_id = trial_id_of(job)
        if trial_id:
            repo.update_trial(trial_id, status="failed", error=message)
        if experiment.status == "ingesting":
            repo.update_experiment(job.experiment_id, status="draft")
    NotebookWriter(repo, job.experiment_id).warning(
        f"{JOB_LABELS[job.kind]} failed",
        message,
        phase=job.phase,
        data={"job_id": str(job.id), "code": job.error_code, "retryable": job.retryable},
    )


def fail_job(
    repo: Repo,
    job: Job,
    *,
    error: str,
    code: str = "stage_error",
    retryable: bool = True,
) -> Job | None:
    """Mark an active job failed and update the rows around it. Returns the failed job, or None
    if it had already reached a final state."""
    finished = repo.finish_job(
        job.id, status="failed", error=error, error_code=code, retryable=retryable
    )
    if finished is None:
        return None
    job_failed(repo, finished)
    log.error(
        "job failed",
        extra={
            "experiment_id": str(job.experiment_id),
            "job_id": str(job.id),
            "phase": finished.phase,
            "code": code,
            "error": error,
        },
    )
    return finished


def reap_stale_jobs(repo: Repo, settings: Settings, experiment_id: UUID | None = None) -> list[Job]:
    """Fail jobs whose worker went quiet: running with no heartbeat for JOB_HEARTBEAT_TIMEOUT_S,
    or queued for longer than JOB_QUEUED_TIMEOUT_S. The API calls this before it reads or creates
    jobs, so a killed worker shows up as failed with a Retry button within about a minute."""
    failed = []
    stale = repo.stale_jobs(
        running_timeout_s=settings.job_heartbeat_timeout_s,
        queued_timeout_s=settings.job_queued_timeout_s,
        experiment_id=experiment_id,
    )
    for job in stale:
        if job.status == "running":
            message = (
                f"The worker stopped responding (no heartbeat for "
                f"{settings.job_heartbeat_timeout_s:.0f} s). Press Retry."
            )
            result = fail_job(repo, job, error=message, code="heartbeat_timeout")
        else:
            message = "The job never started. Press Retry."
            result = fail_job(repo, job, error=message, code="queue_timeout")
        if result:
            failed.append(result)
    return failed
