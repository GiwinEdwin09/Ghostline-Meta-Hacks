"""Smaller routes: job status and retry, the benchmark table, passcode check, admin overview
(build plan Parts 21 and 24)."""

from uuid import UUID

from fastapi import APIRouter, Depends

from ghostline.api.benchmark import BenchmarkTableView, benchmark_table
from ghostline.api.deps import RepoDep, ServicesDep, require_passcode, start_job
from ghostline.api.errors import ERROR_RESPONSES, ApiError
from ghostline.api.schemas import (
    AdminBudget,
    AdminCache,
    AdminExperiment,
    AdminJob,
    AdminOverview,
    AuthCheck,
)
from ghostline.jobs import lifecycle
from ghostline.store.records import BenchmarkEntry, BenchmarkRecord, Job

router = APIRouter(responses=ERROR_RESPONSES)


@router.get("/auth/check", response_model=AuthCheck, dependencies=[Depends(require_passcode)])
def check_passcode() -> AuthCheck:
    """200 if the X-Ghostline-Key header is right, 401 if not. Lets the app validate a passcode
    when the person types it."""
    return AuthCheck()


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: UUID, services: ServicesDep, repo: RepoDep) -> Job:
    job = repo.get_job(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", "That job doesn't exist.")
    lifecycle.reap_stale_jobs(repo, services.settings, job.experiment_id)
    return repo.get_job(job_id) or job


@router.post(
    "/jobs/{job_id}/retry",
    response_model=Job,
    status_code=202,
    dependencies=[Depends(require_passcode)],
)
def retry_job(job_id: UUID, services: ServicesDep, repo: RepoDep) -> Job:
    """The Retry button: run a failed job again as a new job with the same inputs. Stages cache
    their model and SAM calls, so a retry mostly re-reads."""
    lifecycle.reap_stale_jobs(repo, services.settings)
    job = repo.get_job(job_id)
    if job is None:
        raise ApiError(404, "job_not_found", "That job doesn't exist.")
    if job.status != "failed":
        raise ApiError(409, "not_failed", "Only a failed job can be retried.")
    if not job.retryable:
        raise ApiError(
            409, "not_retryable", "Retrying won't help. The clip itself needs to change."
        )
    return start_job(services, repo, job.experiment_id, job.kind, job.params, retry_of=job.id)


@router.get("/benchmark", response_model=BenchmarkTableView)
def get_benchmark(repo: RepoDep) -> BenchmarkTableView:
    return benchmark_table(repo.list_benchmark())


@router.put(
    "/benchmark/{experiment_id}",
    response_model=BenchmarkRecord,
    dependencies=[Depends(require_passcode)],
)
def put_benchmark(experiment_id: UUID, body: BenchmarkEntry, repo: RepoDep) -> BenchmarkRecord:
    """Set the ground truth for one quantity of an experiment, and mark it as a benchmark."""
    if repo.get_experiment(experiment_id) is None:
        raise ApiError(404, "experiment_not_found", "That experiment doesn't exist.")
    return repo.upsert_benchmark(experiment_id, body)


@router.get(
    "/admin/overview", response_model=AdminOverview, dependencies=[Depends(require_passcode)]
)
def admin_overview(services: ServicesDep, repo: RepoDep) -> AdminOverview:
    lifecycle.reap_stale_jobs(repo, services.settings)
    stats = repo.llm_stats_today()
    calls = stats.get("calls", 0)
    jobs = repo.list_jobs(limit=50)
    titles = repo.experiment_titles({j.experiment_id for j in jobs})
    return AdminOverview(
        budget=AdminBudget(
            daily_usd=services.settings.budget_usd_daily,
            spent_today_usd=repo.spend_today(),
            per_experiment_usd=services.settings.budget_usd_per_experiment,
        ),
        cache=AdminCache(
            hit_rate=stats.get("cache_hits", 0) / calls if calls else 0.0, calls_24h=calls
        ),
        jobs=[
            AdminJob(**j.model_dump(), experiment_title=titles.get(j.experiment_id, ""))
            for j in jobs
        ],
        experiments=[
            AdminExperiment(**r, llm_mode=services.settings.llm_mode)
            for r in repo.experiment_costs()
        ],
    )
