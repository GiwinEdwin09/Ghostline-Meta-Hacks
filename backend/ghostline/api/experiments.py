"""Experiment routes: create, read, poll, analyze, prove, revise, simulate, export, delete
(build plan Part 21)."""

import concurrent.futures
import shutil
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from ghostline.api.deps import (
    AdminDep,
    ExperimentDep,
    RepoDep,
    ServicesDep,
    client_ip,
    hash_ip,
    redact,
    require_passcode,
    start_job,
)
from ghostline.api.errors import ERROR_RESPONSES, ApiError
from ghostline.api.schemas import (
    AnalyzeRequest,
    DeleteResult,
    EventsPage,
    ExperimentCreate,
    ExperimentPatch,
    ExperimentState,
    ProveRequest,
    RerunRequest,
    ReviseRequest,
)
from ghostline.jobs import lifecycle, stages
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import SimulateRequest, SimulateResult
from ghostline.store.core_schemas import SimExport
from ghostline.store.records import Experiment, Job

router = APIRouter(responses=ERROR_RESPONSES)

_simulate_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="simulate")


@router.post(
    "/experiments",
    response_model=Experiment,
    status_code=201,
    dependencies=[Depends(require_passcode)],
)
def create_experiment(body: ExperimentCreate, repo: RepoDep) -> Experiment:
    experiment = repo.create_experiment(
        title=body.title,
        description=body.description,
        system_type=body.system_hint,
        declared_capture=body.declared_capture,
        setup=body.setup,
    )
    return redact(experiment, admin=True)


@router.get("/experiments", response_model=list[Experiment])
def list_experiments(
    request: Request,
    services: ServicesDep,
    repo: RepoDep,
    admin: AdminDep,
    examples: Annotated[bool, Query(description="Only the public example experiments")] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Experiment]:
    """The gallery (`examples=true`) is public. Listing everything needs the passcode."""
    if not examples:
        require_passcode(request, services)
    rows = repo.list_experiments(examples_only=examples, limit=limit, offset=offset)
    return [redact(e, admin) for e in rows]


@router.get("/experiments/{experiment_id}", response_model=ExperimentState)
def get_experiment(
    experiment: ExperimentDep, services: ServicesDep, repo: RepoDep, admin: AdminDep
) -> ExperimentState:
    lifecycle.reap_stale_jobs(repo, services.settings, experiment.id)
    experiment = repo.get_experiment(experiment.id) or experiment
    trials = repo.list_trials(experiment.id)
    trial_ids = [t.id for t in trials]
    jobs = repo.list_jobs(experiment.id, limit=20)
    return ExperimentState(
        **redact(experiment, admin).model_dump(),
        trials=trials,
        tracks=repo.list_tracks(trial_ids),
        trajectories=repo.list_trajectories(trial_ids),
        models=repo.list_models(experiment.id),
        fits=repo.list_fits(experiment.id),
        reviews=repo.list_reviews(experiment.id),
        ghosts=repo.list_ghosts(experiment.id),
        jobs=jobs,
        active_job=next((j for j in jobs if j.status in ("queued", "running")), None),
    )


@router.patch(
    "/experiments/{experiment_id}",
    response_model=Experiment,
    dependencies=[Depends(require_passcode)],
)
def patch_experiment(experiment: ExperimentDep, body: ExperimentPatch, repo: RepoDep) -> Experiment:
    """Rename, or publish as an example (an example is always public)."""
    fields = body.model_dump(exclude_none=True)
    if fields.get("is_example"):
        fields["visibility"] = "public"
    if not fields:
        return redact(experiment, admin=True)
    updated = repo.update_experiment(experiment.id, **fields)
    return redact(updated or experiment, admin=True)


@router.get("/experiments/{experiment_id}/events", response_model=EventsPage)
def list_events(
    experiment: ExperimentDep,
    services: ServicesDep,
    repo: RepoDep,
    after: Annotated[int, Query(ge=0, description="Return events with a higher seq")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> EventsPage:
    """Poll this every second while a job runs. One response carries the new events, the
    experiment status and the job's phase and progress."""
    lifecycle.reap_stale_jobs(repo, services.settings, experiment.id)
    events = repo.list_events(experiment.id, after=after, limit=limit)
    fresh = repo.get_experiment(experiment.id) or experiment
    jobs = repo.list_jobs(experiment.id, limit=1)
    return EventsPage(
        events=events,
        last_seq=events[-1].seq if events else after,
        status=fresh.status,
        job=jobs[0] if jobs else None,
    )


# --- jobs ----------------------------------------------------------------------------------
@router.post(
    "/experiments/{experiment_id}/analyze",
    response_model=Job,
    status_code=202,
    dependencies=[Depends(require_passcode)],
)
def analyze(
    experiment: ExperimentDep,
    request: Request,
    services: ServicesDep,
    repo: RepoDep,
    body: AnalyzeRequest | None = None,
) -> Job:
    """Start the analysis of the newest ready clip. Asking again while it runs returns the same
    job."""
    trial_id = body.trial_id if body else None
    trial = (
        repo.get_trial(trial_id)
        if trial_id
        else repo.latest_trial(experiment.id, "fit", status="ready")
    )
    if trial is None or trial.experiment_id != experiment.id or trial.status != "ready":
        raise ApiError(409, "no_ready_trial", "Upload a clip and wait for it to be prepared first.")
    if trial.role != "fit":
        raise ApiError(409, "wrong_trial_role", "Analysis uses the first clip, not a proof clip.")
    return start_job(
        services,
        repo,
        experiment.id,
        "analyze_experiment",
        {"trial_id": str(trial.id)},
        client_ip_hash=hash_ip(client_ip(request)),
    )


@router.post(
    "/experiments/{experiment_id}/prove",
    response_model=Job,
    status_code=202,
    dependencies=[Depends(require_passcode)],
)
def prove(
    experiment: ExperimentDep,
    services: ServicesDep,
    repo: RepoDep,
    body: ProveRequest | None = None,
) -> Job:
    """Run the ghost prediction on the second clip."""
    if experiment.status != "complete" or experiment.conclusion is None:
        raise ApiError(409, "not_analyzed", "Analyze the first clip before proving the law.")
    trial_id = body.trial_id if body else None
    trial = (
        repo.get_trial(trial_id)
        if trial_id
        else repo.latest_trial(experiment.id, "prove", status="ready")
    )
    if trial is None or trial.experiment_id != experiment.id or trial.status != "ready":
        raise ApiError(409, "no_prove_trial", "Upload the second clip and wait for it first.")
    if trial.role != "prove":
        raise ApiError(409, "wrong_trial_role", "Choose a clip uploaded for proving.")
    return start_job(services, repo, experiment.id, "prove_trial", {"trial_id": str(trial.id)})


@router.post(
    "/experiments/{experiment_id}/revise",
    response_model=Job,
    status_code=202,
    dependencies=[Depends(require_passcode)],
)
def revise(
    experiment: ExperimentDep,
    services: ServicesDep,
    repo: RepoDep,
    body: ReviseRequest | None = None,
) -> Job:
    """Diagnose where the ghost drifted and try a revised model."""
    ghosts = repo.list_ghosts(experiment.id)
    if not ghosts:
        raise ApiError(409, "no_ghost", "Run Prove it first; there is no ghost to diagnose.")
    ghost_id = body.ghost_id if body else None
    if ghost_id and all(g.id != ghost_id for g in ghosts):
        raise ApiError(404, "ghost_not_found", "That ghost doesn't belong to this experiment.")
    params = {"ghost_id": str(ghost_id or ghosts[-1].id)}
    return start_job(services, repo, experiment.id, "revise", params)


@router.post(
    "/experiments/{experiment_id}/models/{model_id}/rerun",
    response_model=Job,
    status_code=202,
    dependencies=[Depends(require_passcode)],
)
def rerun_model(
    experiment: ExperimentDep,
    model_id: str,
    body: RerunRequest,
    services: ServicesDep,
    repo: RepoDep,
) -> Job:
    """Re-run one fit with an edited split and initial guesses (notebook Re-run button)."""
    model = repo.get_model(experiment.id, model_id)
    if model is None:
        raise ApiError(404, "model_not_found", "That model doesn't belong to this experiment.")
    if model.params_spec is not None:
        unknown = sorted(set(body.initial_guesses) - set(model.params_spec))
        if unknown:
            raise ApiError(
                400,
                "unknown_parameter",
                f"{model.name} has no parameter named {', '.join(unknown)}.",
            )
    params = {
        "model_id": model.id,
        "split_pct": body.split_pct,
        "initial_guesses": body.initial_guesses,
    }
    return start_job(services, repo, experiment.id, "rerun", params)


@router.post(
    "/experiments/{experiment_id}/export/verify",
    response_model=Job,
    status_code=202,
    dependencies=[Depends(require_passcode)],
)
def verify_export(experiment: ExperimentDep, services: ServicesDep, repo: RepoDep) -> Job:
    """Spawn the PyBullet replay check (Part 19)."""
    if experiment.status != "complete" or experiment.conclusion is None:
        raise ApiError(409, "not_analyzed", "There is no result to export yet.")
    return start_job(services, repo, experiment.id, "export_verify")


# --- synchronous: simulate and export ------------------------------------------------------
@router.post(
    "/experiments/{experiment_id}/simulate",
    response_model=SimulateResult,
    dependencies=[Depends(require_passcode)],
)
def simulate(
    experiment: ExperimentDep, body: SimulateRequest, services: ServicesDep, repo: RepoDep
) -> SimulateResult:
    """Simulate a model, including a Wonder-written one, in at most SIMULATE_TIMEOUT_S."""
    model = repo.get_model(experiment.id, body.model_id)
    if model is None:
        raise ApiError(404, "model_not_found", "That model doesn't belong to this experiment.")
    future = _simulate_pool.submit(stages.simulate_custom_model, repo, experiment, model, body)
    try:
        return future.result(timeout=services.settings.simulate_timeout_s)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise ApiError(
            504,
            "simulate_timeout",
            f"The simulation took longer than {services.settings.simulate_timeout_s:.0f} s.",
        ) from None
    except StageError as exc:
        raise ApiError(422, exc.code, exc.message) from exc


@router.get(
    "/experiments/{experiment_id}/export",
    response_model=SimExport,
    responses={200: {"content": {"text/x-python": {"schema": {"type": "string"}}}}},
)
def export(
    experiment: ExperimentDep,
    repo: RepoDep,
    format: Annotated[Literal["json", "pybullet"], Query()] = "json",
    mass_g: Annotated[
        float | None, Query(gt=0, le=1_000_000, description="Object mass, grams")
    ] = None,
) -> Response:
    """Simulator parameters as Ghostline JSON (`format=json`) or a PyBullet snippet."""
    if experiment.status != "complete" or experiment.conclusion is None:
        raise ApiError(409, "not_analyzed", "There is no result to export yet.")
    result = stages.build_export(repo, experiment, format, mass_g)
    extension = "json" if format == "json" else "py"
    headers = {
        "Content-Disposition": f'attachment; filename="ghostline-{experiment.id}.{extension}"'
    }
    if isinstance(result, str):
        return PlainTextResponse(result, media_type="text/x-python", headers=headers)
    return JSONResponse(result.model_dump(mode="json"), headers=headers)


# --- delete --------------------------------------------------------------------------------
@router.delete(
    "/experiments/{experiment_id}",
    response_model=DeleteResult,
    dependencies=[Depends(require_passcode)],
)
def delete_experiment(
    experiment: ExperimentDep, services: ServicesDep, repo: RepoDep
) -> DeleteResult:
    """Remove the experiment's rows, its Volume folder and its Meta Files API uploads. Examples
    and benchmark experiments are protected: unpublish them first."""
    if experiment.is_example or experiment.is_benchmark:
        raise ApiError(
            409,
            "protected",
            "This is an example or benchmark experiment. Unpublish it before deleting.",
        )
    lifecycle.reap_stale_jobs(repo, services.settings, experiment.id)
    if repo.active_job(experiment.id):
        raise ApiError(409, "job_active", "A job is still running for this experiment.")

    file_ids = [fid for t in repo.list_trials(experiment.id) for fid in t.meta_file_ids.values()]
    upload_ids = repo.upload_ids_for_experiment(experiment.id)
    deleted = sum(1 for fid in file_ids if services.file_deleter.delete(fid))

    for upload_id in upload_ids:
        shutil.rmtree(services.paths.upload_dir(upload_id), ignore_errors=True)
    shutil.rmtree(services.paths.experiment_dir(experiment.id), ignore_errors=True)
    if services.paths.experiment_dir(experiment.id).exists():
        raise ApiError(503, "volume_busy", "Couldn't remove the files yet. Try again.")
    services.volume.commit()
    repo.delete_experiment(experiment.id)
    return DeleteResult(meta_files_deleted=deleted, meta_files_failed=len(file_ids) - deleted)
