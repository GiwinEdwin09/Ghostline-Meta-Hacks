"""Shared FastAPI dependencies: services, the passcode check, client identity (build plan Parts
21 and 28)."""

import hashlib
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request

from ghostline.api.errors import ApiError
from ghostline.config import Settings
from ghostline.jobs import lifecycle
from ghostline.jobs.spawn import JobSpawner, dispatch
from ghostline.store.meta_files import FileDeleter
from ghostline.store.records import Experiment, Job, JobKind
from ghostline.store.repo import ActiveJobError, Repo
from ghostline.store.volume import VolumePaths, VolumeSync

PASSCODE_HEADER = "X-Ghostline-Key"


class FailureLimiter:
    """Counts failed passcode attempts per client in a sliding window, so the passcode can't be
    guessed at speed. In memory, so per container: good enough against casual guessing."""

    def __init__(self, max_failures: int, window_s: float = 600.0, clock=time.monotonic):
        self._max = max_failures
        self._window = window_s
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str) -> deque[float]:
        events = self._failures[key]
        cutoff = self._clock() - self._window
        while events and events[0] < cutoff:
            events.popleft()
        return events

    def blocked(self, key: str) -> bool:
        return len(self._prune(key)) >= self._max

    def record(self, key: str) -> None:
        self._prune(key).append(self._clock())


@dataclass
class Services:
    """Everything the routes need, built once in create_app and kept on app.state."""

    settings: Settings
    paths: VolumePaths
    volume: VolumeSync
    spawner: JobSpawner
    file_deleter: FileDeleter
    repo: Repo | None
    meta_ping: Callable[[], bool] | None
    passcode_failures: FailureLimiter
    _last_reload: float = field(default=float("-inf"))
    meta_health: tuple[float, str] | None = None

    def reload_volume(self) -> bool:
        """Reload the Volume to see other containers' commits, at most once a second. A busy
        volume must not fail a request, so errors are swallowed."""
        now = time.monotonic()
        if now - self._last_reload < 1.0:
            return False
        self._last_reload = now
        try:
            self.volume.reload()
        except Exception:
            return False
        return True


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_repo(services: ServicesDep) -> Repo:
    if services.repo is None:
        raise ApiError(503, "database_not_configured", "The database is not configured.")
    return services.repo


RepoDep = Annotated[Repo, Depends(get_repo)]


def client_ip(request: Request) -> str:
    """The caller's address. Behind Modal's proxy the last X-Forwarded-For entry is the one the
    proxy added; earlier entries come from the client and can be forged."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def hash_ip(ip: str) -> str:
    return hashlib.sha256(f"ghostline:{ip}".encode()).hexdigest()[:32]


def _passcode_ok(request: Request, services: Services) -> bool:
    expected = services.settings.app_passcode
    given = request.headers.get(PASSCODE_HEADER, "")
    # Compare bytes so a non-ASCII header can't raise.
    return bool(expected) and secrets.compare_digest(given.encode(), expected.encode())


def require_passcode(request: Request, services: ServicesDep) -> None:
    """Write routes need the X-Ghostline-Key header. With no APP_PASSCODE configured they are
    refused outright rather than left open."""
    if not services.settings.app_passcode:
        raise ApiError(503, "passcode_not_configured", "Write access is not configured.")
    key = hash_ip(client_ip(request))
    if services.passcode_failures.blocked(key):
        raise ApiError(429, "too_many_attempts", "Too many wrong passcodes. Try again later.")
    if not request.headers.get(PASSCODE_HEADER):
        raise ApiError(401, "passcode_required", "Enter the passcode to do that.")
    if not _passcode_ok(request, services):
        services.passcode_failures.record(key)
        raise ApiError(401, "passcode_invalid", "That passcode isn't right.")


PasscodeDep = Depends(require_passcode)


def is_admin(request: Request, services: ServicesDep) -> bool:
    """True when the request carries a valid passcode. Used to reveal costs; never raises."""
    return _passcode_ok(request, services)


AdminDep = Annotated[bool, Depends(is_admin)]


def load_experiment(experiment_id: UUID, repo: RepoDep) -> Experiment:
    experiment = repo.get_experiment(experiment_id)
    if experiment is None:
        raise ApiError(404, "experiment_not_found", "That experiment doesn't exist.")
    return experiment


ExperimentDep = Annotated[Experiment, Depends(load_experiment)]


def redact(experiment: Experiment, admin: bool) -> Experiment:
    """Costs are only for requests that carry the passcode."""
    if admin:
        return experiment
    return experiment.model_copy(update={"cost_usd": None})


def check_budget(services: Services, repo: Repo) -> None:
    """Refuse a new analysis once today's spend has reached BUDGET_USD_DAILY (Part 23)."""
    if repo.spend_today() >= services.settings.budget_usd_daily:
        raise ApiError(
            429,
            "daily_budget_reached",
            "Ghostline has used up today's model budget, so new analyses are paused until "
            "tomorrow. The example experiments still work.",
        )


def start_job(
    services: Services,
    repo: Repo,
    experiment_id: UUID,
    kind: JobKind,
    params: dict | None = None,
    *,
    client_ip_hash: str | None = None,
    retry_of: UUID | None = None,
) -> Job:
    """Create a job and spawn it. At most one job is active per experiment: asking again for the
    same job returns the running one (a double tap is harmless), asking for another is a 409."""
    params = params or {}
    lifecycle.reap_stale_jobs(repo, services.settings, experiment_id)
    active = repo.active_job(experiment_id)
    if active is None:
        if kind == "analyze_experiment":
            check_budget(services, repo)
            if client_ip_hash and retry_of is None:
                used = repo.count_recent_analyses(client_ip_hash)
                if used >= services.settings.analyses_per_ip_per_day:
                    raise ApiError(
                        429,
                        "rate_limited",
                        f"The limit is {services.settings.analyses_per_ip_per_day} analyses per "
                        "day from one address. Try again tomorrow.",
                    )
        try:
            job = repo.create_job(
                experiment_id=experiment_id,
                kind=kind,
                params=params,
                client_ip_hash=client_ip_hash,
                retry_of=retry_of,
            )
        except ActiveJobError as exc:
            active = exc.job
        else:
            return dispatch(repo, services.spawner, job)
    if active.kind == kind and active.params == params:
        return active
    raise ApiError(
        409,
        "job_active",
        "Another job is still running for this experiment. Wait for it to finish.",
        job_id=str(active.id),
    )
