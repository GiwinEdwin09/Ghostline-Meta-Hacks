"""What a running job hands to its stages: the repo, paths, notebook, and phase reporting."""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import get_args

from ghostline.config import Settings
from ghostline.jobs.clients import make_runtime
from ghostline.jobs.errors import JobAbandoned, StageError
from ghostline.llm import MetaClient, ReviewerClient, Runtime
from ghostline.notebook import NotebookWriter
from ghostline.store.records import Job, JobKind, Phase
from ghostline.store.repo import Repo
from ghostline.store.volume import VolumePaths, VolumeSync

log = logging.getLogger("ghostline.jobs")

PHASES: tuple[str, ...] = get_args(Phase)

# Each job kind's phases with their share of the job's 0-1 progress bar. Weights follow where the
# time goes: SAM tracking and the physicist loop dominate an analysis.
PHASE_PLANS: dict[JobKind, list[tuple[str, float]]] = {
    "ingest_trial": [("normalizing", 1.0)],
    "analyze_experiment": [
        ("watching", 0.08),
        ("tracking", 0.27),
        ("calibrating", 0.10),
        ("hypothesizing", 0.15),
        ("fitting", 0.25),
        ("peer_review", 0.15),
    ],
    "prove_trial": [
        ("tracking", 0.35),
        ("calibrating", 0.15),
        ("predicting", 0.20),
        ("rendering", 0.30),
    ],
    "revise": [
        ("diagnosing", 0.25),
        ("fitting", 0.25),
        ("rendering", 0.35),
        ("peer_review", 0.15),
    ],
    "rerun": [("fitting", 1.0)],
    "export_verify": [("verifying", 1.0)],
}

_PROGRESS_WRITE_INTERVAL_S = 0.5


def phase_span(kind: JobKind, phase: str) -> tuple[float, float]:
    """(start, weight) of a phase within its job's progress bar. A phase missing from the plan
    (a revision round re-entering an earlier one, say) gets (0, 0): progress stays where it is."""
    start = 0.0
    for name, weight in PHASE_PLANS[kind]:
        if name == phase:
            return start, weight
        start += weight
    return 0.0, 0.0


@dataclass
class JobContext:
    """Passed to every handler and stage. Stages report with `phase()` and `progress()`, write the
    notebook through `notebook`, and may persist rows incrementally through `repo`."""

    job: Job
    repo: Repo
    paths: VolumePaths
    volume: VolumeSync
    settings: Settings
    abandoned: threading.Event = field(default_factory=threading.Event)
    current_phase: str | None = None
    _span: tuple[float, float] = (0.0, 0.0)
    _last_progress_write: float = float("-inf")
    notebook: NotebookWriter = field(init=False)
    _runtime: Runtime | None = field(default=None, init=False, repr=False)
    _meta: MetaClient | None = field(default=None, init=False, repr=False)
    _reviewer: ReviewerClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.notebook = NotebookWriter(
            self.repo, self.job.experiment_id, current_phase=lambda: self.current_phase
        )

    @property
    def runtime(self) -> Runtime:
        """The job's model runtime: mode and caps from settings, spend recorded in Postgres."""
        if self._runtime is None:
            self._runtime = make_runtime(self.settings, self.repo, self.paths)
        return self._runtime

    @property
    def meta(self) -> MetaClient:
        """Muse Wonder, SAM 3.1 and the Files API."""
        if self._meta is None:
            if self.runtime.mode != "replay" and not self.settings.model_api_key:
                raise StageError(
                    "The Meta Model API key isn't configured on the server.",
                    code="not_configured",
                    retryable=False,
                )
            self._meta = MetaClient(self.runtime)
        return self._meta

    @property
    def reviewer(self) -> ReviewerClient:
        """Muse Spark Open on the Modal Shared Endpoint."""
        if self._reviewer is None:
            self._reviewer = ReviewerClient(self.runtime)
        return self._reviewer

    @property
    def experiment_id(self):
        return self.job.experiment_id

    def check_alive(self) -> None:
        """Raise JobAbandoned if the job was failed from outside (heartbeat timeout)."""
        if self.abandoned.is_set():
            raise JobAbandoned(str(self.job.id))

    def phase(self, name: str) -> None:
        """Enter a phase: commit the Volume (the previous phase's files), then publish the new
        phase and its progress so pollers see it."""
        if name not in PHASES:
            raise ValueError(f"unknown phase {name!r}; expected one of {PHASES}")
        self.check_alive()
        self.commit()
        self.current_phase = name
        self._span = phase_span(self.job.kind, name)
        if not self.repo.set_job_phase(self.job.id, name, self._span[0]):
            self.abandoned.set()
            raise JobAbandoned(str(self.job.id))
        log.info(
            "phase",
            extra={
                "experiment_id": str(self.job.experiment_id),
                "job_id": str(self.job.id),
                "phase": name,
            },
        )

    def progress(self, fraction: float) -> None:
        """Progress within the current phase, 0 to 1. Writes are throttled."""
        self.check_alive()
        now = time.monotonic()
        if fraction < 1.0 and now - self._last_progress_write < _PROGRESS_WRITE_INTERVAL_S:
            return
        self._last_progress_write = now
        start, weight = self._span
        self.repo.set_job_progress(self.job.id, start + weight * min(max(fraction, 0.0), 1.0))

    def commit(self) -> None:
        """Make this job's Volume writes visible to other containers."""
        self.volume.commit()
