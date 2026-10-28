"""Shared fixtures.

Postgres: tests that need a database use the `db` / `repo` fixtures, which start one throwaway
server in a temp directory on a free port (never the default port, never an existing server) and
give every test its own database cloned from a migrated template. They skip when the Postgres
binaries (`initdb`, `pg_ctl`) aren't installed; everything else still runs.
"""

import dataclasses
import glob
import os
import shutil
import socket
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ghostline.config import Settings, get_settings
from ghostline.jobs import stages
from ghostline.jobs.runner import run_job
from ghostline.jobs.spawn import InlineSpawner
from ghostline.jobs.stage_types import (
    CalibrationResult,
    GhostResult,
    IngestResult,
    ReviewOutcome,
    RevisionResult,
    SimulateResult,
    TrackingResult,
    TrajectoryResult,
)
from ghostline.store.core_schemas import Conclusion, Quantity, SceneAnalysis, TrackedObject
from ghostline.store.db import Database
from ghostline.store.migrate import apply_migrations
from ghostline.store.records import (
    FitParam,
    FitRecord,
    ModelRecord,
    ReviewRecord,
    TrackRecord,
    Trim,
)
from ghostline.store.repo import Repo
from ghostline.store.volume import NullVolume, VolumePaths

PASSCODE = "test-passcode"
AUTH = {"X-Ghostline-Key": PASSCODE}


@pytest.fixture
def auth() -> dict[str, str]:
    return AUTH


@pytest.fixture
def passcode() -> str:
    return PASSCODE


@pytest.fixture(autouse=True)
def fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- throwaway Postgres ---------------------------------------------------------------------
def _find_pg_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    patterns = [
        f"/usr/lib/postgresql/*/bin/{name}",  # Debian and Ubuntu (GitHub runners)
        f"/opt/homebrew/opt/postgresql@*/bin/{name}",
        f"/usr/local/opt/postgresql@*/bin/{name}",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class PgServer:
    port: int

    def dsn(self, dbname: str = "postgres") -> str:
        return f"host=127.0.0.1 port={self.port} user=postgres dbname={dbname}"


@pytest.fixture(scope="session")
def pg_server():
    import psycopg

    initdb, pg_ctl = _find_pg_binary("initdb"), _find_pg_binary("pg_ctl")
    if not initdb or not pg_ctl:
        pytest.skip("Postgres binaries (initdb, pg_ctl) are not installed")
    root = Path(tempfile.mkdtemp(prefix="glpg-"))
    data = root / "data"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    port = _free_port()
    started = False
    try:
        try:
            subprocess.run(
                [initdb, "-D", str(data), "-U", "postgres", "-A", "trust", "-E", "UTF8"],
                check=True,
                capture_output=True,
                env=env,
            )
            options = (
                f"-p {port} -h 127.0.0.1 -k {root} -c fsync=off -c synchronous_commit=off"
                " -c full_page_writes=off -c max_connections=100"
            )
            subprocess.run(
                [
                    pg_ctl,
                    "-D",
                    str(data),
                    "-o",
                    options,
                    "-w",
                    "-t",
                    "60",
                    "-l",
                    str(root / "log"),
                    "start",
                ],
                check=True,
                capture_output=True,
                env=env,
            )
            started = True
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", b"") or b""
            pytest.skip(f"could not start a throwaway Postgres: {exc} {detail.decode()[:300]}")
        server = PgServer(port)
        with psycopg.connect(server.dsn(), autocommit=True) as conn:
            conn.execute("create database gl_template")
        apply_migrations(server.dsn("gl_template"))
        yield server
    finally:
        if started:
            subprocess.run(
                [pg_ctl, "-D", str(data), "-m", "immediate", "-w", "stop"],
                capture_output=True,
                env=env,
            )
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def pg_dsn(pg_server):
    """A fresh, migrated database for one test."""
    import psycopg

    name = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_server.dsn(), autocommit=True) as admin:
        admin.execute(f"create database {name} template gl_template")
    yield pg_server.dsn(name)
    with psycopg.connect(pg_server.dsn(), autocommit=True) as admin:
        admin.execute(f"drop database if exists {name} with (force)")


@pytest.fixture
def db(pg_dsn):
    database = Database(pg_dsn, min_size=1, max_size=8)
    yield database
    database.close()


@pytest.fixture
def repo(db) -> Repo:
    return Repo(db)


# --- settings, paths, app -------------------------------------------------------------------
@pytest.fixture
def paths(tmp_path) -> VolumePaths:
    return VolumePaths(tmp_path / "data")


@pytest.fixture
def settings(tmp_path) -> Settings:
    (tmp_path / "data").mkdir(exist_ok=True)
    return dataclasses.replace(
        get_settings(),
        app_passcode=PASSCODE,
        volume_root=str(tmp_path / "data"),
        model_api_key="",
        job_heartbeat_interval_s=0.05,
        upload_chunk_bytes=1024,
        llm_mode="replay",
    )


@dataclass
class RecordingSpawner:
    """Remembers the jobs it was asked to start, and starts nothing."""

    jobs: list = field(default_factory=list)
    fail_with: Exception | None = None

    def spawn(self, job):
        if self.fail_with:
            raise self.fail_with
        self.jobs.append(job)
        return f"call-{len(self.jobs)}"


@pytest.fixture
def spawner() -> RecordingSpawner:
    return RecordingSpawner()


@dataclass
class RecordingDeleter:
    """Stands in for the Meta Files API: remembers which file ids were deleted."""

    deleted: list = field(default_factory=list)
    failing: set = field(default_factory=set)

    def delete(self, file_id: str) -> bool:
        if file_id in self.failing:
            return False
        self.deleted.append(file_id)
        return True


@pytest.fixture
def file_deleter() -> RecordingDeleter:
    return RecordingDeleter()


@pytest.fixture
def make_client(db, settings, file_deleter):
    """Build a TestClient around a real database. Pass spawner= to choose how jobs start."""
    from fastapi.testclient import TestClient

    from ghostline.api.app import create_app

    def build(spawner, **overrides) -> Any:
        app_settings = dataclasses.replace(settings, **overrides) if overrides else settings
        app = create_app(
            settings=app_settings,
            db=db,
            spawner=spawner,
            volume=NullVolume(),
            file_deleter=file_deleter,
        )
        return TestClient(app, headers=AUTH)

    return build


@pytest.fixture
def client(make_client, spawner):
    return make_client(spawner)


@pytest.fixture
def anon(client):
    """The same app, but a caller who has no passcode."""
    from fastapi.testclient import TestClient

    return TestClient(client.app)


@pytest.fixture
def inline_client(make_client, repo, paths, settings, fake_stages):
    """An API client whose spawned jobs run to completion right away, on the fake stages."""

    def run(job_id):
        run_job(job_id, repo=repo, paths=paths, volume=NullVolume(), settings=settings)

    return make_client(InlineSpawner(run))


# --- fake pipeline stages -------------------------------------------------------------------
class FakeStages:
    """Stand-ins for jobs/stages.py. `fail_on[name]` makes a stage raise that exception."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_on: dict[str, Exception] = {}
        self.simulate_delay_s = 0.0

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        failure = self.fail_on.get(name)
        if failure:
            raise failure

    def ingest(self, ctx, trial):
        self._enter("ingest")
        original = ctx.paths.experiment_dir(trial.experiment_id) / trial.paths["original"]
        master = ctx.paths.trial_file(trial.experiment_id, trial.id, "master.mp4")
        master.write_bytes(original.read_bytes())
        poster = ctx.paths.trial_file(trial.experiment_id, trial.id, "poster.jpg")
        poster.write_bytes(b"jpeg")
        ctx.notebook.append(actor="system", kind="observation", title="Normalized the clip")
        rel = ctx.paths.rel
        return IngestResult(
            probe={"codec": "h264"},
            timing={"classification": "cfr", "time_factor": 1.0},
            n_frames=600,
            duration_s=10.0,
            paths={
                "master": rel(trial.experiment_id, master),
                "poster": rel(trial.experiment_id, poster),
            },
            meta_file_ids={"master": f"file_{trial.id.hex[:8]}"},
            trim=Trim(start_s=0.5, end_s=9.5, suggested_start_s=0.5, suggested_end_s=9.5),
            warnings=["Footage is a little dark."],
        )

    def analyze_scene(self, ctx, experiment, trial):
        self._enter("analyze_scene")
        return SceneAnalysis(
            system_type="pendulum",
            description="An orange ball swinging on a string.",
            tracked_objects=[
                TrackedObject(
                    key="bob", sam_prompt="orange ball", role="moving_body", why="the bob"
                )
            ],
            calibration={"sam_prompt": "white sheet of paper", "found": True},
            object_properties={},
            initial_hypotheses=[],
            quality_issues=[],
            proceed=True,
        )

    def track_objects(self, ctx, trial, scene):
        self._enter("track_objects")
        return TrackingResult(
            tracks=[
                TrackRecord(
                    trial_id=trial.id,
                    object_key="bob",
                    sam_prompt="orange ball",
                    parquet_path=f"trials/{trial.id}/tracks.parquet",
                    stats={"frames": 600},
                )
            ]
        )

    def calibrate(self, ctx, experiment, trial, scene, tracking):
        self._enter("calibrate")
        return CalibrationResult(
            calibration={"mm_per_px": 0.8, "gravity_source": "camera_level"},
            paths={"calib_overlay": f"trials/{trial.id}/calib/overlay.png"},
        )

    def extract_trajectory(self, ctx, experiment, trial, scene, tracking, calibration):
        self._enter("extract_trajectory")
        return TrajectoryResult(
            parquet_path=f"trials/{trial.id}/trajectory.parquet", summary={"n_points": 580}
        )

    def run_physicist(self, ctx, experiment, trial, scene, trajectory, calibration):
        self._enter("run_physicist")
        ctx.phase("hypothesizing")
        ctx.repo.upsert_model(
            ModelRecord(experiment_id=experiment.id, id="P1", name="Nonlinear pendulum")
        )
        ctx.repo.upsert_model(
            ModelRecord(experiment_id=experiment.id, id="P0", name="Small-angle pendulum")
        )
        ctx.notebook.append(
            actor="wonder",
            kind="prediction",
            title="Wonder's favorite: P1",
            body_md="P1 fails if the period doesn't grow with amplitude.",
        )
        ctx.phase("fitting")
        ctx.repo.insert_fit(
            FitRecord(
                experiment_id=experiment.id,
                model_id="P1",
                trial_id=trial.id,
                params=[FitParam(name="g", value=9.8, ci_low=9.6, ci_high=10.0, unit="m/s^2")],
                heldout_rmse=0.002,
            )
        )
        ctx.progress(0.5)
        return Conclusion(
            winner_id="P1",
            quantities=[Quantity(name="g", value=9.8, ci_low=9.6, ci_high=10.0, unit="m/s^2")],
            rejected=[{"id": "P0", "reason": "worse held-out error", "evidence": "0.011 vs 0.002"}],
            confidence="high",
            caveats=[],
            narrative_md="The bob follows the nonlinear pendulum law.",
        )

    def run_review(self, ctx, experiment, conclusion):
        self._enter("run_review")
        ctx.repo.insert_review(
            ReviewRecord(experiment_id=experiment.id, verdict="accept", summary="Holds up.")
        )
        return ReviewOutcome(review_status="accepted", conclusion=conclusion)

    def predict_ghost(self, ctx, experiment, fit_trial, prove_trial, scene, trajectory):
        self._enter("predict_ghost")
        ctx.phase("predicting")
        version = ctx.repo.next_ghost_version(experiment.id)
        ctx.phase("rendering")
        video = ctx.paths.ghost_video(experiment.id, version)
        track = ctx.paths.ghost_track(experiment.id, version)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"ghost-video")
        track.write_text("{}")
        return GhostResult(
            version=version,
            model_id=experiment.winner_model_id,
            video_path=ctx.paths.rel(experiment.id, video),
            track_path=ctx.paths.rel(experiment.id, track),
            divergence_onset_s=2.8,
            max_error_mm=14.0,
        )

    def diagnose_revision(self, ctx, experiment, ghost):
        self._enter("diagnose_revision")
        return RevisionResult(diagnosis_md="Damping was underestimated.", adopted=False)

    def rerun_model(self, ctx, experiment, model, split_pct, initial_guesses):
        self._enter("rerun_model")
        trial = ctx.repo.latest_trial(experiment.id, "fit", status="ready")
        return FitRecord(
            experiment_id=experiment.id,
            model_id=model.id,
            trial_id=trial.id,
            split={"pct": split_pct},
            params=[FitParam(name="g", value=9.81, ci_low=9.5, ci_high=10.1, unit="m/s^2")],
            train_rmse=0.001,
            heldout_rmse=0.003,
        )

    def verify_export(self, ctx, experiment):
        self._enter("verify_export")
        return {"simulator": "pybullet", "quantity": "g", "simulated": 9.79, "passed": True}

    def build_export(self, repo, experiment, fmt, mass_g):
        self._enter("build_export")
        if fmt == "pybullet":
            return "import pybullet as p\n"
        from ghostline.store.core_schemas import SimExport

        return SimExport(
            schema_version="ghostline.sim/1",
            experiment_url=f"https://example.test/e/{experiment.id}",
            object="ball",
            confidence="high",
            mass_kg=None if mass_g is None else mass_g / 1000,
            parameters=[],
            randomization={},
            valid_for={},
        )

    def simulate_custom_model(self, repo, experiment, model, request):
        self._enter("simulate_custom_model")
        if self.simulate_delay_s:
            import time

            time.sleep(self.simulate_delay_s)
        return SimulateResult(model_id=model.id, t=[0.0, 1.0], series={"theta": [0.1, -0.1]})

    NAMES = (
        "ingest",
        "analyze_scene",
        "track_objects",
        "calibrate",
        "extract_trajectory",
        "run_physicist",
        "run_review",
        "predict_ghost",
        "diagnose_revision",
        "rerun_model",
        "verify_export",
        "build_export",
        "simulate_custom_model",
    )


@pytest.fixture
def fake_stages(monkeypatch) -> FakeStages:
    fake = FakeStages()
    for name in FakeStages.NAMES:
        monkeypatch.setattr(stages, name, getattr(fake, name))
    # No sleeping between automatic retries.
    monkeypatch.setattr("ghostline.jobs.handlers.RETRY_BASE_DELAY_S", 0.0)
    return fake
