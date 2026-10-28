"""Modal app: image, storage, secrets and functions (build plan Parts 21 and 27).

Run every Modal command from the repo root, because uv_sync reads pyproject.toml and uv.lock
from the current directory:

    uv run modal serve backend/modal_app.py    # live-reloading dev URL
    uv run modal deploy backend/modal_app.py   # production

Keep the API warm for judging (Part 27) by deploying with GHOSTLINE_KEEP_WARM=1.
"""

import os

import modal

app = modal.App("ghostline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .uv_sync(extra_options="--no-dev")
    # The default ignore drops non-Python files, which would leave out the prompt templates.
    .add_local_python_source("ghostline", ignore=["**/*.pyc"])
)

# The scientific stack only, for the network-blocked fitting Sandbox (Part 11). PyBullet for the
# export replay check (Part 19) is added when that part pins a Python version it has wheels for.
fit_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy", "scipy", "pandas", "matplotlib", "sympy"
)

VOLUME_PATH = "/data"  # matches the VOLUME_ROOT default in ghostline.config
data = modal.Volume.from_name("ghostline-data", create_if_missing=True)
secrets = [modal.Secret.from_name("ghostline")]

KEEP_WARM = os.environ.get("GHOSTLINE_KEEP_WARM") == "1"
worker = {"image": image, "volumes": {VOLUME_PATH: data}, "secrets": secrets}


def _run_job(job_id: str) -> str:
    # Imported inside the function so the container only loads the package when a job runs.
    from ghostline.jobs.worker import run_job_entry
    from ghostline.store.volume import ModalVolume

    return run_job_entry(job_id, ModalVolume(data))


# Each worker takes the id of a row in the jobs table. Failures are recorded on that row, so no
# Modal-level retries: a crashed worker is caught by the heartbeat timeout and the user retries.
@app.function(**worker, cpu=4, memory=8192, timeout=900)
def ingest_trial(job_id: str) -> str:
    return _run_job(job_id)


@app.function(**worker, cpu=2, memory=4096, timeout=1800)
def analyze_experiment(job_id: str) -> str:
    return _run_job(job_id)


@app.function(**worker, cpu=4, memory=8192, timeout=1200)
def prove_trial(job_id: str) -> str:
    return _run_job(job_id)


# Abandoned uploads (chunks that never got a "complete") would otherwise fill the Volume.
@app.function(**worker, cpu=0.5, memory=512, timeout=300, schedule=modal.Period(hours=6))
def purge_stale_uploads() -> int:
    from ghostline.config import get_settings
    from ghostline.store.db import Database
    from ghostline.store.repo import Repo
    from ghostline.store.retention import purge_stale_uploads as purge
    from ghostline.store.volume import ModalVolume, VolumePaths

    settings = get_settings()
    volume = ModalVolume(data)
    volume.reload()
    db = Database(settings.database_url, min_size=1, max_size=2)
    try:
        return purge(
            Repo(db),
            VolumePaths(settings.volume_root),
            volume,
            older_than_hours=settings.stale_upload_hours,
        )
    finally:
        db.close()


# Many requests share one container: chunk uploads, 1 s polling and media reads would otherwise
# queue behind each other at Modal's default of one request per container.
@app.function(**worker, cpu=0.5, memory=512, max_containers=5, min_containers=1 if KEEP_WARM else 0)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def api():
    from ghostline.api.app import create_app
    from ghostline.jobs.spawn import ModalSpawner
    from ghostline.store.volume import ModalVolume

    spawner = ModalSpawner(
        {
            "ingest_trial": ingest_trial,
            "analyze_experiment": analyze_experiment,
            "prove_trial": prove_trial,
        }
    )
    return create_app(volume=ModalVolume(data), spawner=spawner)
