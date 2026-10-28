"""Entry point the Modal Functions call: build the dependencies from settings and run one job."""

from uuid import UUID

from ghostline.config import get_settings
from ghostline.jobs.log import configure_logging
from ghostline.jobs.runner import run_job
from ghostline.store.db import Database
from ghostline.store.repo import Repo
from ghostline.store.volume import NullVolume, VolumePaths, VolumeSync


def run_job_entry(job_id: str, volume: VolumeSync | None = None) -> str:
    """Run job `job_id` in this process and return its final status.

    `volume` is the Modal Volume wrapper; the job sees the other containers' commits after a
    reload, and its own writes are committed at each phase and at the end."""
    configure_logging()
    settings = get_settings()
    volume = volume or NullVolume()
    volume.reload()
    db = Database(settings.database_url, min_size=1, max_size=3)
    try:
        job = run_job(
            UUID(job_id),
            repo=Repo(db),
            paths=VolumePaths(settings.volume_root),
            volume=volume,
            settings=settings,
        )
        return job.status if job else "missing"
    finally:
        db.close()
