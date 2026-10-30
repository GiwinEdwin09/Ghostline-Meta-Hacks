"""The analysis window, the trimmed clip SAM sees, and uploads to the Meta Files API
(build plan Parts 5 and 7).

SAM numbers the frames of the video it is sent from 0. Ghostline sends the master trimmed to the
analysis window, so SAM frame n is master frame `window.start + n`; every track written to disk
uses master frame numbers.

Uploaded files are named by what they hold (`wonder_480`, `master@120-480`) and remembered in
`trials.meta_file_ids`. A retry reuses them while they are young enough (files expire after 7
days). Their identity in the replay cache is derived from the sha256 of the original upload, not
from the bytes of the derived file, so a recording made with one ffmpeg build replays with
another.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ghostline.config import Settings
from ghostline.ingest.normalize import trim_master
from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.store.records import Trial

log = logging.getLogger("ghostline.jobs")

MIN_WINDOW_FRAMES = 10
# Meta Files uploads expire after 7 days; a file older than this is uploaded again.
FILE_REUSE_MAX_AGE = timedelta(days=6)
WONDER_KEY = "wonder_480"


@dataclass(frozen=True)
class Window:
    """Master frames [start, stop) that the analysis uses."""

    start: int
    stop: int

    @property
    def n(self) -> int:
        return self.stop - self.start

    @property
    def key(self) -> str:
        return f"master@{self.start}-{self.stop}"

    def covers(self, n_frames: int) -> bool:
        return self.start == 0 and self.stop >= n_frames


def analysis_window(trial: Trial, t_real: np.ndarray, settings: Settings) -> Window:
    """The frames between the trial's trim, within the usable range of the timing model."""
    n = len(t_real)
    timing = trial.timing or {}
    lo = max(int(timing.get("usable_start", 0)), 0)
    hi = min(int(timing.get("usable_stop", n)), n)
    if trial.trim is not None:
        eps = 1e-4  # a trim edge that falls on a frame time keeps that frame at the start only
        lo = max(lo, int(np.searchsorted(t_real, trial.trim.start_s - eps, side="left")))
        hi = min(hi, int(np.searchsorted(t_real, trial.trim.end_s - eps, side="left")))
    if hi - lo < MIN_WINDOW_FRAMES:
        raise StageError(
            f"The trimmed clip holds only {max(hi - lo, 0)} frames. Widen the trim so it covers "
            "the whole motion.",
            code="window_too_short",
            retryable=False,
        )
    if hi - lo > settings.max_frames:
        raise StageError(
            f"The trimmed clip holds {hi - lo} frames; the limit is {settings.max_frames}. "
            "Trim it shorter.",
            code="too_many_frames",
            retryable=False,
        )
    return Window(lo, hi)


def sam_clip_path(ctx: JobContext, trial: Trial, window: Window, n_frames: int) -> Path:
    """Where the clip for this window lives: the master itself when the window is all of it."""
    master = ctx.paths.trial_file(trial.experiment_id, trial.id, "master.mp4")
    if window.covers(n_frames):
        return master
    return ctx.paths.trial_file(
        trial.experiment_id, trial.id, f"master_{window.start}-{window.stop}.mp4"
    )


def ensure_sam_clip(ctx: JobContext, trial: Trial, window: Window, n_frames: int) -> Path:
    """The trimmed clip SAM is sent, written if it isn't there yet."""
    clip = sam_clip_path(ctx, trial, window, n_frames)
    if not clip.is_file():
        master = ctx.paths.trial_file(trial.experiment_id, trial.id, "master.mp4")
        trim_master(master, clip, window.start, window.stop)
    return clip


def cache_identity(trial: Trial, key: str) -> str:
    """A stable sha256-shaped identity for a derived file, independent of its encoded bytes."""
    return hashlib.sha256(f"{trial.sha256 or trial.id}:{key}".encode()).hexdigest()


def register_file(ctx: JobContext, trial: Trial, key: str, file_id: str) -> str:
    """Tell the runtime what an uploaded file is, so cache keys don't depend on its ID."""
    identity = cache_identity(trial, key)
    ctx.runtime.file_hashes[file_id] = identity
    return identity


def ensure_uploaded(ctx: JobContext, trial: Trial, key: str, path: Path) -> tuple[str, str]:
    """(file ID, cache identity) of `path` on the Files API, uploading it unless this trial
    already has a fresh upload under `key`."""
    trial = ctx.repo.get_trial(trial.id) or trial
    known = trial.meta_file_ids.get(key)
    young = datetime.now(UTC) - trial.created_at < FILE_REUSE_MAX_AGE
    if known and young:
        return known, register_file(ctx, trial, key, known)
    uploaded = ctx.meta.upload_file(path)
    ctx.repo.merge_trial_meta_file_ids(trial.id, {key: uploaded.id})
    return uploaded.id, register_file(ctx, trial, key, uploaded.id)
