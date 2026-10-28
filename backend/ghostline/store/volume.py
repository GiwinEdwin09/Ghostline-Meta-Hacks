"""Modal Volume layout and sync helpers (build plan Part 22).

    /experiments/{exp}/trials/{trial}/original.*, master.mp4, playback_720.mp4, wonder_480.mp4,
        audio.wav, poster.jpg, timestamps.npy, probe.json, timing.json,
        sam/{object_key}.events.jsonl, tracks.parquet, trajectory.parquet,
        calib/overlay.png, calib/rectified.png, sprite.png
    /experiments/{exp}/fits/{fit}/fit.png, residuals.png
    /experiments/{exp}/ghost/v{n}.mp4, v{n}_track.json, v{n}_slow.mp4
    /cache/llm/{hash}.json        /cache/sam/{hash}.events.jsonl
    /uploads/{upload_id}/{n}.part

Paths stored in Postgres are relative to the experiment folder, so the media URL for a stored path
is /api/media/{experiment_id}/{path}.
"""

import re
from pathlib import Path
from typing import Protocol
from uuid import UUID

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HASHED_NAME = re.compile(r"(^|[._-])(v\d+|[0-9a-f]{12,})([._-]|$)")


class UnsafePath(ValueError):
    pass


def safe_name(name: str) -> str:
    """A single path component: letters, digits, dot, dash and underscore, never starting with a
    dot, so it can't be '..' or a hidden file."""
    if not _SAFE_NAME.match(name) or ".." in name:
        raise UnsafePath(f"unsafe path component: {name!r}")
    return name


def is_content_hashed(path: str) -> bool:
    """True for file names that embed a version (v2) or a hash, which never change in place."""
    return bool(_HASHED_NAME.search(Path(path).name))


class VolumePaths:
    """Where things live under the Volume mount. Methods return absolute Paths."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # --- experiments and trials -------------------------------------------------------------
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    def experiment_dir(self, experiment_id: UUID | str) -> Path:
        return self.experiments_dir() / safe_name(str(experiment_id))

    def trial_dir(self, experiment_id: UUID | str, trial_id: UUID | str) -> Path:
        return self.experiment_dir(experiment_id) / "trials" / safe_name(str(trial_id))

    def original(self, experiment_id: UUID | str, trial_id: UUID | str, ext: str) -> Path:
        return self.trial_dir(experiment_id, trial_id) / f"original.{safe_name(ext)}"

    def trial_file(self, experiment_id: UUID | str, trial_id: UUID | str, name: str) -> Path:
        """A file directly in the trial folder (master.mp4, probe.json, ...)."""
        return self.trial_dir(experiment_id, trial_id) / safe_name(name)

    def sam_events(self, experiment_id: UUID | str, trial_id: UUID | str, object_key: str) -> Path:
        name = f"{safe_name(object_key)}.events.jsonl"
        return self.trial_dir(experiment_id, trial_id) / "sam" / name

    def calib_file(self, experiment_id: UUID | str, trial_id: UUID | str, name: str) -> Path:
        return self.trial_dir(experiment_id, trial_id) / "calib" / safe_name(name)

    def fit_dir(self, experiment_id: UUID | str, fit_id: UUID | str) -> Path:
        return self.experiment_dir(experiment_id) / "fits" / safe_name(str(fit_id))

    def ghost_video(self, experiment_id: UUID | str, version: int) -> Path:
        return self.experiment_dir(experiment_id) / "ghost" / f"v{int(version)}.mp4"

    def ghost_track(self, experiment_id: UUID | str, version: int) -> Path:
        return self.experiment_dir(experiment_id) / "ghost" / f"v{int(version)}_track.json"

    def ghost_slow(self, experiment_id: UUID | str, version: int) -> Path:
        return self.experiment_dir(experiment_id) / "ghost" / f"v{int(version)}_slow.mp4"

    # --- caches, uploads, backups -----------------------------------------------------------
    def llm_cache(self, key: str) -> Path:
        return self.root / "cache" / "llm" / f"{safe_name(key)}.json"

    def sam_cache(self, key: str) -> Path:
        return self.root / "cache" / "sam" / f"{safe_name(key)}.events.jsonl"

    def uploads_dir(self) -> Path:
        return self.root / "uploads"

    def upload_dir(self, upload_id: UUID | str) -> Path:
        return self.uploads_dir() / safe_name(str(upload_id))

    def upload_part(self, upload_id: UUID | str, n: int) -> Path:
        return self.upload_dir(upload_id) / f"{int(n)}.part"

    def backups_dir(self) -> Path:
        return self.root / "backups"

    # --- conversion -------------------------------------------------------------------------
    def rel(self, experiment_id: UUID | str, path: Path | str) -> str:
        """The experiment-relative form stored in Postgres and used in media URLs."""
        return Path(path).relative_to(self.experiment_dir(experiment_id)).as_posix()

    def resolve_media(self, experiment_id: UUID | str, relative: str) -> Path:
        """Map a media URL path to a file inside the experiment folder, or raise UnsafePath."""
        if not relative or "\\" in relative or "\x00" in relative or relative.startswith("/"):
            raise UnsafePath(relative)
        parts = relative.split("/")
        if any(p in ("", ".", "..") or p.startswith(".") for p in parts):
            raise UnsafePath(relative)
        base = self.experiment_dir(experiment_id).resolve()
        target = (base / relative).resolve()
        if base != target and base not in target.parents:
            raise UnsafePath(relative)
        return target


class VolumeSync(Protocol):
    """The two Modal Volume calls the app depends on (Part 22 volume rules)."""

    def commit(self) -> None: ...

    def reload(self) -> None: ...


class NullVolume:
    """For local development and tests, where the directory is already shared."""

    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class ModalVolume:
    """Wraps a modal.Volume. Workers commit after each phase; the API reloads before serving a
    path it hasn't seen. Errors propagate: a worker must not lose a commit silently, and the API
    wraps reload in try/except because a busy volume must not fail a request."""

    def __init__(self, volume):
        self._volume = volume

    def commit(self) -> None:
        self._volume.commit()

    def reload(self) -> None:
        self._volume.reload()
