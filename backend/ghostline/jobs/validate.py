"""Checks on an uploaded file before anything is done with it (build plan Part 28).

The file is whatever a stranger's browser sent, so before ffmpeg decodes it: `ffprobe` must
succeed, it must hold a video (not an image, a document or an audio file), it may carry audio
and the data tracks phones add but nothing else, and it must stay within the size and frame
limits. Filenames play no part: the original is stored under a name Ghostline chose.
"""

import json
import subprocess
from pathlib import Path

from ghostline.config import Settings
from ghostline.ingest import IngestError, probe_file
from ghostline.ingest.models import ProbeInfo
from ghostline.jobs.errors import StageError

# Phones put timecode and metadata tracks ("data") next to the video and audio.
ALLOWED_STREAM_TYPES = {"video", "audio", "data"}
MIN_FRAMES = 10  # a still image decodes as a one-frame "video"


def _refuse(message: str, code: str) -> StageError:
    return StageError(message, code=code, retryable=False)


def _stream_types(path: Path) -> list[str]:
    try:
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json",
             str(path)],
            capture_output=True, text=True, check=True, timeout=120,
        )  # fmt: skip
    except FileNotFoundError as exc:
        raise StageError("ffprobe is not installed on the server.", code="not_configured") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise _refuse(
            "That file isn't a video Ghostline can read. Upload an MP4 or MOV from your camera.",
            "not_a_video",
        ) from exc
    try:
        return [s.get("codec_type", "unknown") for s in json.loads(done.stdout).get("streams", [])]
    except ValueError as exc:
        raise _refuse("That file isn't a video Ghostline can read.", "not_a_video") from exc


def validate_upload(path: Path, settings: Settings) -> ProbeInfo:
    """Return the file's probe, or raise a non-retryable StageError saying what is wrong."""
    size = path.stat().st_size
    if size > settings.max_upload_bytes:
        limit = settings.max_upload_bytes / 1024**3
        raise _refuse(
            f"Clips can be at most {limit:.0f} GiB. Trim it and upload again.", "too_large"
        )
    types = _stream_types(path)
    if "video" not in types:
        raise _refuse(
            "That file has no video in it. Upload a video recorded with your camera.",
            "not_a_video",
        )
    extra = sorted(set(types) - ALLOWED_STREAM_TYPES)
    if extra:
        raise _refuse(
            f"That file carries {', '.join(extra)} streams, which Ghostline doesn't accept. "
            "Upload the plain video from your camera.",
            "unsupported_streams",
        )
    try:
        probe = probe_file(path).info
    except IngestError as exc:
        raise _refuse(f"That video can't be read: {exc}", "not_a_video") from exc
    if probe.nb_frames < MIN_FRAMES:
        raise _refuse(
            f"That file has only {probe.nb_frames} video frames. Upload a video of the motion.",
            "not_a_video",
        )
    if probe.nb_frames > settings.max_frames:
        raise _refuse(
            f"That clip has {probe.nb_frames} frames; the limit is {settings.max_frames}. "
            "Trim it to the motion in your phone's Photos app and upload it again.",
            "too_many_frames",
        )
    return probe
