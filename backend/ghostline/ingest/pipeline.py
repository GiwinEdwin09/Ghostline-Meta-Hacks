"""ingest_trial: original upload -> master, derived files, timing model and trim suggestion."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .frames import count_frames
from .models import DeclaredCapture, NormalizeInfo, Notice, ProbeInfo, TimingModel, TrimSuggestion
from .normalize import make_derived, make_poster, normalize_master
from .probe import IngestError, probe_file
from .timing import build_timing, real_timestamps
from .trim import motion_energy, suggest_trim

SAM_MAX_FRAMES = 15_000
DARK_LUMA = 35.0  # median grey level (0-255) below which the clip is called very dark


@dataclass
class IngestResult:
    sha256: str
    trial_dir: Path
    probe: ProbeInfo
    timing: TimingModel
    timestamps: np.ndarray  # real seconds since the first frame
    trim: TrimSuggestion
    normalize: NormalizeInfo
    warnings: list[Notice] = field(default_factory=list)

    @property
    def master(self) -> Path:
        return self.trial_dir / "master.mp4"

    @property
    def needs_confirmation(self) -> bool:
        return self.timing.needs_confirmation


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assemble_chunks(chunks: Iterable[str | Path], dest: str | Path) -> str:
    """Concatenate upload chunks in order into `dest`; returns the sha256 of the result."""
    h = hashlib.sha256()
    with open(dest, "wb") as out:
        for chunk in chunks:
            with open(chunk, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
                    out.write(block)
    return h.hexdigest()


def ingest_trial(
    src: str | Path,
    trial_dir: str | Path,
    declared: DeclaredCapture | None = None,
    zscale: bool | None = None,
) -> IngestResult:
    """Run the whole of Part 5 on one uploaded clip. Writes into `trial_dir`:

    original.<ext>, master.mp4, playback_720.mp4, wonder_480.mp4, audio.wav (if any), poster.jpg,
    probe.json, timing.json, timestamps.npy (real seconds), trim.json and ingest.json.
    Uploading to the Meta Files API is left to the job that calls this.
    """
    declared = declared or DeclaredCapture()
    out = Path(trial_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    original = out / f"original{src.suffix.lower() or '.mp4'}"
    if src.resolve() != original.resolve():
        shutil.copyfile(src, original)
    digest = sha256_file(original)

    probe = probe_file(original)
    timing = build_timing(probe.pts, probe.info, declared)
    notices = list(timing.warnings)
    if probe.info.hdr:
        notices.append(
            Notice(
                code="hdr_tonemapped",
                severity="info",
                message=f"HDR ({probe.info.hdr.upper()}) clip converted to SDR for analysis.",
            )
        )

    norm = normalize_master(original, out / "master.mp4", probe.info, probe.pts, zscale)
    if norm.tonemap == "colorspace_fallback":
        notices.append(
            Notice(
                code="hdr_tonemap_fallback",
                severity="info",
                message="This server's ffmpeg has no zscale filter; HDR was converted by "
                "primaries only and colours may look flat.",
            )
        )
    t_real = real_timestamps(probe.pts, timing)

    lag = max(1, round(timing.real_fps / 60))
    energy, luma, n_decoded = motion_energy(out / "master.mp4", lag)
    if n_decoded != len(probe.pts) or count_frames(out / "master.mp4") != len(probe.pts):
        raise IngestError(
            f"master decodes to {n_decoded} frames but the source has {len(probe.pts)}"
        )
    trim = suggest_trim(energy, t_real, (timing.usable_start, timing.usable_stop))
    if luma < DARK_LUMA:
        notices.append(
            Notice(
                code="dark_clip",
                message="The clip is very dark; tracking may fail. Film in brighter light.",
            )
        )
    if trim.reason == "no_motion_detected":
        notices.append(
            Notice(
                code="no_motion_detected",
                message="Nothing seems to move in this clip. Check that the object is in frame "
                "and the camera is steady.",
            )
        )
    if trim.stop_frame - trim.start_frame > SAM_MAX_FRAMES:
        notices.append(
            Notice(
                code="too_long",
                severity="error",
                message=f"More than {SAM_MAX_FRAMES} frames after trimming. Please trim the clip.",
            )
        )

    make_derived(out / "master.mp4", out, original if probe.info.has_audio else None)
    mid = (trim.start_frame + trim.stop_frame - 1) // 2
    make_poster(out / "master.mp4", out / "poster.jpg", float(probe.pts[mid] - probe.pts[0]))

    np.save(out / "timestamps.npy", t_real)
    (out / "probe.json").write_text(probe.info.model_dump_json(indent=2))
    (out / "timing.json").write_text(timing.model_dump_json(indent=2))
    (out / "trim.json").write_text(trim.model_dump_json(indent=2))
    result = IngestResult(digest, out, probe.info, timing, t_real, trim, norm, notices)
    (out / "ingest.json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "normalize": norm.model_dump(),
                "median_luma": luma,
                "needs_confirmation": result.needs_confirmation,
                "warnings": [n.model_dump() for n in notices],
            },
            indent=2,
        )
    )
    return result
