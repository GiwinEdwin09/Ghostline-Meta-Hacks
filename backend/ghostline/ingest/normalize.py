"""Normalization to the analysis master, plus the derived playback, Wonder and audio files."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

import numpy as np

from .models import NormalizeInfo, ProbeInfo
from .probe import IngestError, probe_file, read_timestamps

MAX_SIDE = 1920
TONEMAP_ZSCALE = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)
# HLG is designed to look acceptable as plain BT.2020 SDR, so without zscale only the primaries
# and matrix are converted. PQ will look dark but tracks fine.
TONEMAP_FALLBACK = "format=yuv420p10le,colorspace=all=bt709:iall=bt2020:fast=1,format=yuv420p"
SCALE_CAP = (
    f"scale='if(gt(iw,ih),min({MAX_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({MAX_SIDE},ih))'"
    ":flags=lanczos"
)


def _ffmpeg(*args: str) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
    except FileNotFoundError as e:
        raise IngestError("ffmpeg is not installed") from e
    except subprocess.CalledProcessError as e:
        raise IngestError(f"ffmpeg failed: {e.stderr.strip()[-800:]}") from e


@functools.cache
def has_filter(name: str) -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines())


def master_filters(probe: ProbeInfo, zscale: bool | None = None) -> tuple[str, str]:
    """The -vf chain for master.mp4 and which tone-map path it takes.

    ffmpeg applies the display matrix before the filters, so scaling sees the upright image.
    """
    parts = []
    tonemap = "none"
    if probe.hdr:
        zscale = has_filter("zscale") if zscale is None else zscale
        parts.append(TONEMAP_ZSCALE if zscale else TONEMAP_FALLBACK)
        tonemap = "zscale_hable" if zscale else "colorspace_fallback"
    parts.append(SCALE_CAP)
    parts.append("format=yuv420p")
    return ",".join(parts), tonemap


def _pts_error(src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    if len(src_pts) != len(dst_pts):
        return float("inf")
    return float(np.max(np.abs((dst_pts - dst_pts[0]) - (src_pts - src_pts[0]))))


def normalize_master(
    src: str | Path, dst: str | Path, probe: ProbeInfo, src_pts: np.ndarray, zscale=None
) -> NormalizeInfo:
    """Write master.mp4: upright, SDR BT.709, long side <= 1920, same frames and timestamps."""
    filters, tonemap = master_filters(probe, zscale)
    _ffmpeg(
        "-i", str(src), "-map", "0:v:0", "-an", "-vf", filters,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-color_range", "tv", "-fps_mode", "passthrough", "-enc_time_base", "1/90000",
        "-video_track_timescale", "90000", "-movflags", "+faststart", str(dst),
    )  # fmt: skip
    out = probe_file(dst)
    if len(out.pts) != len(src_pts):
        raise IngestError(
            f"master has {len(out.pts)} frames but the source has {len(src_pts)}; "
            "frame indices would no longer match"
        )
    if out.info.rotation or out.info.hdr:
        raise IngestError("master still carries a rotation tag or HDR transfer")
    return NormalizeInfo(
        filters=filters,
        tonemap=tonemap,
        width=out.info.display_width,
        height=out.info.display_height,
        n_frames=len(out.pts),
        max_pts_error_s=_pts_error(src_pts, out.pts),
    )


def trim_master(src: str | Path, dst: str | Path, start: int, stop: int) -> int:
    """Write frames [start, stop) of the master as a new file; returns its frame count.

    The result is what gets uploaded to SAM, so SAM frame n is master frame start + n.
    """
    _ffmpeg(
        "-i", str(src), "-an",
        "-vf", f"trim=start_frame={start}:end_frame={stop},setpts=PTS-STARTPTS",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough", "-movflags", "+faststart", str(dst),
    )  # fmt: skip
    n = len(read_timestamps(dst))
    if n != stop - start:
        raise IngestError(f"trimmed file has {n} frames, expected {stop - start}")
    return n


def _short_side(limit: int) -> str:
    return f"scale='if(gt(iw,ih),-2,min({limit},iw))':'if(gt(iw,ih),min({limit},ih),-2)'"


def make_derived(master: str | Path, out_dir: str | Path, has_audio_from: str | Path | None):
    """playback_720.mp4, wonder_480.mp4 and, if the source has audio, audio.wav."""
    out = Path(out_dir)
    _ffmpeg(
        "-i", str(master), "-an", "-vf", _short_side(720), "-c:v", "libx264", "-crf", "23",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
        "-movflags", "+faststart", str(out / "playback_720.mp4"),
    )  # fmt: skip
    _ffmpeg(
        "-i", str(master), "-an", "-vf", f"fps=10,{_short_side(480)}", "-c:v", "libx264",
        "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out / "wonder_480.mp4"),
    )  # fmt: skip
    if has_audio_from:
        _ffmpeg(
            "-i",
            str(has_audio_from),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(out / "audio.wav"),
        )


def make_poster(master: str | Path, dest: str | Path, at_s: float) -> None:
    _ffmpeg(
        "-ss", f"{max(at_s, 0):.3f}", "-i", str(master), "-frames:v", "1", "-q:v", "3", str(dest)
    )
