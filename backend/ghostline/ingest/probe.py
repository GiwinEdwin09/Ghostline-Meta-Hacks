"""ffprobe wrapper: codec, orientation, colour, frame rate, tags and every frame's timestamp."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .models import ProbeInfo

HDR_TRANSFERS = {"arib-std-b67": "hlg", "smpte2084": "pq"}


class IngestError(RuntimeError):
    """The file can't be turned into an analysis master."""


@dataclass
class ProbeResult:
    info: ProbeInfo
    pts: np.ndarray  # file timestamps in seconds, ascending, one per frame


def _run(cmd: list[str]) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise IngestError(f"{cmd[0]} is not installed") from e
    except subprocess.CalledProcessError as e:
        raise IngestError(f"{cmd[0]} failed: {e.stderr.strip()[-500:]}") from e
    return done.stdout


def _rate(text: str | None) -> float | None:
    if not text or text in ("0/0", "N/A"):
        return None
    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None
    return value or None


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return int(round(float(side["rotation"]))) % 360
    rotate = stream.get("tags", {}).get("rotate")
    return int(float(rotate)) % 360 if rotate else 0


def _capture_fps(tags: dict[str, str]) -> float | None:
    for key, value in tags.items():
        if re.search(r"capture.*(fps|frame.?rate)", key, re.IGNORECASE):
            try:
                return float(Fraction(str(value).strip()))
            except (ValueError, ZeroDivisionError):
                continue
    return None


def read_timestamps(path: str | Path) -> np.ndarray:
    """Presentation time of every video frame in file seconds, from packets (no decoding).

    Integer timestamps times the stream time base, because ffprobe's own *_time fields are rounded
    to microseconds, which would skew a frame rate computed from them.
    """
    base = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base",
         "-of", "csv=p=0", str(path)]
    ).strip().strip(",")  # fmt: skip
    try:
        time_base = float(Fraction(base))
    except (ValueError, ZeroDivisionError) as e:
        raise IngestError("the video stream has no time base") from e
    out = _run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts,dts", "-of", "csv=p=0", str(path)]
    )  # fmt: skip
    values = []
    for line in out.splitlines():
        pts, _, dts = line.strip().partition(",")
        for token in (pts, dts.strip(",")):
            if token and token != "N/A":
                values.append(int(token) * time_base)
                break
    if not values:
        raise IngestError("the video stream has no timestamps")
    return np.sort(np.asarray(values, dtype=float))


def probe_file(path: str | Path) -> ProbeResult:
    """Probe a video file. Frame timestamps come from the packets, sorted."""
    meta = json.loads(
        _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    )
    streams = meta.get("streams", [])
    video = next(
        (
            s for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )  # fmt: skip
    if video is None:
        raise IngestError("no video stream found in the file")
    tags = {k.lower(): str(v) for k, v in meta.get("format", {}).get("tags", {}).items()}
    tags.update({k.lower(): str(v) for k, v in video.get("tags", {}).items()})
    pts = read_timestamps(path)
    width, height = int(video["width"]), int(video["height"])
    rotation = _rotation(video)
    swap = rotation in (90, 270)
    transfer = video.get("color_transfer")
    duration = video.get("duration") or meta.get("format", {}).get("duration")
    info = ProbeInfo(
        codec=video.get("codec_name", "unknown"),
        width=width,
        height=height,
        rotation=rotation,
        display_width=height if swap else width,
        display_height=width if swap else height,
        pix_fmt=video.get("pix_fmt"),
        color_transfer=transfer,
        color_primaries=video.get("color_primaries"),
        color_space=video.get("color_space"),
        hdr=HDR_TRANSFERS.get(transfer or ""),
        r_frame_rate=_rate(video.get("r_frame_rate")),
        avg_frame_rate=_rate(video.get("avg_frame_rate")),
        nb_frames=len(pts),
        duration_s=float(duration) if duration else None,
        capture_fps_tag=_capture_fps(tags),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        tags=tags,
    )
    return ProbeResult(info, pts)
