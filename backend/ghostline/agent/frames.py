"""view_frames: still frames of the trial's video for Wonder, with the track drawn on them.

Frames come from the trial's `playback_720.mp4` (else `master.mp4`) through `ctx.paths`. With
`overlay="trail"` the tracked objects' recent path is drawn; with `"mask"` their segmentation mask
is outlined (decoded from the saved SAM event stream, or a disc of the same area when that stream
is not there). Nothing is drawn when the experiment has no tracks. Times after the end of the
training window are refused: those frames show the held-out motion.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ghostline.vision.frames import encode_image, read_frames

from .lab import Lab, LabError

log = logging.getLogger("ghostline.agent")

MAX_FRAMES = 6
MAX_SIDE = 960
TRAIL_FRAMES = 90
_COLORS = [(40, 160, 255), (80, 220, 80), (220, 120, 220), (240, 200, 40)]  # BGR


def _video_path(lab: Lab):
    for key in ("playback_720", "master", "wonder_480"):
        relative = lab.trial.paths.get(key)
        if relative:
            path = lab.ctx.paths.experiment_dir(lab.exp_id) / relative
            if path.is_file():
                return path
    return None


def _timestamps(lab: Lab, n_frames: int | None) -> np.ndarray | None:
    relative = lab.trial.paths.get("timestamps")
    if relative:
        path = lab.ctx.paths.experiment_dir(lab.exp_id) / relative
        if path.is_file():
            return np.load(path)
    duration = lab.trial.duration_s
    if n_frames and duration:
        return np.linspace(0.0, duration, n_frames, endpoint=False)
    return None


def _track_table(lab: Lab) -> pd.DataFrame | None:
    """All tracked objects' per-frame rows (object_key, frame, t, cx, cy, area, ...)."""
    frames = []
    for record in lab.ctx.repo.list_tracks([lab.trial.id]):
        if not record.parquet_path:
            continue
        path = lab.ctx.paths.experiment_dir(lab.exp_id) / record.parquet_path
        if path.is_file() and path not in {f.attrs.get("path") for f in frames}:
            table = pd.read_parquet(path)
            table.attrs["path"] = path
            frames.append(table)
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    return table.drop_duplicates(subset=["object_key", "frame"]) if "object_key" in table else table


def render_frames(
    lab: Lab, times: list[float], overlay: str
) -> tuple[list[tuple[str, bytes]], list[str]]:
    """JPEG bytes with a caption for each requested time, and any notes for Wonder."""
    video = _video_path(lab)
    if video is None:
        raise LabError("This experiment has no video to look at.")
    if not times:
        raise LabError("Give at least one time in seconds.")
    notes: list[str] = []
    if len(times) > MAX_FRAMES:
        notes.append(f"Only the first {MAX_FRAMES} times were used.")
    times = times[:MAX_FRAMES]
    t_split = lab.session.t_split
    allowed = [t for t in times if t <= t_split]
    if len(allowed) < len(times):
        notes.append(
            f"Times after {t_split:.2f} s were left out: that part of the motion is held out."
        )
    if not allowed:
        raise LabError(f"Every time is after the end of the training window ({t_split:.2f} s).")

    n_frames = lab.trial.n_frames
    stamps = _timestamps(lab, n_frames)
    if stamps is None:
        raise LabError("The clip's frame times are not available.")
    indices = [int(np.clip(np.abs(stamps - t).argmin(), 0, len(stamps) - 1)) for t in allowed]
    images = read_frames(video, indices)
    table = _track_table(lab) if overlay != "none" else None
    if overlay != "none" and table is None:
        notes.append("There are no tracks to draw for this experiment.")
    master_w = None
    size = (lab.trial.calibration or {}).get("image_size")
    if size:
        master_w = float(size[0])

    out: list[tuple[str, bytes]] = []
    for t, idx in zip(allowed, indices, strict=True):
        frame = images.get(idx)
        if frame is None:
            notes.append(f"No frame could be read at t = {t:.2f} s.")
            continue
        scale = 1.0 if not master_w else frame.shape[1] / master_w
        if table is not None:
            frame = _draw(lab, frame, table, idx, scale, overlay)
        frame = _fit_side(frame)
        out.append((f"t = {stamps[idx]:.3f} s (frame {idx})", encode_image(frame, ".jpg", 80)))
    return out, notes


def _fit_side(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    biggest = max(h, w)
    if biggest <= MAX_SIDE:
        return frame
    ratio = MAX_SIDE / biggest
    return cv2.resize(frame, (round(w * ratio), round(h * ratio)), interpolation=cv2.INTER_AREA)


def _draw(lab: Lab, frame: np.ndarray, table: pd.DataFrame, idx: int, scale: float, overlay: str):
    out = frame.copy()
    keys = [o.key for o in (lab.experiment.scene.tracked_objects if lab.experiment.scene else [])]
    if "object_key" in table:
        present = list(dict.fromkeys(table["object_key"]))
        keys = [k for k in keys if k in present] or present
    for color, key in zip(_COLORS * 3, keys, strict=False):
        rows = (
            table[table["object_key"] == key].sort_values("frame")
            if "object_key" in table
            else table
        )
        recent = rows[(rows["frame"] <= idx) & (rows["frame"] > idx - TRAIL_FRAMES)]
        if overlay == "trail" and len(recent) >= 2:
            pts = np.round(recent[["cx", "cy"]].to_numpy() * scale).astype(np.int32)
            cv2.polylines(out, [pts], False, color, 2, cv2.LINE_AA)
        here = rows[rows["frame"] == idx]
        if here.empty:
            continue
        cx, cy = float(here["cx"].iloc[0]) * scale, float(here["cy"].iloc[0]) * scale
        if overlay == "mask":
            mask = _mask(lab, key, idx, out.shape[:2])
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
            else:
                area = float(here["area"].iloc[0]) if "area" in here else 0.0
                radius = max(3, round(np.sqrt(max(area, 1.0) / np.pi) * scale))
                cv2.circle(out, (round(cx), round(cy)), radius, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (round(cx), round(cy)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(out, key, (round(cx) + 8, round(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)  # fmt: skip
    return out


def _mask(lab: Lab, key: str, idx: int, shape: tuple[int, int]) -> np.ndarray | None:
    """The object's mask at a frame from the saved SAM events, resized to the frame, or None."""
    try:
        from ghostline.vision import (
            frame_size_of,
            parse_sam_events,
            read_events_jsonl,
            select_primary,
            tracks_from_result,
        )
        from ghostline.vision.parse import output_text_of

        path = lab.ctx.paths.sam_events(lab.exp_id, lab.trial.id, key)
        if not path.is_file():
            return None
        cache: dict[str, Any] = lab.__dict__.setdefault("_mask_tracks", {})
        if key not in cache:
            events = read_events_jsonl(path)
            size = frame_size_of(output_text_of(events)[0])
            track = select_primary(tracks_from_result(parse_sam_events(events), size or (0, 0)))
            cache[key] = track
        track = cache[key]
        if track is None or idx not in set(map(int, track.frame)):
            return None
        mask = np.asarray(track.mask(idx), dtype=np.uint8)
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception:
        log.debug("could not decode the mask of %s", key, exc_info=True)
        return None
