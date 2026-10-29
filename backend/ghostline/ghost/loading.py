"""Read a trial's measured data and the ghost sprite from the experiment folder on the Volume."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from ghostline.jobs.context import JobContext
from ghostline.physics.library.ramp import STANDARD_GRAVITY
from ghostline.store.core_schemas import SceneAnalysis
from ghostline.store.records import Experiment, Trial
from ghostline.vision import (
    parse_sam_events,
    read_events_jsonl,
    read_frames,
    select_primary,
    stitch_tracks,
    tracks_from_result,
    video_size,
)

from .model import GhostError
from .series import TrialSeries
from .sprite import (
    MaskedTrack,
    Sprite,
    build_sprite,
    disc_sprite,
    pick_frame,
    rank_frames,
)

log = logging.getLogger("ghostline.ghost")

SUPPORTED_SYSTEMS = (
    "free_fall",
    "projectile",
    "pendulum",
    "ramp",
    "bounce",
    "spin_down",
    "oscillator",
)


def body_key(scene: SceneAnalysis) -> str | None:
    """The key of the moving body that was tracked."""
    for obj in scene.tracked_objects:
        if obj.role == "moving_body":
            return obj.key
    return scene.tracked_objects[0].key if scene.tracked_objects else None


def trial_file(ctx: JobContext, trial: Trial, name: str) -> Path | None:
    """A file of the trial from its `paths` (relative to the experiment folder), if it is there."""
    rel = trial.paths.get(name)
    if not rel:
        return None
    path = ctx.paths.experiment_dir(trial.experiment_id) / rel
    return path if path.exists() else None


def load_series(
    ctx: JobContext,
    trial: Trial,
    scene: SceneAnalysis,
    *,
    trajectory_path: str | None = None,
    summary: dict | None = None,
) -> TrialSeries:
    """The trial's trajectory, calibration, timestamps and pixel track.

    `trajectory_path` and `summary` come from the TrajectoryResult when the caller has it (the
    trial that was just measured); otherwise the stored trajectory row is read.
    """
    if trajectory_path is None or summary is None:
        rows = ctx.repo.list_trajectories([trial.id])
        if not rows or not rows[0].parquet_path:
            raise GhostError("That clip has no measured trajectory yet.")
        trajectory_path, summary = rows[0].parquet_path, rows[0].summary
    if not trial.calibration:
        raise GhostError("That clip has no calibration, so the ghost can't be placed on it.")
    root = ctx.paths.experiment_dir(trial.experiment_id)
    key = body_key(scene)
    tracks = next((t for t in ctx.repo.list_tracks([trial.id]) if t.object_key == key), None)
    return TrialSeries.load(
        system=scene.system_type,
        trajectory_parquet=root / trajectory_path,
        summary=summary,
        calibration=trial.calibration,
        timestamps_npy=trial_file(ctx, trial, "timestamps"),
        tracks_parquet=(root / tracks.parquet_path) if tracks and tracks.parquet_path else None,
        object_key=key,
    )


def constants_for(system: str, series: TrialSeries, experiment: Experiment) -> dict[str, float]:
    """The setup constants a model is fitted with, from trial 1's measurements: the pendulum's
    effective length, or the ramp's incline angle."""
    s = series.summary
    if system == "pendulum":
        info = s.get("pendulum")
        if not info:
            raise GhostError("The pendulum's length was not measured in the first clip.")
        return {"L": float(info["length_eff_m"])}
    if system == "ramp":
        ramp = series.calib.ramp
        if ramp is None:
            raise GhostError("The ramp's angle was not measured in the first clip.")
        return {"alpha": math.radians(ramp.incline_deg), "g": STANDARD_GRAVITY}
    return {}


def masked_track(ctx: JobContext, trial: Trial, scene: SceneAnalysis, master: Path):
    """The trial's SAM track with decodable masks, or None when the events weren't kept."""
    key = body_key(scene)
    if key is None:
        return None
    events = ctx.paths.sam_events(trial.experiment_id, trial.id, key)
    if not events.exists():
        return None
    try:
        result = parse_sam_events(read_events_jsonl(events))
        tracks = tracks_from_result(result, video_size(master))
        return select_primary(stitch_tracks(tracks))
    except Exception:
        log.warning("could not read the SAM events at %s", events, exc_info=True)
        return None


def make_sprite(
    master: Path,
    track: MaskedTrack | None,
    series: TrialSeries,
    frames: range | None = None,
) -> Sprite:
    """The sharpest, highest-confidence mask of the moving body, cut out as the ghost sprite.

    Without a SAM track the sprite is a plain disc the size of the body, in the body's colour.
    """
    f0, f1 = series.frame_range
    window = frames or range(f0, f1 + 1)
    angle_col = "phi" if "phi" in series.df else None
    if track is not None:
        ranked = rank_frames(track, window) or rank_frames(track)
        images = read_frames(master, ranked[:5]) if ranked else {}
        frame = pick_frame(track, images, ranked)
        if frame is not None:
            sprite = build_sprite(
                images[frame],
                track.mask(frame),
                source_frame=frame,
                source_angle=_angle_at(series, angle_col, frame),
            )
            if sprite is not None:
                return sprite
    return _disc_from_track(master, series, window)


def _angle_at(series: TrialSeries, column: str | None, frame: int) -> float | None:
    if column is None:
        return None
    rows = series.df
    return float(np.interp(frame, rows["frame"].to_numpy(float), rows[column].to_numpy(float)))


def _disc_from_track(master: Path, series: TrialSeries, window: range) -> Sprite:
    radius_px = series.object_radius_mm() / 1000.0 / max(series.calib.m_per_px, 1e-9)
    radius_px = radius_px if radius_px >= 3 else 8.0
    frame = window[len(window) // 2]
    colour = (200.0, 200.0, 200.0)
    px, measured = series.actual_px(np.array([frame]))
    if measured.any():
        image = read_frames(master, [frame]).get(frame)
        if image is not None:
            x, y = int(round(px[0, 0])), int(round(px[0, 1]))
            patch = image[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3].reshape(-1, 3)
            if len(patch):
                colour = tuple(float(c) for c in np.median(patch, axis=0))
    sprite = disc_sprite(radius_px, colour, frame)
    assert sprite is not None
    return sprite


def write_sprite_png(path: Path, sprite: Sprite) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sprite.to_rgba())
