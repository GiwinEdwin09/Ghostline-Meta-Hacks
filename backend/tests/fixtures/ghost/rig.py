"""A two-trial pendulum experiment on disk, for the ghost tests: no database, no SAM, no network.

Trial 1 and trial 2 come from the synthetic generator's analytic tracks (true motion, projected
through a real camera, with pixel noise), run through the real trajectory stage. Trial 2's master
is a plain video with a disc drawn where the tracker saw the bob, so rendering needs no SAM masks.
"""

from __future__ import annotations

import math
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd

TESTS = Path(__file__).resolve().parents[2]
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from synth_helpers import analytic_tracks  # noqa: E402

from ghostline.jobs.context import JobContext  # noqa: E402
from ghostline.jobs.stage_types import TrajectoryResult  # noqa: E402
from ghostline.store.core_schemas import SceneAnalysis, TrackedObject  # noqa: E402
from ghostline.store.records import (  # noqa: E402
    DeclaredCapture,  # noqa: E402
    Event,
    Experiment,
    Job,
    ModelRecord,
    Setup,
    TrackRecord,
    TrajectoryRecord,
    Trial,
)
from ghostline.store.volume import NullVolume, VolumePaths  # noqa: E402
from ghostline.traj import build_trajectory  # noqa: E402

WIDTH, HEIGHT = 854, 480


class FakeRepo:
    """The slice of `Repo` the ghost stages use, in memory."""

    def __init__(self) -> None:
        self.models: dict[tuple[uuid.UUID, str], ModelRecord] = {}
        self.trials: dict[uuid.UUID, Trial] = {}
        self.trajectories: dict[uuid.UUID, TrajectoryRecord] = {}
        self.tracks: list[TrackRecord] = []
        self.events: list[dict[str, Any]] = []
        self.phases: list[str] = []
        self.progress: list[float] = []
        self.ghost_count = 0

    def get_model(self, experiment_id, model_id):
        return self.models.get((experiment_id, model_id))

    def list_models(self, experiment_id):
        return [m for (e, _), m in self.models.items() if e == experiment_id]

    def get_trial(self, trial_id):
        return self.trials.get(trial_id)

    def list_trajectories(self, trial_ids):
        return [self.trajectories[t] for t in trial_ids if t in self.trajectories]

    def list_tracks(self, trial_ids):
        return [t for t in self.tracks if t.trial_id in trial_ids]

    def next_ghost_version(self, experiment_id):
        return self.ghost_count + 1

    def set_job_phase(self, job_id, phase, progress):
        self.phases.append(phase)
        return True

    def set_job_progress(self, job_id, progress):
        self.progress.append(progress)
        return True

    def append_event(self, experiment_id, **kwargs):
        self.events.append(kwargs)
        return Event(
            id=len(self.events),
            experiment_id=experiment_id,
            seq=len(self.events),
            ts=datetime.now(UTC),
            phase=kwargs.get("phase"),
            actor=kwargs["actor"],
            kind=kwargs["kind"],
            title=kwargs["title"],
            body_md=kwargs.get("body_md", ""),
            data=kwargs.get("data") or {},
            attachments=kwargs.get("attachments") or [],
            refs=kwargs.get("refs") or [],
        )


@dataclass
class Rig:
    ctx: JobContext
    repo: FakeRepo
    experiment: Experiment
    fit_trial: Trial
    prove_trial: Trial
    scene: SceneAnalysis
    trajectory: TrajectoryResult
    paths: VolumePaths
    system: str = "pendulum"
    extra: dict[str, Any] = field(default_factory=dict)


def _write_video(path: Path, px: np.ndarray, radius_px: float, fps: float, n: int) -> None:
    """A dark textured video with a bright disc at `px` on each frame."""
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur(rng.normal(60, 20, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 8)
    background = np.repeat(np.clip(background, 20, 110)[:, :, None], 3, axis=2).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.codec_context.options = {"crf": "18", "preset": "veryfast", "bf": "0"}
    for i in range(n):
        frame = background.copy()
        cv2.circle(
            frame, (round(px[i, 0]), round(px[i, 1])), round(radius_px), (60, 140, 255), -1,
            cv2.LINE_AA,
        )  # fmt: skip
        out = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in stream.encode(out):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _trial(
    paths: VolumePaths,
    repo: FakeRepo,
    experiment_id,
    role: str,
    system: str,
    dur: float,
    fps: float,
    seed: int,
    params: dict,
    video: bool,
):
    sc, cal, track, _, _ = analytic_tracks(system, dur, fps, sigma_px=0.2, seed=seed, params=params)
    res = build_trajectory(track, cal, system)
    trial_id = uuid.uuid4()
    folder = paths.trial_dir(experiment_id, trial_id)
    folder.mkdir(parents=True, exist_ok=True)
    n = int(round(dur * fps))
    np.save(folder / "timestamps.npy", np.arange(n) / fps)
    res.trajectory.to_parquet(folder / "trajectory.parquet", index=False)
    table = pd.DataFrame(
        {
            "object_key": "bob", "frame": track.frame, "t": track.t, "cx": track.cx,
            "cy": track.cy, "area": track.area, "conf": track.conf,
        }
    )  # fmt: skip
    table.to_parquet(folder / "tracks.parquet", index=False)
    rel = lambda name: paths.rel(experiment_id, folder / name)  # noqa: E731
    trial_paths = {"timestamps": rel("timestamps.npy")}
    if video:
        radius = float(np.sqrt(np.median(track.area) / np.pi))
        full = np.full((n, 2), np.nan)
        full[track.frame] = track.px
        for k in range(2):
            ok = np.isfinite(full[:, k])
            full[:, k] = np.interp(np.arange(n), np.nonzero(ok)[0], full[ok, k])
        _write_video(folder / "master.mp4", full, radius, fps, n)
        trial_paths["master"] = rel("master.mp4")
    trial = Trial(
        id=trial_id,
        experiment_id=experiment_id,
        role=role,  # type: ignore[arg-type]
        status="ready",
        paths=trial_paths,
        n_frames=n,
        duration_s=dur,
        calibration=cal.model_dump(mode="json"),
        created_at=datetime.now(UTC),
    )
    repo.trials[trial_id] = trial
    traj_record = TrajectoryRecord(
        trial_id=trial_id, parquet_path=rel("trajectory.parquet"), summary=_plain(res.summary)
    )
    repo.trajectories[trial_id] = traj_record
    repo.tracks.append(
        TrackRecord(trial_id=trial_id, object_key="bob", parquet_path=rel("tracks.parquet"))
    )
    return trial, traj_record, sc


def _plain(value):
    """JSON-safe copy of a summary (numpy scalars and arrays become Python values)."""
    import json

    from ghostline.traj.model import _jsonable

    return json.loads(json.dumps(value, default=_jsonable))


def pendulum_rig(
    paths: VolumePaths,
    settings,
    *,
    gamma: float = 0.0,
    theta1: float = 0.5,
    theta2: float = 0.35,
    d1: float = 8.0,
    d2: float = 10.0,
    fps: float = 60.0,
    winner: str = "P1",
    video: bool = True,
    seeds: tuple[int, int] = (1, 2),
) -> Rig:
    """Experiment with trial 1 (`d1` s) and trial 2 (`d2` s) of the same damped pendulum."""
    repo = FakeRepo()
    experiment_id = uuid.uuid4()
    system = "pendulum"
    fit, _, _ = _trial(
        paths, repo, experiment_id, "fit", system, d1, fps, seeds[0],
        {"gamma": gamma, "theta0": theta1}, False,
    )  # fmt: skip
    prove, traj2, _ = _trial(
        paths, repo, experiment_id, "prove", system, d2, fps, seeds[1],
        {"gamma": gamma, "theta0": theta2}, video,
    )  # fmt: skip
    experiment = Experiment(
        id=experiment_id,
        created_at=datetime.now(UTC),
        title="A pendulum",
        system_type=system,
        status="complete",
        declared_capture=DeclaredCapture(),
        setup=Setup(),
        winner_model_id=winner,
    )
    scene = SceneAnalysis(
        system_type="pendulum",
        description="A steel ball on a string.",
        tracked_objects=[
            TrackedObject(key="bob", sam_prompt="steel ball", role="moving_body", why="the bob")
        ],
        calibration={"found": True},
        object_properties={},
        initial_hypotheses=[],
        quality_issues=[],
        proceed=True,
    )
    job = Job(
        id=uuid.uuid4(),
        experiment_id=experiment_id,
        kind="prove_trial",
        status="running",
        created_at=datetime.now(UTC),
        heartbeat_at=datetime.now(UTC),
    )
    ctx = JobContext(job=job, repo=repo, paths=paths, volume=NullVolume(), settings=settings)  # type: ignore[arg-type]
    trajectory = TrajectoryResult(parquet_path=traj2.parquet_path or "", summary=traj2.summary)
    return Rig(ctx, repo, experiment, fit, prove, scene, trajectory, paths, system)


def series_for(system: str, dur: float, fps: float, seed: int, params: dict | None = None):
    """A `TrialSeries` from analytic tracks (no video), and the harness constants for `system`."""
    from ghostline.ghost.loading import constants_for
    from ghostline.ghost.series import TrialSeries
    from ghostline.physics.library.ramp import STANDARD_GRAVITY

    sc, cal, track, _, _ = analytic_tracks(
        system, dur, fps, sigma_px=0.2, seed=seed, params=params or {}
    )
    kw = {}
    if system == "spin_down":
        _, _, hub, _, _ = analytic_tracks(
            system, dur, fps, key="hub", seed=seed + 9, params=params or {}
        )
        kw["hub"] = hub
    res = build_trajectory(track, cal, system, **kw)
    table = pd.DataFrame({"frame": track.frame, "cx": track.cx, "cy": track.cy, "area": track.area})
    series = TrialSeries(
        system, res.trajectory, res.summary, cal, np.arange(int(round(dur * fps))) / fps, table
    )
    if system == "ramp":  # the synthetic calibration has no ramp mask: use the true incline
        alpha = math.radians((params or {}).get("alpha_deg", 20.0))
        return series, {"alpha": alpha, "g": STANDARD_GRAVITY}
    return series, constants_for(system, series, None)  # type: ignore[arg-type]
