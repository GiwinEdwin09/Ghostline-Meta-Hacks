"""view_frames: frames from the trial's video, the track drawn on them, no held-out times."""

import cv2
import numpy as np
import pandas as pd
import pytest
from fixtures.agent.support import make_world

from ghostline.agent.frames import render_frames
from ghostline.agent.lab import Lab, LabError
from ghostline.agent.rerun import load_inputs
from ghostline.agent.runners import RunnerPool
from ghostline.store.records import TrackRecord

FPS, N, SIZE = 30, 60, (320, 240)


@pytest.fixture
def lab(repo, paths, settings):
    world = make_world(repo, paths, settings)
    exp, trial = world.experiment, world.trial
    video = paths.trial_file(exp.id, trial.id, "playback_720.mp4")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    cx = np.linspace(40, 280, N)
    for i in range(N):
        frame = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
        cv2.circle(frame, (int(cx[i]), 120), 12, (40, 160, 255), -1)
        writer.write(frame)
    writer.release()
    stamps = paths.trial_file(exp.id, trial.id, "timestamps.npy")
    np.save(stamps, np.arange(N) / FPS)
    table = paths.trial_file(exp.id, trial.id, "tracks.parquet")
    pd.DataFrame(
        {
            "object_key": "marker",
            "frame": np.arange(N),
            "t": np.arange(N) / FPS,
            "cx": cx,
            "cy": 120.0,
            "area": np.pi * 12**2,
        }
    ).to_parquet(table, index=False)
    repo.upsert_track(
        TrackRecord(trial_id=trial.id, object_key="marker", parquet_path=paths.rel(exp.id, table))
    )
    repo.update_trial(
        trial.id,
        paths={
            "playback_720": paths.rel(exp.id, video),
            "timestamps": paths.rel(exp.id, stamps),
        },
    )
    trial = repo.get_trial(trial.id)
    inputs = load_inputs(world.ctx, world.experiment, trial)
    pool = RunnerPool("inprocess")
    yield Lab(world.ctx, world.experiment, trial, inputs, Lab.new_session(inputs), pool)
    pool.close()


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_frames_come_from_the_video_at_the_asked_times(lab):
    frames, notes = render_frames(lab, [0.0, 1.0], "none")

    assert [c for c, _ in frames] == ["t = 0.000 s (frame 0)", "t = 1.000 s (frame 30)"]
    first, second = (decode(j) for j, _ in [(f[1], 0) for f in frames])
    assert first.shape[:2] == (240, 320) and not notes
    dot = lambda img: np.argwhere((img[..., 2] > 200) & (img[..., 0] < 100))[:, 1].mean()  # noqa: E731
    assert dot(second) > dot(first) + 100  # the object moved right between the two frames


def test_trail_and_mask_overlays_draw_on_the_frame(lab):
    plain, _ = render_frames(lab, [1.0], "none")
    trail, _ = render_frames(lab, [1.0], "trail")
    mask, _ = render_frames(lab, [1.0], "mask")  # no SAM events saved: a disc of the same area

    base = decode(plain[0][1]).astype(int)
    assert np.abs(decode(trail[0][1]).astype(int) - base).sum() > 5000
    assert np.abs(decode(mask[0][1]).astype(int) - base).sum() > 1000


def test_held_out_times_are_refused(lab):
    t_split = lab.session.t_split
    assert t_split < 9.9
    with pytest.raises(LabError, match="training window"):
        render_frames(lab, [t_split + 1.0], "none")
