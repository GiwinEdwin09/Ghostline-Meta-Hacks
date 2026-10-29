"""predict_ghost end to end on a synthetic pendulum experiment (build plan Part 16)."""

import json

import av
import numpy as np
import pytest
from fixtures.ghost.rig import pendulum_rig

from ghostline.ghost.stage import predict_ghost
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import GhostResult

# Trial 1: 8 s. Trial 2: 6 s at 60 fps, about four swings of a 0.6 m pendulum.
D1, D2 = 8.0, 6.0


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One experiment, one predict_ghost call, shared by the tests below."""
    import dataclasses

    from ghostline.config import get_settings
    from ghostline.store.volume import VolumePaths

    root = tmp_path_factory.mktemp("ghost-stage")
    paths = VolumePaths(root / "data")
    settings = dataclasses.replace(get_settings(), volume_root=str(root / "data"))
    rig = pendulum_rig(paths, settings, d1=D1, d2=D2)
    result = predict_ghost(
        rig.ctx, rig.experiment, rig.fit_trial, rig.prove_trial, rig.scene, rig.trajectory
    )
    return rig, result


def test_it_returns_the_paths_and_numbers_the_handler_stores(run):
    rig, result = run
    assert isinstance(result, GhostResult)
    assert result.version == 1 and result.model_id == "P1"
    assert result.video_path == "ghost/v1.mp4" and result.track_path == "ghost/v1_track.json"
    assert result.slow_video_path is None  # a 60 fps clip is not slow motion
    assert result.divergence_onset_s is None  # the ghost stayed with the pendulum
    assert 0 < result.max_error_mm < 5.0
    root = rig.paths.experiment_dir(rig.experiment.id)
    assert (root / result.video_path).stat().st_size > 10_000
    assert (root / result.track_path).exists()
    assert (root / "trials" / str(rig.prove_trial.id) / "sprite.png").exists()


def test_it_reports_the_phases_in_order_and_writes_a_notebook_event(run):
    rig, _ = run
    assert rig.repo.phases == ["predicting", "rendering"]
    (event,) = [e for e in rig.repo.events if e["kind"] == "prediction"]
    assert event["actor"] == "harness"
    data = event["data"]
    assert data["model_id"] == "P1" and data["n_draws"] == 200
    assert data["frozen"]["g"] == pytest.approx(9.81, rel=0.01)
    assert set(data["initial_conditions"]) == {"theta0", "omega0"}
    assert "pure prediction" in event["body_md"]
    assert rig.repo.progress and max(rig.repo.progress) == pytest.approx(1.0, abs=0.5)


def test_the_track_json_matches_the_web_ghosttrack_type(run):
    rig, result = run
    root = rig.paths.experiment_dir(rig.experiment.id)
    track = json.loads((root / result.track_path).read_text())
    # the fields web/src/lib/types.ts GhostTrack declares
    n = len(track["t"])
    for key in ("error_mm", "predicted_px", "actual_px"):
        assert len(track[key]) == n
    assert all(len(p) == 2 for p in track["predicted_px"] + track["actual_px"])
    assert track["tau_mm"] > 0 and track["onset_s"] is None
    # time starts at 0, at the first frame of the motion, and the launch window starts there too
    assert track["t"][0] == 0.0 and 0.25 <= track["launch_window_s"] <= 1.0
    assert track["t"][-1] == pytest.approx(D2 - 0.5, abs=0.5)
    assert np.all(np.diff(track["t"]) > 0)
    # extras
    assert track["version"] == 1 and track["model_id"] == "P1"
    assert track["predicted_ahead_s"] == pytest.approx(track["t"][-1] - track["launch_window_s"])
    assert track["max_error_mm"] == pytest.approx(result.max_error_mm, abs=1e-3)
    assert set(track["status"]) == {"green"} and len(track["spread_mm"]) == n


def test_the_video_has_the_real_timestamps_and_faststart(run):
    rig, result = run
    root = rig.paths.experiment_dir(rig.experiment.id)
    track = json.loads((root / result.track_path).read_text())
    video = root / result.video_path
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        times = [float(f.pts * stream.time_base) for f in container.decode(stream)]
        assert stream.codec_context.name == "h264"
    assert len(times) == len(track["t"])
    assert np.allclose(times, track["t"], atol=2e-5)  # the master's timestamps, from 0
    raw = video.read_bytes()
    assert raw.index(b"moov") < raw.index(b"mdat")  # faststart: playable before it downloads


def test_a_wrong_winner_is_a_stage_error_with_a_readable_message(tmp_path):
    import dataclasses

    from ghostline.config import get_settings
    from ghostline.store.volume import VolumePaths

    paths = VolumePaths(tmp_path / "data")
    settings = dataclasses.replace(get_settings(), volume_root=str(tmp_path / "data"))
    rig = pendulum_rig(paths, settings, d1=3.0, d2=3.0, video=False)
    rig.experiment = rig.experiment.model_copy(update={"winner_model_id": None})
    with pytest.raises(StageError, match="no winning model"):
        predict_ghost(
            rig.ctx, rig.experiment, rig.fit_trial, rig.prove_trial, rig.scene, rig.trajectory
        )
    rig.scene = rig.scene.model_copy(update={"system_type": "other"})
    rig.experiment = rig.experiment.model_copy(update={"winner_model_id": "P1"})
    with pytest.raises(StageError, match="can't draw a ghost"):
        predict_ghost(
            rig.ctx, rig.experiment, rig.fit_trial, rig.prove_trial, rig.scene, rig.trajectory
        )
