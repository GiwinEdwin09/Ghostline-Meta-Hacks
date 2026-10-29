"""The measurement pipeline end to end, through the HTTP API (build plan Part 21's done-when, with
the stages after the trajectory faked): create, chunked upload, complete, ingest, analyze (scene
analysis, SAM tracking, calibration, trajectory), then the trajectory JSON route.

A synthetic free-fall clip from synth/ goes through the real ffmpeg, PyAV, OpenCV and calibration
code. Muse Wonder and SAM answer from replay fixtures in fixtures/cache_pipeline. The physicist
and the reviewer are fakes.

The fixtures were recorded live from this same flow and match requests without looking at media
bytes, because a re-encoded master differs between ffmpeg builds (see `loose_requests`). To
record them again (about $0.2), with the key in the environment:

    GHOSTLINE_RECORD=1 uv run --env-file .env pytest backend/tests/test_integration_pipeline.py
"""

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from synth_helpers import HAVE_FFMPEG, clip

from ghostline.jobs import stages
from ghostline.jobs.runner import run_job
from ghostline.jobs.spawn import InlineSpawner
from ghostline.llm.cache import normalize_request as strict_normalize
from ghostline.store.volume import NullVolume

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg and ffprobe")

CACHE = Path(__file__).parent / "fixtures" / "cache_pipeline"
RECORD = os.environ.get("GHOSTLINE_RECORD") == "1"
CLIP = {
    "system": "free_fall",
    "size": (854, 480),
    "duration_s": 1.2,
    "capture_fps": 120.0,
    "params": {"y0": 0.3, "y_floor": -0.25},  # a short drop keeps the sheet big enough for SAM
}


def loose_requests(value, file_hashes=None):
    """Normalize a request as the cache does, but treat media as interchangeable: the file a
    request refers to and the pixels it carries don't take part in the key."""

    def scrub(item):
        if isinstance(item, dict):
            return {
                k: ("<media>" if k == "file_id" or k == "image_url" else scrub(v))
                for k, v in item.items()
            }
        if isinstance(item, list):
            return [scrub(v) for v in item]
        return item

    return scrub(strict_normalize(value, file_hashes))


@pytest.fixture
def pipeline(monkeypatch, make_client, repo, paths, settings):
    monkeypatch.setattr("ghostline.llm.runtime.normalize_request", loose_requests)
    monkeypatch.setenv("LLM_CACHE_DIR", str(CACHE))
    if RECORD:
        pytest.importorskip("openai")
        key = os.environ.get("MODEL_API_KEY")
        if not key:
            pytest.fail("GHOSTLINE_RECORD=1 needs MODEL_API_KEY in the environment")
        live = dataclasses.replace(
            settings,
            llm_mode="record",
            model_api_key=key,
            meta_base_url=os.environ.get("META_BASE_URL", settings.meta_base_url),
            budget_usd_per_experiment=1.0,
        )
    else:
        live = settings

    def run(job_id):
        run_job(job_id, repo=repo, paths=paths, volume=NullVolume(), settings=live)

    client = make_client(InlineSpawner(run), upload_chunk_bytes=256 * 1024)
    return client


def upload(client, experiment_id, video: Path, role="fit"):
    data = video.read_bytes()
    created = client.post(
        "/api/uploads",
        json={
            "experiment_id": experiment_id,
            "role": role,
            "filename": "../../etc/IMG_0001.MOV",  # never used as a path
            "size": len(data),
        },
    )
    assert created.status_code == 201, created.text
    info = created.json()
    size = info["chunk_size"]
    for n in range(info["n_chunks"]):
        piece = data[n * size : (n + 1) * size]
        put = client.put(f"/api/uploads/{info['upload_id']}/chunks/{n}", content=piece)
        assert put.status_code == 200, put.text
    done = client.post(f"/api/uploads/{info['upload_id']}/complete")
    assert done.status_code == 200, done.text
    return done.json()


def test_a_clip_goes_from_upload_to_trajectory(pipeline, paths, monkeypatch):
    from conftest import FakeStages

    truth = clip(**CLIP)
    client = pipeline
    created = client.post(
        "/api/experiments",
        json={
            "title": "A dropped ball",
            "system_hint": "free_fall",
            "setup": {"plane_offset_cm": 2},
        },
    )
    assert created.status_code == 201
    eid = created.json()["id"]

    # The measurement stages are the real ones; only the physicist and the reviewer are fakes.
    fake = FakeStages()
    monkeypatch.setattr(stages, "run_physicist", fake.run_physicist)
    monkeypatch.setattr(stages, "run_review", fake.run_review)

    # --- upload and ingest ---------------------------------------------------------------
    first = upload(client, eid, truth.video_path)
    assert first["job"]["kind"] == "ingest_trial" and first["job"]["status"] == "succeeded", first
    state = client.get(f"/api/experiments/{eid}").json()
    assert state["status"] == "ready"
    [trial] = state["trials"]
    assert trial["status"] == "ready" and trial["n_frames"] == truth.n_frames
    assert trial["duration_s"] == pytest.approx(truth.n_frames / 120, abs=0.02)
    assert {"original", "master", "wonder_480", "playback_720", "poster", "timestamps"} <= set(
        trial["paths"]
    )
    assert "wonder_480" in trial["meta_file_ids"]
    assert trial["trim"]["source"] == "auto" and trial["trim"]["end_s"] > trial["trim"]["start_s"]
    assert not any(c in trial["paths"]["original"] for c in ("..", "MOV"))  # the name is ours
    timestamps = np.load(paths.experiment_dir(eid) / trial["paths"]["timestamps"])
    assert len(timestamps) == truth.n_frames and timestamps[0] == pytest.approx(0, abs=1e-6)

    # --- analyze -------------------------------------------------------------------------
    job = client.post(f"/api/experiments/{eid}/analyze")
    assert job.status_code == 202, job.text
    assert job.json()["status"] == "succeeded", job.json()
    state = client.get(f"/api/experiments/{eid}").json()
    assert state["status"] == "complete"
    assert state["scene"]["system_type"] in ("free_fall", "projectile")
    assert state["scene"]["proceed"] is True

    # tracking left a track per object and the sheet, and the raw SAM stream for the browser
    [moving] = [t for t in state["tracks"] if t["stats"].get("role") == "moving_body"]
    in_window = (trial["trim"]["end_s"] - trial["trim"]["start_s"]) * 120
    assert moving["stats"]["frames"] >= 0.85 * in_window
    sheet = next(t for t in state["tracks"] if t["object_key"] == "sheet")
    assert sheet["stats"]["area_px"] > 10_000
    exp_dir = paths.experiment_dir(eid)
    assert (exp_dir / moving["stats"]["events_path"]).is_file()
    tracks = pd.read_parquet(exp_dir / moving["parquet_path"])
    assert {"object_key", "frame", "t", "cx", "cy", "area", "conf", "flags"} <= set(tracks.columns)

    # calibration: millimetres per pixel in the plane of motion, within 1% of the truth
    [trial] = state["trials"]
    cal = trial["calibration"]
    assert cal["corner_method"] and cal["uncertainty"]["sigma_scale_rel"] < 0.02
    assert cal["paper"] == "A4" and cal["camera_motion"]["status"] == "ok"
    assert {"calibration", "calib_overlay", "rectified"} <= set(trial["paths"])
    assert (exp_dir / trial["paths"]["calib_overlay"]).is_file()

    # trajectory: metres, with the fall visible, and a parquet other stages can read
    [traj] = state["trajectories"]
    assert traj["summary"]["n_points"] > 0.7 * in_window
    frame = pd.read_parquet(exp_dir / traj["parquet_path"])
    assert list(frame.columns[:3]) == ["t", "x", "y"]
    assert frame["y"].max() - frame["y"].min() > 0.3  # the ball falls more than 30 cm
    # positions agree with the truth the clip was rendered from, to a centimetre
    true_xy = truth.track()["xyz_m"][frame["frame"].to_numpy(), :2]
    error = np.hypot(*(frame[["x", "y"]].to_numpy() - true_xy).T)
    assert np.sqrt(np.mean(error**2)) < 0.015, np.sqrt(np.mean(error**2))

    # the faked downstream stages ran after the real ones, and the notebook tells the story
    assert state["winner_model_id"] == "P1" and state["review_status"] == "accepted"
    page = client.get(f"/api/experiments/{eid}/events").json()
    titles = [e["title"] for e in page["events"]]
    assert titles[:1] == ["Prepared the clip"]
    for expected in (
        "What Wonder saw",
        "First guesses, before any measurement",
        "Calibrated the scale from the sheet",
        "Extracted the trajectory",
    ):
        assert expected in titles
    assert page["status"] == "complete"
    hypo = next(e for e in page["events"] if e["kind"] == "hypothesis")
    assert 2 <= len(hypo["data"]["hypotheses"]) <= 4
    assert fake.calls == ["run_physicist", "run_review"]
