"""The whole flow through the HTTP API alone, as `curl` would drive it, with jobs running to
completion inline on fake pipeline stages (Part 21's "Done when")."""

import os

import pytest

from ghostline.jobs.errors import StageError
from ghostline.jobs.runner import run_job
from ghostline.jobs.spawn import InlineSpawner
from ghostline.store.volume import NullVolume

CHUNK = 1024


def upload_clip(client, exp_id, role="fit", size=2600):
    data = os.urandom(size)
    upload = client.post(
        "/api/uploads",
        json={"experiment_id": exp_id, "role": role, "filename": "IMG_1.MOV", "size": size},
    ).json()
    for n in range(upload["n_chunks"]):
        piece = data[n * CHUNK : (n + 1) * CHUNK]
        assert (
            client.put(f"/api/uploads/{upload['upload_id']}/chunks/{n}", content=piece).status_code
            == 200
        )
    done = client.post(f"/api/uploads/{upload['upload_id']}/complete")
    assert done.status_code == 200, done.text
    return done.json()


def test_the_full_flow(inline_client, paths):
    client = inline_client

    # create
    exp = client.post(
        "/api/experiments",
        json={
            "title": "Pendulum at 40 degrees",
            "system_hint": "pendulum",
            "setup": {"plane_offset_cm": 2.5, "pendulum_length_cm": 60},
        },
    ).json()
    eid = exp["id"]

    # upload -> assemble -> ingest runs
    first = upload_clip(client, eid)
    assert first["job"]["kind"] == "ingest_trial" and first["job"]["status"] == "succeeded"
    state = client.get(f"/api/experiments/{eid}").json()
    assert state["status"] == "ready"
    [trial] = state["trials"]
    assert trial["id"] == first["trial_id"] and trial["status"] == "ready"
    assert trial["n_frames"] == 600 and trial["trim"]["source"] == "auto"
    assert {"original", "master", "poster"} <= set(trial["paths"])

    # trim
    trimmed = client.post(f"/api/trials/{trial['id']}/trim", json={"start_s": 1.0, "end_s": 8.0})
    assert trimmed.json()["trim"]["start_s"] == 1.0

    # analyze
    job = client.post(f"/api/experiments/{eid}/analyze").json()
    assert job["status"] == "succeeded" and job["progress"] == 1.0
    state = client.get(f"/api/experiments/{eid}").json()
    assert state["status"] == "complete" and state["active_job"] is None
    assert state["scene"]["system_type"] == "pendulum"
    assert state["winner_model_id"] == "P1" and state["confidence"] == "high"
    assert state["conclusion"]["quantities"][0]["name"] == "g"
    assert state["review_status"] == "accepted"
    assert {m["id"] for m in state["models"]} == {"P0", "P1"}
    assert len(state["fits"]) == 1 and len(state["reviews"]) == 1
    assert state["tracks"][0]["object_key"] == "bob" and state["trajectories"]
    assert state["cost_usd"] == 0  # the passcode holder sees costs

    # the notebook tells the story, in order
    page = client.get(f"/api/experiments/{eid}/events").json()
    titles = [e["title"] for e in page["events"]]
    assert titles[0] == "Normalized the clip"
    assert "Wonder's favorite: P1" in titles
    assert [e["seq"] for e in page["events"]] == list(range(1, len(titles) + 1))
    assert page["status"] == "complete" and page["job"]["status"] == "succeeded"

    # prove: second clip -> ingest -> ghost
    second = upload_clip(client, eid, role="prove")
    assert second["job"]["status"] == "succeeded"
    proved = client.post(f"/api/experiments/{eid}/prove").json()
    assert proved["status"] == "succeeded" and proved["kind"] == "prove_trial"
    state = client.get(f"/api/experiments/{eid}").json()
    [ghost] = state["ghosts"]
    assert ghost["version"] == 1 and ghost["divergence_onset_s"] == 2.8
    assert state["status"] == "complete"
    video = client.get(f"/api/media/{eid}/{ghost['video_path']}", headers={"Range": "bytes=0-4"})
    assert video.status_code == 206 and video.content == b"ghost"

    # revise
    revised = client.post(f"/api/experiments/{eid}/revise").json()
    assert revised["status"] == "succeeded"
    later = client.get(f"/api/experiments/{eid}/events", params={"after": page["last_seq"]}).json()
    assert any(e["kind"] == "ghost" for e in later["events"])  # written by the prove job
    assert any(e["kind"] == "revision" for e in later["events"])

    # simulate
    sim = client.post(f"/api/experiments/{eid}/simulate", json={"model_id": "P1", "t_end_s": 5})
    assert sim.status_code == 200
    assert sim.json() == {
        "model_id": "P1",
        "t": [0.0, 1.0],
        "series": {"theta": [0.1, -0.1]},
        "derived": {},
    }

    # export, and the check
    export = client.get(f"/api/experiments/{eid}/export", params={"mass_g": 120})
    assert export.status_code == 200 and export.json()["schema_version"] == "ghostline.sim/1"
    assert export.json()["mass_kg"] == pytest.approx(0.12)
    snippet = client.get(f"/api/experiments/{eid}/export", params={"format": "pybullet"})
    assert snippet.headers["content-type"].startswith("text/x-python")
    assert snippet.text == "import pybullet as p\n"
    assert "attachment" in snippet.headers["content-disposition"]
    check = client.post(f"/api/experiments/{eid}/export/verify").json()
    assert check["status"] == "succeeded"
    assert client.get(f"/api/experiments/{eid}").json()["replay_check"]["passed"] is True

    # re-run one fit
    rerun = client.post(f"/api/experiments/{eid}/models/P1/rerun", json={"split_pct": 0.5}).json()
    assert rerun["status"] == "succeeded"
    assert len(client.get(f"/api/experiments/{eid}").json()["fits"]) == 2

    # every job is on record
    kinds = [j["kind"] for j in client.get(f"/api/experiments/{eid}").json()["jobs"]]
    assert sorted(kinds) == sorted(
        [
            "ingest_trial",
            "ingest_trial",
            "analyze_experiment",
            "prove_trial",
            "revise",
            "export_verify",
            "rerun",
        ]
    )


def test_a_second_analysis_appends_to_the_notebook_instead_of_rewriting_it(inline_client):
    client = inline_client
    eid = client.post("/api/experiments", json={"title": "x"}).json()["id"]
    upload_clip(client, eid)
    client.post(f"/api/experiments/{eid}/analyze")
    before = client.get(f"/api/experiments/{eid}/events").json()

    client.post(f"/api/experiments/{eid}/analyze")
    after = client.get(f"/api/experiments/{eid}/events").json()

    assert after["events"][: len(before["events"])] == before["events"]
    assert len(after["events"]) > len(before["events"])


def test_a_stage_failure_reaches_the_user_with_a_retry_that_works(inline_client, fake_stages):
    client = inline_client
    eid = client.post("/api/experiments", json={"title": "x"}).json()["id"]
    upload_clip(client, eid)
    fake_stages.fail_on["track_objects"] = RuntimeError("SAM returned nothing")

    failed = client.post(f"/api/experiments/{eid}/analyze").json()

    assert failed["status"] == "failed" and "SAM returned nothing" in failed["error"]
    state = client.get(f"/api/experiments/{eid}").json()
    assert state["status"] == "failed" and "SAM returned nothing" in state["error"]
    assert state["active_job"] is None

    fake_stages.fail_on.clear()
    retried = client.post(f"/api/jobs/{failed['id']}/retry").json()

    assert retried["status"] == "succeeded"
    assert client.get(f"/api/experiments/{eid}").json()["status"] == "complete"


def test_a_user_error_has_no_retry(inline_client, fake_stages):
    client = inline_client
    eid = client.post("/api/experiments", json={"title": "x"}).json()["id"]
    first = upload_clip(client, eid)
    fake_stages.fail_on["analyze_scene"] = StageError(
        "The pendulum leaves the frame.", code="cannot_measure", retryable=False
    )

    failed = client.post(f"/api/experiments/{eid}/analyze").json()

    assert (failed["error_code"], failed["retryable"]) == ("cannot_measure", False)
    assert client.post(f"/api/jobs/{failed['id']}/retry").status_code == 409
    assert first["trial_id"]


def test_a_failed_ingest_can_be_retried_from_the_stored_original(inline_client, fake_stages, repo):
    client = inline_client
    eid = client.post("/api/experiments", json={"title": "x"}).json()["id"]
    fake_stages.fail_on["ingest"] = RuntimeError("ffprobe crashed")

    bad = upload_clip(client, eid)

    assert bad["job"]["status"] == "failed"
    assert client.get(f"/api/experiments/{eid}").json()["status"] == "draft"
    assert repo.get_trial(bad["trial_id"]).status == "failed"
    fake_stages.fail_on.clear()
    good = client.post(f"/api/jobs/{bad['job_id']}/retry").json()
    assert good["status"] == "succeeded"
    assert repo.get_trial(bad["trial_id"]).status == "ready"
    assert client.get(f"/api/experiments/{eid}").json()["status"] == "ready"


# --- unwired stages (before wave 2) -----------------------------------------------------------
@pytest.fixture
def unwired_client(make_client, repo, paths, settings):
    def run(job_id):
        run_job(job_id, repo=repo, paths=paths, volume=NullVolume(), settings=settings)

    return make_client(InlineSpawner(run))


def test_jobs_report_that_the_stages_are_not_wired_yet(unwired_client):
    eid = unwired_client.post("/api/experiments", json={"title": "x"}).json()["id"]

    upload = upload_clip(unwired_client, eid)

    assert upload["job"]["status"] == "failed"
    assert upload["job"]["error_code"] == "not_implemented"
    assert "wired in wave 2" in upload["job"]["error"]


def test_export_and_simulate_say_501_until_they_are_wired(unwired_client, repo):
    from ghostline.store.records import ModelRecord

    eid = unwired_client.post("/api/experiments", json={"title": "x"}).json()["id"]
    repo.update_experiment(
        eid,
        status="complete",
        conclusion={
            "winner_id": "P1",
            "quantities": [],
            "rejected": [],
            "confidence": "low",
            "caveats": [],
            "narrative_md": "",
        },
    )
    repo.upsert_model(ModelRecord(experiment_id=eid, id="P1", name="n"))

    export = unwired_client.get(f"/api/experiments/{eid}/export")
    simulate = unwired_client.post(f"/api/experiments/{eid}/simulate", json={"model_id": "P1"})

    assert (export.status_code, export.json()["code"]) == (501, "not_implemented")
    assert (simulate.status_code, simulate.json()["code"]) == (501, "not_implemented")
    assert "wired in wave 2" in export.json()["detail"]


# --- simulate and export details --------------------------------------------------------------
def test_simulate_checks_the_model_and_times_out(make_client, spawner, repo, fake_stages):
    from ghostline.store.records import ModelRecord

    client = make_client(spawner, simulate_timeout_s=0.1)
    eid = client.post("/api/experiments", json={"title": "x"}).json()["id"]
    repo.upsert_model(ModelRecord(experiment_id=eid, id="W1", name="Wonder model", source="wonder"))
    url = f"/api/experiments/{eid}/simulate"

    assert client.post(url, json={"model_id": "nope"}).status_code == 404
    assert client.post(url, json={"model_id": "W1", "t_end_s": 0}).status_code == 422
    assert client.post(url, json={"model_id": "W1", "t_end_s": 500}).status_code == 422
    assert client.post(url, json={"model_id": "W1", "n_points": 1}).status_code == 422
    assert client.post(url, json={"model_id": "W1"}).status_code == 200

    fake_stages.simulate_delay_s = 0.5
    slow = client.post(url, json={"model_id": "W1"})
    assert (slow.status_code, slow.json()["code"]) == (504, "simulate_timeout")

    fake_stages.fail_on["simulate_custom_model"] = StageError(
        "Model code failed the AST check.", code="bad_model"
    )
    fake_stages.simulate_delay_s = 0
    bad = client.post(url, json={"model_id": "W1"})
    assert (bad.status_code, bad.json()["code"]) == (422, "bad_model")


def test_export_needs_a_result_and_valid_parameters(client, repo, fake_stages):
    eid = client.post("/api/experiments", json={"title": "x"}).json()["id"]

    early = client.get(f"/api/experiments/{eid}/export")
    repo.update_experiment(
        eid,
        status="complete",
        conclusion={
            "winner_id": "P1",
            "quantities": [],
            "rejected": [],
            "confidence": "low",
            "caveats": [],
            "narrative_md": "",
        },
    )

    assert (early.status_code, early.json()["code"]) == (409, "not_analyzed")
    assert (
        client.get(f"/api/experiments/{eid}/export", params={"format": "mujoco"}).status_code == 422
    )
    assert client.get(f"/api/experiments/{eid}/export", params={"mass_g": -3}).status_code == 422
    ok = client.get(f"/api/experiments/{eid}/export")
    assert ok.status_code == 200 and f"ghostline-{eid}.json" in ok.headers["content-disposition"]
