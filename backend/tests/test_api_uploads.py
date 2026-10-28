import hashlib
import os
import random
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from ghostline.store.records import Trim

CHUNK = 1024  # the settings fixture uses 1 KiB chunks so tests stay small


def make_experiment(client, **overrides):
    return client.post("/api/experiments", json={"title": "clip", **overrides}).json()["id"]


def start(client, exp_id, data: bytes, *, role="fit", filename="IMG_0001.MOV"):
    response = client.post(
        "/api/uploads",
        json={"experiment_id": exp_id, "role": role, "filename": filename, "size": len(data)},
    )
    assert response.status_code == 201, response.text
    return response.json()


def put(client, upload, data: bytes, n: int, body: bytes | None = None):
    piece = data[n * CHUNK : (n + 1) * CHUNK] if body is None else body
    return client.put(f"/api/uploads/{upload['upload_id']}/chunks/{n}", content=piece)


def upload_all(client, upload, data: bytes):
    for n in range(upload["n_chunks"]):
        assert put(client, upload, data, n).status_code == 200


# --- starting -------------------------------------------------------------------------------
def test_start_returns_the_chunk_plan(client):
    exp = make_experiment(client)

    upload = start(client, exp, os.urandom(2500))

    assert upload["chunk_size"] == CHUNK and upload["n_chunks"] == 3
    assert upload["size"] == 2500 and upload["received_chunks"] == [] and upload["status"] == "open"
    assert upload["trial_id"] is None and upload["role"] == "fit"


@pytest.mark.parametrize("size", [0, -5])
def test_start_rejects_empty_sizes(client, size):
    exp = make_experiment(client)

    response = client.post(
        "/api/uploads",
        json={"experiment_id": exp, "role": "fit", "filename": "a.mov", "size": size},
    )

    assert response.status_code == 422


def test_start_rejects_files_over_the_limit(make_client, spawner):
    client = make_client(spawner, max_upload_bytes=5000)
    exp = make_experiment(client)

    response = client.post(
        "/api/uploads",
        json={"experiment_id": exp, "role": "fit", "filename": "big.mov", "size": 5001},
    )

    assert response.status_code == 413 and response.json()["code"] == "file_too_large"
    assert (
        client.post(
            "/api/uploads",
            json={"experiment_id": exp, "role": "fit", "filename": "ok.mov", "size": 5000},
        ).status_code
        == 201
    )


def test_the_default_limit_is_one_gib(settings):
    assert settings.max_upload_bytes == 1024**3


def test_start_needs_a_real_experiment_and_a_sensible_role(client):
    body = {"role": "fit", "filename": "a.mov", "size": 10}

    assert (
        client.post("/api/uploads", json={**body, "experiment_id": str(uuid4())}).status_code == 404
    )
    exp = make_experiment(client)
    assert (
        client.post("/api/uploads", json={**body, "experiment_id": exp, "role": "x"}).status_code
        == 422
    )
    # A proof clip only makes sense after an analysis.
    early = client.post("/api/uploads", json={**body, "experiment_id": exp, "role": "prove"})
    assert (early.status_code, early.json()["code"]) == (409, "not_analyzed")


def test_a_finished_analysis_takes_proof_clips_but_no_new_first_clip(client, repo):
    exp = make_experiment(client)
    repo.update_experiment(exp, status="complete")
    body = {"experiment_id": exp, "filename": "a.mov", "size": 10}

    assert client.post("/api/uploads", json={**body, "role": "prove"}).status_code == 201
    locked = client.post("/api/uploads", json={**body, "role": "fit"})
    assert (locked.status_code, locked.json()["code"]) == (409, "experiment_locked")


def test_an_experiment_cannot_collect_unlimited_uploads(make_client, spawner):
    client = make_client(spawner, max_open_uploads_per_experiment=2)
    exp = make_experiment(client)
    data = os.urandom(10)

    start(client, exp, data)
    start(client, exp, data)

    response = client.post(
        "/api/uploads", json={"experiment_id": exp, "role": "fit", "filename": "a.mov", "size": 10}
    )
    assert (response.status_code, response.json()["code"]) == (429, "too_many_uploads")


# --- chunks ---------------------------------------------------------------------------------
def test_chunks_are_stored_and_counted(client, paths):
    exp = make_experiment(client)
    data = os.urandom(2500)
    upload = start(client, exp, data)

    first = put(client, upload, data, 0)
    last = put(client, upload, data, 2)

    assert first.json() == {
        "upload_id": upload["upload_id"],
        "n": 0,
        "size": CHUNK,
        "received_count": 1,
        "n_chunks": 3,
    }
    assert last.json()["size"] == 2500 - 2 * CHUNK and last.json()["received_count"] == 2
    assert paths.upload_part(upload["upload_id"], 0).read_bytes() == data[:CHUNK]
    assert client.get(f"/api/uploads/{upload['upload_id']}").json()["received_chunks"] == [0, 2]


def test_repeating_a_chunk_is_harmless(client, paths):
    exp = make_experiment(client)
    data = os.urandom(2000)
    upload = start(client, exp, data)

    for _ in range(3):
        response = put(client, upload, data, 1)
        assert response.status_code == 200 and response.json()["received_count"] == 1

    assert paths.upload_part(upload["upload_id"], 1).read_bytes() == data[CHUNK:]


def test_a_chunk_may_be_replaced_with_corrected_bytes(client, paths):
    exp = make_experiment(client)
    data = os.urandom(CHUNK)
    upload = start(client, exp, data)

    put(client, upload, data, 0, body=b"\x00" * CHUNK)
    put(client, upload, data, 0)

    assert paths.upload_part(upload["upload_id"], 0).read_bytes() == data


def test_chunks_with_the_wrong_size_or_number_are_refused(client, paths):
    exp = make_experiment(client)
    data = os.urandom(2500)
    upload = start(client, exp, data)

    short = put(client, upload, data, 0, body=data[: CHUNK - 1])
    long = put(client, upload, data, 0, body=data[: CHUNK + 1])
    wrong_last = put(client, upload, data, 2, body=data[:CHUNK])
    bad_number = put(client, upload, data, 3)
    negative = client.put(f"/api/uploads/{upload['upload_id']}/chunks/-1", content=b"x")
    empty = put(client, upload, data, 0, body=b"")

    assert (short.status_code, short.json()["code"]) == (400, "bad_chunk_size")
    assert (long.status_code, long.json()["code"]) == (413, "chunk_too_large")
    assert wrong_last.status_code == 413
    assert (bad_number.status_code, bad_number.json()["code"]) == (400, "bad_chunk_number")
    assert negative.status_code == 400
    assert empty.status_code == 400
    assert not paths.upload_dir(upload["upload_id"]).exists()  # nothing written
    assert client.get(f"/api/uploads/{upload['upload_id']}").json()["received_chunks"] == []


def test_the_default_chunk_size_is_eight_mib(settings):
    from ghostline.config import MIB

    assert MIB * 8 == 8 * 1024 * 1024


def test_unknown_upload(client):
    missing = str(uuid4())

    assert client.put(f"/api/uploads/{missing}/chunks/0", content=b"x").status_code == 404
    assert client.get(f"/api/uploads/{missing}").status_code == 404
    assert client.post(f"/api/uploads/{missing}/complete").status_code == 404


def test_parallel_chunks_from_a_shuffled_client_all_land(client, repo):
    exp = make_experiment(client)
    data = os.urandom(CHUNK * 12 + 77)
    upload = start(client, exp, data)
    order = list(range(upload["n_chunks"]))
    random.Random(7).shuffle(order)

    with ThreadPoolExecutor(max_workers=3) as pool:  # three in parallel, like the browser
        codes = list(pool.map(lambda n: put(client, upload, data, n).status_code, order))

    assert set(codes) == {200}
    assert client.get(f"/api/uploads/{upload['upload_id']}").json()["received_chunks"] == list(
        range(13)
    )


# --- completing -----------------------------------------------------------------------------
def test_complete_assembles_the_file_creates_the_trial_and_spawns_ingest(
    client, spawner, paths, repo
):
    exp = make_experiment(client)
    data = os.urandom(2500)
    upload = start(client, exp, data, filename="IMG_0001.MOV")
    upload_all(client, upload, data)

    response = client.post(f"/api/uploads/{upload['upload_id']}/complete")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sha256"] == hashlib.sha256(data).hexdigest()
    assert body["size"] == 2500
    trial = repo.get_trial(body["trial_id"])
    assert (trial.role, trial.status, trial.sha256) == ("fit", "uploaded", body["sha256"])
    assert trial.original_filename == "IMG_0001.MOV"
    original = paths.experiment_dir(exp) / trial.paths["original"]
    assert original.name == "original.mov"
    assert original.read_bytes() == data
    assert not paths.upload_dir(upload["upload_id"]).exists()  # chunks are cleaned up
    assert body["job"]["kind"] == "ingest_trial" and body["job"]["status"] == "queued"
    assert body["job"]["params"] == {"trial_id": str(trial.id)}
    assert [j.id for j in spawner.jobs] == [repo.get_job(body["job_id"]).id]
    assert repo.get_upload(upload["upload_id"]).status == "complete"
    assert client.get(f"/api/uploads/{upload['upload_id']}").json()["trial_id"] == body["trial_id"]


def test_complete_never_uses_the_client_filename_as_a_path(client, repo, paths):
    exp = make_experiment(client)
    data = os.urandom(100)
    upload = start(client, exp, data, filename="../../../etc/cron.d/evil.sh")
    upload_all(client, upload, data)

    body = client.post(f"/api/uploads/{upload['upload_id']}/complete").json()

    trial = repo.get_trial(body["trial_id"])
    assert trial.paths["original"] == f"trials/{trial.id}/original.bin"
    assert (paths.experiment_dir(exp) / trial.paths["original"]).is_file()
    assert not (paths.root.parent / "etc").exists()


def test_complete_lists_the_chunks_that_are_still_missing(client, spawner, repo):
    exp = make_experiment(client)
    data = os.urandom(CHUNK * 4)
    upload = start(client, exp, data)
    put(client, upload, data, 0)
    put(client, upload, data, 3)

    response = client.post(f"/api/uploads/{upload['upload_id']}/complete")

    assert response.status_code == 409
    assert response.json()["code"] == "chunks_missing" and response.json()["missing"] == [1, 2]
    assert spawner.jobs == [] and repo.list_trials(exp) == []
    # Resuming: send what is missing, then complete.
    put(client, upload, data, 1)
    put(client, upload, data, 2)
    assert client.post(f"/api/uploads/{upload['upload_id']}/complete").status_code == 200


def test_complete_notices_a_chunk_file_that_disappeared(client, paths):
    """The database says a chunk arrived, but this container can't see the file (a Volume that
    hasn't synced). The client is asked to send it again."""
    exp = make_experiment(client)
    data = os.urandom(CHUNK * 2)
    upload = start(client, exp, data)
    upload_all(client, upload, data)
    paths.upload_part(upload["upload_id"], 1).unlink()

    response = client.post(f"/api/uploads/{upload['upload_id']}/complete")

    assert (response.status_code, response.json()["missing"]) == (409, [1])


def test_completing_twice_returns_the_same_trial_and_job(client, spawner, repo):
    exp = make_experiment(client)
    data = os.urandom(700)
    upload = start(client, exp, data)
    upload_all(client, upload, data)

    first = client.post(f"/api/uploads/{upload['upload_id']}/complete").json()
    second = client.post(f"/api/uploads/{upload['upload_id']}/complete")

    assert second.status_code == 200
    assert (
        second.json()["trial_id"] == first["trial_id"]
        and second.json()["job_id"] == first["job_id"]
    )
    assert second.json()["sha256"] == first["sha256"]
    assert len(spawner.jobs) == 1 and len(repo.list_trials(exp)) == 1


def test_concurrent_completes_make_one_trial(client, spawner, repo):
    exp = make_experiment(client)
    data = os.urandom(CHUNK * 3)
    upload = start(client, exp, data)
    upload_all(client, upload, data)
    url = f"/api/uploads/{upload['upload_id']}/complete"

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post(url), range(4)))

    ok = [r for r in responses if r.status_code == 200]
    assert len(repo.list_trials(exp)) == 1
    assert len(spawner.jobs) == 1
    assert {r.json()["trial_id"] for r in ok} <= {str(repo.list_trials(exp)[0].id)}
    assert all(r.status_code in (200, 409) for r in responses)


def test_complete_is_refused_while_another_job_runs(client, repo, paths, spawner):
    exp = make_experiment(client)
    data = os.urandom(300)
    upload = start(client, exp, data)
    upload_all(client, upload, data)
    busy = repo.create_job(experiment_id=exp, kind="analyze_experiment")

    response = client.post(f"/api/uploads/{upload['upload_id']}/complete")

    assert (response.status_code, response.json()["code"]) == (409, "job_active")
    assert response.json()["job_id"] == str(busy.id)
    assert repo.list_trials(exp) == [] and spawner.jobs == []
    assert not paths.experiment_dir(exp).joinpath("trials").exists()  # no orphaned assembly


def test_a_failed_spawn_still_returns_the_trial_with_a_retryable_failed_job(client, spawner, repo):
    exp = make_experiment(client)
    data = os.urandom(300)
    upload = start(client, exp, data)
    upload_all(client, upload, data)
    spawner.fail_with = RuntimeError("modal unavailable")

    response = client.post(f"/api/uploads/{upload['upload_id']}/complete")

    assert response.status_code == 200
    job = response.json()["job"]
    assert (job["status"], job["error_code"], job["retryable"]) == ("failed", "spawn_failed", True)
    trial = repo.get_trial(response.json()["trial_id"])
    assert (trial.status, trial.error) == ("failed", "Could not start the job: modal unavailable")


def test_proof_clips_make_prove_trials(client, repo):
    exp = make_experiment(client)
    repo.update_experiment(exp, status="complete")
    data = os.urandom(300)
    upload = start(client, exp, data, role="prove")
    upload_all(client, upload, data)

    body = client.post(f"/api/uploads/{upload['upload_id']}/complete").json()

    assert repo.get_trial(body["trial_id"]).role == "prove"
    assert repo.get_experiment(exp).status == "complete"  # proving doesn't reopen the analysis


def test_a_whole_gib_flows_through_in_big_chunks_without_holding_it_in_memory(make_client, spawner):
    """Not a real GiB: the assembler streams, so a few megabytes exercise the same path."""
    client = make_client(spawner, upload_chunk_bytes=1024 * 1024)
    exp = make_experiment(client)
    data = os.urandom(3 * 1024 * 1024 + 5)
    upload = client.post(
        "/api/uploads",
        json={"experiment_id": exp, "role": "fit", "filename": "a.mp4", "size": len(data)},
    ).json()
    for n in range(upload["n_chunks"]):
        piece = data[n * 1024 * 1024 : (n + 1) * 1024 * 1024]
        assert (
            client.put(f"/api/uploads/{upload['upload_id']}/chunks/{n}", content=piece).status_code
            == 200
        )

    body = client.post(f"/api/uploads/{upload['upload_id']}/complete").json()

    assert body["sha256"] == hashlib.sha256(data).hexdigest() and upload["n_chunks"] == 4


def test_chunk_endpoints_after_completion_are_refused(client):
    exp = make_experiment(client)
    data = os.urandom(300)
    upload = start(client, exp, data)
    upload_all(client, upload, data)
    client.post(f"/api/uploads/{upload['upload_id']}/complete")

    late = put(client, upload, data, 0)

    assert (late.status_code, late.json()["code"]) == (409, "upload_complete")


# --- trim -----------------------------------------------------------------------------------
def ready_trial(repo, exp, **fields):
    trial = repo.create_trial(experiment_id=exp, role="fit")
    repo.update_trial(
        trial.id,
        status="ready",
        n_frames=600,
        duration_s=10.0,
        trim=Trim(start_s=0.5, end_s=9.5, suggested_start_s=0.5, suggested_end_s=9.5),
        **fields,
    )
    return trial


def test_trim_sets_the_window_and_remembers_the_suggestion(client, repo):
    trial = ready_trial(repo, make_experiment(client))

    response = client.post(f"/api/trials/{trial.id}/trim", json={"start_s": 2.0, "end_s": 7.5})

    assert response.status_code == 200
    assert response.json()["trim"] == {
        "start_s": 2.0,
        "end_s": 7.5,
        "suggested_start_s": 0.5,
        "suggested_end_s": 9.5,
        "source": "user",
    }
    assert repo.get_trial(trial.id).trim.source == "user"


def test_trim_defaults_to_the_suggestion(client, repo):
    trial = ready_trial(repo, make_experiment(client))
    client.post(f"/api/trials/{trial.id}/trim", json={"start_s": 3.0, "end_s": 4.0})

    reset = client.post(f"/api/trials/{trial.id}/trim", json={})
    only_start = client.post(f"/api/trials/{trial.id}/trim", json={"start_s": 1.0})
    no_body = client.post(f"/api/trials/{trial.id}/trim")

    assert (reset.json()["trim"]["start_s"], reset.json()["trim"]["end_s"]) == (0.5, 9.5)
    assert (only_start.json()["trim"]["start_s"], only_start.json()["trim"]["end_s"]) == (1.0, 9.5)
    assert no_body.status_code == 200


@pytest.mark.parametrize(
    "body, code",
    [
        ({"start_s": 5, "end_s": 5}, "bad_trim"),
        ({"start_s": 6, "end_s": 2}, "bad_trim"),
        ({"start_s": 1, "end_s": 12}, "bad_trim"),
        ({"start_s": 9.9}, "bad_trim"),
    ],
)
def test_trim_rejects_impossible_windows(client, repo, body, code):
    trial = ready_trial(repo, make_experiment(client))

    response = client.post(f"/api/trials/{trial.id}/trim", json=body)

    assert (response.status_code, response.json()["code"]) == (400, code)
    assert repo.get_trial(trial.id).trim.source == "auto"


def test_trim_validates_types(client, repo):
    trial = ready_trial(repo, make_experiment(client))

    assert client.post(f"/api/trials/{trial.id}/trim", json={"start_s": -1}).status_code == 422
    assert client.post(f"/api/trials/{trial.id}/trim", json={"end_s": 0}).status_code == 422
    assert client.post(f"/api/trials/{trial.id}/trim", json={"start_s": "soon"}).status_code == 422


def test_trim_enforces_the_frame_limit(make_client, spawner, repo):
    client = make_client(spawner, max_frames=300)
    trial = ready_trial(repo, make_experiment(client))  # 600 frames over 10 s

    too_long = client.post(f"/api/trials/{trial.id}/trim", json={"start_s": 0, "end_s": 6})
    fits = client.post(f"/api/trials/{trial.id}/trim", json={"start_s": 0, "end_s": 5})

    assert (too_long.status_code, too_long.json()["code"]) == (400, "too_many_frames")
    assert fits.status_code == 200


def test_trim_needs_a_ready_trial(client, repo):
    exp = make_experiment(client)
    waiting = repo.create_trial(experiment_id=exp, role="fit")

    not_ready = client.post(f"/api/trials/{waiting.id}/trim", json={"start_s": 0, "end_s": 1})
    unknown = client.post(f"/api/trials/{uuid4()}/trim", json={"start_s": 0, "end_s": 1})

    assert (not_ready.status_code, not_ready.json()["code"]) == (409, "trial_not_ready")
    assert unknown.status_code == 404
