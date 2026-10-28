from uuid import uuid4

import pytest

from ghostline.store.records import BenchmarkEntry, Trim


def create(client, **overrides):
    body = {"title": "Pendulum, 40 degrees", "system_hint": "pendulum", **overrides}
    return client.post("/api/experiments", json=body)


# --- create ---------------------------------------------------------------------------------
def test_create_returns_the_new_experiment_with_defaults(client):
    response = create(client)

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Pendulum, 40 degrees"
    assert body["system_type"] == "pendulum"
    assert body["status"] == "draft"
    assert body["visibility"] == "private" and body["is_example"] is False
    assert body["declared_capture"] == {"mode": "normal", "fps": None}
    assert body["setup"] == {"plane_offset_cm": 2.0, "paper": "a4", "pendulum_length_cm": None}


def test_create_keeps_the_declared_capture_and_setup(client):
    body = create(
        client,
        description="Filmed at 240 fps.",
        declared_capture={"mode": "slow_mo", "fps": 240},
        setup={"plane_offset_cm": 3.5, "paper": "letter", "pendulum_length_cm": 62.5},
    ).json()

    fetched = client.get(f"/api/experiments/{body['id']}").json()

    assert fetched["declared_capture"] == {"mode": "slow_mo", "fps": 240}
    assert fetched["setup"]["pendulum_length_cm"] == 62.5
    assert fetched["description"] == "Filmed at 240 fps."


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": ""},
        {"title": "   "},
        {"title": "x" * 121},
        {"system_hint": "teleportation"},
        {"declared_capture": {"mode": "slow_mo"}},
        {"declared_capture": {"mode": "normal", "fps": 0}},
        {"declared_capture": {"mode": "normal", "fps": 5000}},
        {"setup": {"plane_offset_cm": -1}},
        {"setup": {"plane_offset_cm": 51}},
        {"setup": {"paper": "tabloid"}},
        {"setup": {"pendulum_length_cm": 0}},
        {"description": "x" * 2001},
    ],
)
def test_create_rejects_invalid_input(client, overrides):
    assert create(client, **overrides).status_code == 422


def test_create_strips_control_characters_from_text(client):
    body = create(client, title="Drop\x00 test\x1b[31m", description="a\x00b\nc").json()

    assert body["title"] == "Drop test[31m"
    assert body["description"] == "ab\nc"


def test_create_ignores_fields_the_client_should_not_set(client):
    body = create(
        client, is_example=True, visibility="public", status="complete", cost_usd=99
    ).json()

    assert (body["is_example"], body["visibility"], body["status"]) == (False, "private", "draft")


# --- read -----------------------------------------------------------------------------------
def test_unknown_and_malformed_ids(client):
    missing = client.get(f"/api/experiments/{uuid4()}")

    assert missing.status_code == 404
    assert missing.json() == {
        "detail": "That experiment doesn't exist.",
        "code": "experiment_not_found",
    }
    assert client.get("/api/experiments/not-a-uuid").status_code == 422
    assert client.get(f"/api/experiments/{uuid4()}/events").status_code == 404


def test_the_state_carries_everything_the_workspace_needs(client, repo):
    exp = create(client).json()
    trial = repo.create_trial(experiment_id=exp["id"], role="fit")
    repo.update_trial(trial.id, status="ready", trim=Trim(start_s=1, end_s=4))

    state = client.get(f"/api/experiments/{exp['id']}").json()

    assert state["id"] == exp["id"]
    assert [t["id"] for t in state["trials"]] == [str(trial.id)]
    assert state["trials"][0]["trim"]["end_s"] == 4
    for key in ("tracks", "trajectories", "models", "fits", "reviews", "ghosts", "jobs"):
        assert state[key] == []
    assert state["active_job"] is None
    assert state["scene"] is None and state["conclusion"] is None


def test_anyone_with_the_id_can_read_but_costs_need_the_passcode(client, anon, repo):
    exp = create(client).json()
    repo.add_spend(amount_usd=0.42, provider="meta", experiment_id=exp["id"])

    public = anon.get(f"/api/experiments/{exp['id']}").json()
    admin = client.get(f"/api/experiments/{exp['id']}").json()

    assert public["cost_usd"] is None
    assert admin["cost_usd"] == pytest.approx(0.42)


# --- events (polling) -----------------------------------------------------------------------
def test_events_are_polled_with_after_and_carry_job_state(client, repo):
    exp = create(client).json()
    from ghostline.notebook import NotebookWriter

    writer = NotebookWriter(repo, exp["id"])
    for i in range(3):
        writer.append(actor="harness", kind="fit", title=f"fit {i}")

    first = client.get(f"/api/experiments/{exp['id']}/events").json()
    later = client.get(f"/api/experiments/{exp['id']}/events", params={"after": 2}).json()
    caught_up = client.get(f"/api/experiments/{exp['id']}/events", params={"after": 3}).json()

    assert [e["seq"] for e in first["events"]] == [1, 2, 3]
    assert first["last_seq"] == 3 and first["status"] == "draft" and first["job"] is None
    assert [e["title"] for e in later["events"]] == ["fit 2"]
    assert caught_up["events"] == [] and caught_up["last_seq"] == 3


def test_events_paging_and_bounds(client, repo):
    exp = create(client).json()
    from ghostline.notebook import NotebookWriter

    writer = NotebookWriter(repo, exp["id"])
    for i in range(5):
        writer.append(actor="system", kind="observation", title=str(i))

    page = client.get(f"/api/experiments/{exp['id']}/events", params={"limit": 2}).json()

    assert [e["seq"] for e in page["events"]] == [1, 2] and page["last_seq"] == 2
    assert (
        client.get(f"/api/experiments/{exp['id']}/events", params={"after": -1}).status_code == 422
    )
    assert (
        client.get(f"/api/experiments/{exp['id']}/events", params={"limit": 5000}).status_code
        == 422
    )


def test_events_page_reports_the_running_job(client, repo):
    exp = create(client).json()
    job = repo.create_job(experiment_id=exp["id"], kind="analyze_experiment")
    repo.claim_job(job.id)
    repo.set_job_phase(job.id, "tracking", 0.2)

    page = client.get(f"/api/experiments/{exp['id']}/events").json()

    assert page["job"]["status"] == "running"
    assert page["job"]["phase"] == "tracking"
    assert page["job"]["progress"] == pytest.approx(0.2)


# --- listing and publishing -----------------------------------------------------------------
def test_gallery_is_public_and_lists_only_published_examples(client, anon):
    private = create(client, title="mine").json()
    example = create(client, title="Spinner example").json()
    assert (
        client.patch(f"/api/experiments/{example['id']}", json={"is_example": True}).status_code
        == 200
    )

    listed = anon.get("/api/experiments", params={"examples": "true"})

    assert listed.status_code == 200
    assert [e["id"] for e in listed.json()] == [example["id"]]
    assert private["id"] not in listed.text
    assert listed.json()[0]["visibility"] == "public"


def test_listing_everything_needs_the_passcode(client, anon):
    create(client)

    assert anon.get("/api/experiments").status_code == 401
    assert len(client.get("/api/experiments").json()) == 1


def test_listing_hides_costs_from_the_public(client, anon, repo):
    exp = create(client).json()
    client.patch(f"/api/experiments/{exp['id']}", json={"is_example": True})
    repo.add_spend(amount_usd=1.0, provider="meta", experiment_id=exp["id"])

    public = anon.get("/api/experiments", params={"examples": "true"}).json()[0]
    admin = client.get("/api/experiments").json()[0]

    assert public["cost_usd"] is None and admin["cost_usd"] == 1.0


def test_listing_pages(client):
    for i in range(3):
        create(client, title=f"e{i}")

    page = client.get("/api/experiments", params={"limit": 2}).json()
    rest = client.get("/api/experiments", params={"limit": 2, "offset": 2}).json()

    assert [e["title"] for e in page] == ["e2", "e1"]
    assert [e["title"] for e in rest] == ["e0"]


def test_patch_renames_and_unpublishes(client):
    exp = create(client).json()

    published = client.patch(
        f"/api/experiments/{exp['id']}", json={"is_example": True, "title": "Nice"}
    )
    assert published.json()["is_example"] and published.json()["visibility"] == "public"
    assert published.json()["title"] == "Nice"

    hidden = client.patch(
        f"/api/experiments/{exp['id']}", json={"is_example": False, "visibility": "private"}
    )
    assert hidden.json()["visibility"] == "private" and hidden.json()["is_example"] is False
    assert client.patch(f"/api/experiments/{exp['id']}", json={}).status_code == 200
    assert client.patch(f"/api/experiments/{exp['id']}", json={"title": " "}).status_code == 422


# --- delete ---------------------------------------------------------------------------------
def test_delete_removes_rows_volume_folder_and_meta_files(client, repo, paths, file_deleter, db):
    exp = create(client).json()
    trial = repo.create_trial(experiment_id=exp["id"], role="fit")
    repo.update_trial(trial.id, meta_file_ids={"master": "file_a", "wonder_480": "file_b"})
    upload = repo.create_upload(
        experiment_id=exp["id"],
        role="fit",
        filename="a.mov",
        size_bytes=5,
        chunk_size=5,
        n_chunks=1,
    )
    media = paths.trial_file(exp["id"], trial.id, "master.mp4")
    media.parent.mkdir(parents=True)
    media.write_bytes(b"video")
    paths.upload_part(upload.id, 0).parent.mkdir(parents=True)
    paths.upload_part(upload.id, 0).write_bytes(b"12345")
    other = create(client, title="keep me").json()

    response = client.delete(f"/api/experiments/{exp['id']}")

    assert response.status_code == 200
    assert response.json() == {"deleted": True, "meta_files_deleted": 2, "meta_files_failed": 0}
    assert sorted(file_deleter.deleted) == ["file_a", "file_b"]
    assert not paths.experiment_dir(exp["id"]).exists()
    assert not paths.upload_dir(upload.id).exists()
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 404
    assert db.fetch_one("select count(*) as n from trials")["n"] == 0
    assert client.get(f"/api/experiments/{other['id']}").status_code == 200


def test_delete_reports_meta_files_it_could_not_remove(client, repo, file_deleter):
    exp = create(client).json()
    trial = repo.create_trial(experiment_id=exp["id"], role="fit")
    repo.update_trial(trial.id, meta_file_ids={"master": "file_a", "wonder_480": "file_b"})
    file_deleter.failing.add("file_b")

    body = client.delete(f"/api/experiments/{exp['id']}").json()

    assert (body["meta_files_deleted"], body["meta_files_failed"]) == (1, 1)
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 404  # still deleted


def test_delete_needs_the_passcode_and_spares_examples_and_running_work(client, anon, repo):
    exp = create(client).json()
    assert anon.delete(f"/api/experiments/{exp['id']}").status_code == 401

    client.patch(f"/api/experiments/{exp['id']}", json={"is_example": True})
    protected = client.delete(f"/api/experiments/{exp['id']}")
    assert (protected.status_code, protected.json()["code"]) == (409, "protected")

    client.patch(f"/api/experiments/{exp['id']}", json={"is_example": False})
    repo.create_job(experiment_id=exp["id"], kind="analyze_experiment")
    busy = client.delete(f"/api/experiments/{exp['id']}")
    assert (busy.status_code, busy.json()["code"]) == (409, "job_active")
    assert client.get(f"/api/experiments/{exp['id']}").status_code == 200


def test_benchmark_experiments_are_protected_too(client, repo):
    exp = create(client).json()
    repo.upsert_benchmark(
        exp["id"], BenchmarkEntry(system="free_fall", quantity="g", truth_value=9.8)
    )

    assert client.delete(f"/api/experiments/{exp['id']}").json()["code"] == "protected"


# --- benchmark ------------------------------------------------------------------------------
def test_benchmark_table_joins_truth_with_the_measured_result(client, anon, repo):
    exp = create(client, title="Drop 1").json()
    other = create(client, title="Drop 2").json()
    entry = {
        "system": "free_fall",
        "quantity": "g",
        "truth_value": 9.8,
        "truth_sigma": 0.01,
        "truth_method": "latitude formula",
    }
    assert client.put(f"/api/benchmark/{exp['id']}", json=entry).status_code == 200
    client.put(f"/api/benchmark/{other['id']}", json={**entry, "truth_value": 9.79})
    conclusion = {
        "winner_id": "FF0",
        "quantities": [
            {"name": "g", "value": 9.75, "ci_low": 9.6, "ci_high": 9.9, "unit": "m/s^2"}
        ],
        "rejected": [],
        "confidence": "high",
        "caveats": [],
        "narrative_md": "",
    }
    repo.update_experiment(exp["id"], conclusion=conclusion, confidence="high", status="complete")

    table = anon.get("/api/benchmark").json()["rows"]

    assert [r["title"] for r in table] == ["Drop 1", "Drop 2"]
    first = table[0]
    assert first["truth_value"] == 9.8 and first["measured"]["value"] == 9.75
    assert first["confidence"] == "high" and first["status"] == "complete"
    assert table[1]["measured"] is None


def test_benchmark_entry_validation_and_overwrite(client, repo):
    exp = create(client).json()
    url = f"/api/benchmark/{exp['id']}"

    assert client.put(url, json={"system": "free_fall"}).status_code == 422
    assert (
        client.put(
            f"/api/benchmark/{uuid4()}", json={"system": "s", "quantity": "g", "truth_value": 1}
        ).status_code
        == 404
    )
    client.put(url, json={"system": "free_fall", "quantity": "g", "truth_value": 9.8})
    again = client.put(url, json={"system": "free_fall", "quantity": "g", "truth_value": 9.7})

    assert again.json()["truth_value"] == 9.7
    assert client.get(f"/api/experiments/{exp['id']}").json()["is_benchmark"] is True
    assert len(client.get("/api/benchmark").json()["rows"]) == 1


# --- admin ----------------------------------------------------------------------------------
def test_admin_overview_needs_the_passcode_and_shows_spend_and_jobs(client, anon, repo):
    assert anon.get("/api/admin/overview").status_code == 401
    exp = create(client).json()
    repo.add_spend(amount_usd=2.5, provider="meta", experiment_id=exp["id"])
    job = repo.create_job(experiment_id=exp["id"], kind="analyze_experiment")

    body = client.get("/api/admin/overview").json()

    assert body["spent_today_usd"] == 2.5 and body["budget_daily_usd"] == 20
    assert body["recent_jobs"][0]["id"] == str(job.id)
    assert body["experiments_by_cost"][0]["cost_usd"] == 2.5
