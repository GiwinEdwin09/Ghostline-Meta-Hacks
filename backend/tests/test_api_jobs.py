from uuid import uuid4

from ghostline.store.core_schemas import Conclusion, Quantity
from ghostline.store.records import GhostRender, ModelRecord, Trim


def make_experiment(client, **kwargs):
    return client.post("/api/experiments", json={"title": "swing", **kwargs}).json()["id"]


def add_trial(repo, exp, role="fit", status="ready"):
    trial = repo.create_trial(experiment_id=exp, role=role)
    repo.update_trial(
        trial.id, status=status, n_frames=600, duration_s=10.0, trim=Trim(start_s=0, end_s=10)
    )
    return trial


def finish_analysis(repo, exp):
    conclusion = Conclusion(
        winner_id="P1",
        quantities=[Quantity(name="g", value=9.8, ci_low=9.7, ci_high=9.9, unit="m/s^2")],
        rejected=[],
        confidence="high",
        caveats=[],
        narrative_md="",
    )
    repo.update_experiment(
        exp, status="complete", conclusion=conclusion, winner_model_id="P1", confidence="high"
    )


def finish_job(repo, job_id, status="succeeded"):
    repo.claim_job(job_id) if repo.get_job(job_id).status == "queued" else None
    repo.finish_job(job_id, status=status, error=None if status == "succeeded" else "x")


# --- analyze --------------------------------------------------------------------------------
def test_analyze_queues_and_spawns_a_job_for_the_newest_ready_clip(client, spawner, repo, db):
    exp = make_experiment(client)
    add_trial(repo, exp)
    newest = add_trial(repo, exp)

    response = client.post(f"/api/experiments/{exp}/analyze")

    assert response.status_code == 202
    job = response.json()
    assert (job["kind"], job["status"], job["progress"]) == ("analyze_experiment", "queued", 0)
    assert job["params"] == {"trial_id": str(newest.id)}
    assert [j.id for j in spawner.jobs] == [repo.get_job(job["id"]).id]
    assert (
        db.fetch_one("select call_id from jobs where id = %s", (job["id"],))["call_id"] == "call-1"
    )


def test_analyze_needs_a_ready_clip(client, spawner, repo):
    exp = make_experiment(client)
    none = client.post(f"/api/experiments/{exp}/analyze")
    add_trial(repo, exp, status="ingesting")
    still_preparing = client.post(f"/api/experiments/{exp}/analyze")
    prove_only = add_trial(repo, exp, role="prove")
    wrong_role = client.post(
        f"/api/experiments/{exp}/analyze", json={"trial_id": str(prove_only.id)}
    )

    assert (none.status_code, none.json()["code"]) == (409, "no_ready_trial")
    assert still_preparing.json()["code"] == "no_ready_trial"
    assert wrong_role.json()["code"] == "wrong_trial_role"
    assert spawner.jobs == []


def test_analyze_cannot_use_another_experiments_clip(client, repo):
    mine, theirs = make_experiment(client), make_experiment(client)
    foreign = add_trial(repo, theirs)

    response = client.post(f"/api/experiments/{mine}/analyze", json={"trial_id": str(foreign.id)})

    assert (response.status_code, response.json()["code"]) == (409, "no_ready_trial")


def test_asking_twice_returns_the_running_job(client, spawner, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)

    first = client.post(f"/api/experiments/{exp}/analyze").json()
    second = client.post(f"/api/experiments/{exp}/analyze")

    assert second.status_code == 202 and second.json()["id"] == first["id"]
    assert len(spawner.jobs) == 1


def test_a_different_job_cannot_start_while_one_is_active(client, spawner, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)
    running = client.post(f"/api/experiments/{exp}/analyze").json()
    finish_analysis(repo, exp)
    prove = add_trial(repo, exp, role="prove")

    blocked = client.post(f"/api/experiments/{exp}/prove", json={"trial_id": str(prove.id)})

    assert (blocked.status_code, blocked.json()["code"]) == (409, "job_active")
    assert blocked.json()["job_id"] == running["id"]


def test_two_experiments_can_analyze_at_the_same_time(client, spawner, repo):
    a, b = make_experiment(client), make_experiment(client)
    add_trial(repo, a)
    add_trial(repo, b)

    assert client.post(f"/api/experiments/{a}/analyze").status_code == 202
    assert client.post(f"/api/experiments/{b}/analyze").status_code == 202
    assert len(spawner.jobs) == 2


def test_racing_analyze_requests_start_one_job(client, spawner, repo):
    from concurrent.futures import ThreadPoolExecutor

    exp = make_experiment(client)
    add_trial(repo, exp)

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(
            pool.map(lambda _: client.post(f"/api/experiments/{exp}/analyze"), range(6))
        )

    assert {r.status_code for r in responses} == {202}
    assert len({r.json()["id"] for r in responses}) == 1
    assert len(spawner.jobs) == 1


def test_a_failed_spawn_leaves_a_retryable_failed_job(client, spawner, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)
    spawner.fail_with = RuntimeError("no workers")

    job = client.post(f"/api/experiments/{exp}/analyze").json()

    assert (job["status"], job["error_code"]) == ("failed", "spawn_failed")
    assert client.get(f"/api/experiments/{exp}").json()["status"] == "failed"
    spawner.fail_with = None
    assert client.post(f"/api/jobs/{job['id']}/retry").json()["status"] == "queued"


# --- the daily budget and the per-address limit ------------------------------------------------
def test_new_analyses_are_refused_once_the_daily_budget_is_spent(make_client, spawner, repo):
    client = make_client(spawner, budget_usd_daily=5.0)
    exp = make_experiment(client)
    add_trial(repo, exp)
    repo.add_spend(amount_usd=4.99, provider="meta")
    assert client.post(f"/api/experiments/{exp}/analyze").status_code == 202
    finish_job(repo, spawner.jobs[0].id)
    repo.add_spend(amount_usd=0.01, provider="meta")

    refused = client.post(f"/api/experiments/{exp}/analyze")

    assert refused.status_code == 429 and refused.json()["code"] == "daily_budget_reached"
    assert "tomorrow" in refused.json()["detail"]
    assert len(spawner.jobs) == 1


def test_the_budget_blocks_analyses_but_not_viewing_examples(make_client, spawner, repo, anon):
    client = make_client(spawner, budget_usd_daily=1.0)
    exp = make_experiment(client)
    client.patch(f"/api/experiments/{exp}", json={"is_example": True})
    repo.add_spend(amount_usd=1.0, provider="meta")

    from fastapi.testclient import TestClient

    public = TestClient(client.app)
    assert public.get("/api/experiments", params={"examples": "true"}).status_code == 200
    assert public.get(f"/api/experiments/{exp}").status_code == 200


def test_budget_only_counts_today(repo, db, make_client, spawner):
    client = make_client(spawner, budget_usd_daily=1.0)
    exp = make_experiment(client)
    add_trial(repo, exp)
    repo.add_spend(amount_usd=5.0, provider="meta")
    db.execute("update spend_ledger set day = day - 1")

    assert client.post(f"/api/experiments/{exp}/analyze").status_code == 202


def test_ten_analyses_per_address_per_day(make_client, spawner, repo):
    client = make_client(spawner, analyses_per_ip_per_day=2)
    headers = {"X-Forwarded-For": "9.9.9.9"}
    exps = []
    for _ in range(3):
        exp = make_experiment(client)
        add_trial(repo, exp)
        exps.append(exp)

    for exp in exps[:2]:
        assert client.post(f"/api/experiments/{exp}/analyze", headers=headers).status_code == 202
        finish_job(repo, spawner.jobs[-1].id)
    third = client.post(f"/api/experiments/{exps[2]}/analyze", headers=headers)
    elsewhere = client.post(
        f"/api/experiments/{exps[2]}/analyze", headers={"X-Forwarded-For": "8.8.8.8"}
    )

    assert (third.status_code, third.json()["code"]) == (429, "rate_limited")
    assert "2 analyses" in third.json()["detail"]
    assert elsewhere.status_code == 202


def test_a_forged_forwarded_for_header_does_not_dodge_the_limit(make_client, spawner, repo):
    client = make_client(spawner, analyses_per_ip_per_day=1)
    first, second = make_experiment(client), make_experiment(client)
    add_trial(repo, first)
    add_trial(repo, second)
    real = "5.5.5.5"
    client.post(
        f"/api/experiments/{first}/analyze", headers={"X-Forwarded-For": f"1.1.1.1, {real}"}
    )
    finish_job(repo, spawner.jobs[-1].id)

    blocked = client.post(
        f"/api/experiments/{second}/analyze", headers={"X-Forwarded-For": f"2.2.2.2, {real}"}
    )

    assert blocked.status_code == 429


def test_retrying_a_failed_analysis_does_not_use_up_the_limit(make_client, spawner, repo):
    client = make_client(spawner, analyses_per_ip_per_day=1)
    exp = make_experiment(client)
    add_trial(repo, exp)
    job = client.post(f"/api/experiments/{exp}/analyze").json()
    finish_job(repo, job["id"], status="failed")

    retry = client.post(f"/api/jobs/{job['id']}/retry")

    assert retry.status_code == 202


# --- prove / revise / rerun / export check --------------------------------------------------
def test_prove_needs_an_analyzed_experiment_and_a_ready_proof_clip(client, spawner, repo):
    exp = make_experiment(client)
    early = client.post(f"/api/experiments/{exp}/prove")
    finish_analysis(repo, exp)
    no_clip = client.post(f"/api/experiments/{exp}/prove")
    clip = add_trial(repo, exp, role="prove")

    ok = client.post(f"/api/experiments/{exp}/prove")

    assert (early.status_code, early.json()["code"]) == (409, "not_analyzed")
    assert no_clip.json()["code"] == "no_prove_trial"
    assert ok.status_code == 202
    assert ok.json()["kind"] == "prove_trial" and ok.json()["params"] == {"trial_id": str(clip.id)}


def test_prove_with_the_first_clip_is_refused(client, repo):
    exp = make_experiment(client)
    finish_analysis(repo, exp)
    first = add_trial(repo, exp, role="fit")

    response = client.post(f"/api/experiments/{exp}/prove", json={"trial_id": str(first.id)})

    assert response.json()["code"] == "wrong_trial_role"


def test_revise_needs_a_ghost(client, repo):
    exp = make_experiment(client)
    finish_analysis(repo, exp)
    none = client.post(f"/api/experiments/{exp}/revise")
    prove = add_trial(repo, exp, role="prove")
    ghost = repo.add_ghost_render(GhostRender(experiment_id=exp, trial_id=prove.id))
    repo.add_ghost_render(GhostRender(experiment_id=exp, trial_id=prove.id))

    wrong = client.post(f"/api/experiments/{exp}/revise", json={"ghost_id": str(uuid4())})
    chosen = client.post(f"/api/experiments/{exp}/revise", json={"ghost_id": str(ghost.id)})

    assert (none.status_code, none.json()["code"]) == (409, "no_ghost")
    assert wrong.status_code == 404
    assert chosen.status_code == 202 and chosen.json()["params"] == {"ghost_id": str(ghost.id)}


def test_revise_defaults_to_the_newest_ghost(client, repo):
    exp = make_experiment(client)
    prove = add_trial(repo, exp, role="prove")
    repo.add_ghost_render(GhostRender(experiment_id=exp, trial_id=prove.id))
    newest = repo.add_ghost_render(GhostRender(experiment_id=exp, trial_id=prove.id))

    job = client.post(f"/api/experiments/{exp}/revise").json()

    assert job["params"] == {"ghost_id": str(newest.id)}


def test_rerun_checks_the_model_and_its_parameters(client, spawner, repo):
    exp = make_experiment(client)
    from ghostline.store.core_schemas import ParamSpec

    repo.upsert_model(
        ModelRecord(
            experiment_id=exp,
            id="P1",
            name="Nonlinear",
            params_spec={
                "g": ParamSpec(init=9.8, lower=5, upper=15, unit="m/s^2", kind="physical")
            },
        )
    )
    url = f"/api/experiments/{exp}/models/P1/rerun"

    ok = client.post(url, json={"split_pct": 0.5, "initial_guesses": {"g": 9.0}})
    missing_model = client.post(f"/api/experiments/{exp}/models/ZZ/rerun", json={})

    assert ok.status_code == 202 and ok.json()["kind"] == "rerun"
    assert ok.json()["params"] == {
        "model_id": "P1",
        "split_pct": 0.5,
        "initial_guesses": {"g": 9.0},
    }
    assert missing_model.status_code == 404
    finish_job(repo, ok.json()["id"])
    assert (
        client.post(url, json={"initial_guesses": {"nope": 1.0}}).json()["code"]
        == "unknown_parameter"
    )
    assert client.post(url, json={"split_pct": 0.95}).status_code == 422
    assert client.post(url, json={"split_pct": 0.1}).status_code == 422
    assert client.post(url, json={"initial_guesses": {"g": "high"}}).status_code == 422
    assert client.post(url, json={}).json()["params"]["split_pct"] == 0.65  # the default split


def test_export_check_needs_a_result(client, repo):
    exp = make_experiment(client)
    early = client.post(f"/api/experiments/{exp}/export/verify")
    finish_analysis(repo, exp)

    ok = client.post(f"/api/experiments/{exp}/export/verify")

    assert early.json()["code"] == "not_analyzed"
    assert ok.status_code == 202 and ok.json()["kind"] == "export_verify"


# --- retry ----------------------------------------------------------------------------------
def test_retry_starts_a_new_job_with_the_same_inputs(client, spawner, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)
    failed = client.post(f"/api/experiments/{exp}/analyze").json()
    finish_job(repo, failed["id"], status="failed")

    retry = client.post(f"/api/jobs/{failed['id']}/retry")

    assert retry.status_code == 202
    job = retry.json()
    assert job["id"] != failed["id"] and job["kind"] == "analyze_experiment"
    assert job["params"] == failed["params"] and job["status"] == "queued"
    assert repo.get_job(job["id"]).retry_of == repo.get_job(failed["id"]).id
    assert len(spawner.jobs) == 2


def test_only_failed_retryable_jobs_can_be_retried(client, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)
    running = client.post(f"/api/experiments/{exp}/analyze").json()

    assert client.post(f"/api/jobs/{running['id']}/retry").json()["code"] == "not_failed"
    repo.claim_job(running["id"])
    repo.finish_job(running["id"], status="failed", error="bad clip", retryable=False)
    assert client.post(f"/api/jobs/{running['id']}/retry").json()["code"] == "not_retryable"
    assert client.post(f"/api/jobs/{uuid4()}/retry").status_code == 404


def test_get_job(client, anon, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)
    job = client.post(f"/api/experiments/{exp}/analyze").json()

    fetched = anon.get(f"/api/jobs/{job['id']}")

    assert fetched.status_code == 200 and fetched.json()["id"] == job["id"]
    assert anon.get(f"/api/jobs/{uuid4()}").status_code == 404
    assert client.get("/api/jobs/not-a-uuid").status_code == 422


# --- a killed worker recovers through Retry (Part 27's done-when) -----------------------------
def test_a_worker_that_goes_quiet_is_failed_on_the_next_poll_and_can_be_retried(
    client, spawner, repo, db
):
    exp = make_experiment(client)
    add_trial(repo, exp)
    job = client.post(f"/api/experiments/{exp}/analyze").json()
    repo.claim_job(job["id"])
    repo.set_job_phase(job["id"], "tracking", 0.2)
    # The worker is killed: its heartbeat stops.
    db.execute(
        "update jobs set heartbeat_at = now() - interval '2 minutes' where id = %s", (job["id"],)
    )

    events = client.get(f"/api/experiments/{exp}/events").json()

    assert events["job"]["status"] == "failed"
    assert events["job"]["error_code"] == "heartbeat_timeout"
    assert events["job"]["phase"] == "tracking"
    assert events["status"] == "failed"
    assert "Retry" in events["job"]["error"]
    assert events["events"][-1]["kind"] == "warning"

    retry = client.post(f"/api/jobs/{job['id']}/retry")

    assert retry.status_code == 202 and retry.json()["status"] == "queued"
    state = client.get(f"/api/experiments/{exp}").json()
    assert state["job"]["id"] == retry.json()["id"]
    assert [j.status for j in repo.list_jobs(exp)] == ["queued", "failed"]


def test_a_job_nobody_picked_up_is_failed_too(client, repo, db):
    exp = make_experiment(client)
    add_trial(repo, exp)
    job = client.post(f"/api/experiments/{exp}/analyze").json()
    db.execute(
        "update jobs set heartbeat_at = now() - interval '4 minutes' where id = %s", (job["id"],)
    )

    state = client.get(f"/api/experiments/{exp}").json()

    assert state["job"]["status"] == "failed"
    assert state["job"]["error_code"] == "queue_timeout"


def test_polling_a_healthy_job_leaves_it_alone(client, repo):
    exp = make_experiment(client)
    add_trial(repo, exp)
    job = client.post(f"/api/experiments/{exp}/analyze").json()
    repo.claim_job(job["id"])

    state = client.get(f"/api/experiments/{exp}").json()

    assert state["job"]["status"] == "running"
