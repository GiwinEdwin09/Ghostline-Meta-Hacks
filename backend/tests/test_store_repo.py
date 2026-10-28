import threading
from uuid import uuid4

import pytest

from ghostline.store.core_schemas import Conclusion, Quantity
from ghostline.store.records import (
    BenchmarkEntry,
    DeclaredCapture,
    FitParam,
    FitRecord,
    GhostRender,
    LlmCall,
    ModelRecord,
    ReviewRecord,
    Setup,
    TrackRecord,
    TrajectoryRecord,
    Trim,
)
from ghostline.store.repo import ActiveJobError


def make_experiment(repo, **kwargs):
    return repo.create_experiment(title=kwargs.pop("title", "Drop test"), **kwargs)


def test_create_and_read_experiment_round_trips_setup(repo):
    created = make_experiment(
        repo,
        system_type="pendulum",
        declared_capture=DeclaredCapture(mode="slow_mo", fps=240),
        setup=Setup(plane_offset_cm=3.5, paper="letter", pendulum_length_cm=62.0),
    )

    loaded = repo.get_experiment(created.id)

    assert loaded.status == "draft"
    assert loaded.visibility == "private"
    assert loaded.declared_capture == DeclaredCapture(mode="slow_mo", fps=240)
    assert loaded.setup.pendulum_length_cm == 62.0
    assert loaded.cost_usd == 0


def test_unknown_experiment_is_none(repo):
    assert repo.get_experiment(uuid4()) is None
    assert repo.get_visibility(uuid4()) is None


def test_update_experiment_stores_pydantic_json(repo):
    exp = make_experiment(repo)
    conclusion = Conclusion(
        winner_id="P1",
        quantities=[Quantity(name="g", value=9.8, ci_low=9.7, ci_high=9.9, unit="m/s^2")],
        rejected=[],
        confidence="medium",
        caveats=["short clip"],
        narrative_md="text",
    )

    updated = repo.update_experiment(
        exp.id, conclusion=conclusion, winner_model_id="P1", confidence="medium", status="complete"
    )

    assert updated.conclusion.quantities[0].value == 9.8
    assert repo.get_experiment(exp.id).confidence == "medium"


def test_update_rejects_unknown_columns(repo):
    exp = make_experiment(repo)

    with pytest.raises(ValueError, match="cannot update"):
        repo.update_experiment(exp.id, cost_usd=5)
    with pytest.raises(ValueError, match="nothing to update"):
        repo.update_experiment(exp.id)


def test_list_experiments_examples_only_returns_public_examples(repo):
    plain = make_experiment(repo, title="private one")
    example = make_experiment(repo, title="example")
    repo.update_experiment(example.id, is_example=True, visibility="public")
    hidden = make_experiment(repo, title="example but private")
    repo.update_experiment(hidden.id, is_example=True)

    listed = repo.list_experiments(examples_only=True)

    assert [e.id for e in listed] == [example.id]
    assert {e.id for e in repo.list_experiments()} == {plain.id, example.id, hidden.id}


def test_upload_chunk_tracking_is_a_sorted_set(repo):
    exp = make_experiment(repo)
    upload = repo.create_upload(
        experiment_id=exp.id,
        role="fit",
        filename="a.mov",
        size_bytes=3000,
        chunk_size=1024,
        n_chunks=3,
    )

    for n in (2, 0, 2, 1, 0):
        updated = repo.add_received_chunk(upload.id, n)

    assert updated.received_chunks == [0, 1, 2]


def test_chunks_received_in_parallel_are_all_recorded(repo):
    exp = make_experiment(repo)
    upload = repo.create_upload(
        experiment_id=exp.id,
        role="fit",
        filename="a.mov",
        size_bytes=9000,
        chunk_size=1000,
        n_chunks=9,
    )

    threads = [
        threading.Thread(target=repo.add_received_chunk, args=(upload.id, n)) for n in range(9)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert repo.get_upload(upload.id).received_chunks == list(range(9))


def test_completed_upload_stops_accepting_chunks(repo):
    exp = make_experiment(repo)
    upload = repo.create_upload(
        experiment_id=exp.id,
        role="fit",
        filename="a.mov",
        size_bytes=10,
        chunk_size=1024,
        n_chunks=1,
    )
    trial = repo.create_trial(experiment_id=exp.id, role="fit")
    repo.mark_upload_complete(upload.id, trial.id)

    assert repo.add_received_chunk(upload.id, 0) is None
    assert repo.get_upload(upload.id).status == "complete"


def test_trial_updates_and_path_merge(repo):
    exp = make_experiment(repo)
    trial = repo.create_trial(
        experiment_id=exp.id, role="fit", paths={"original": "a/original.mp4"}
    )

    repo.update_trial(
        trial.id,
        status="ready",
        n_frames=600,
        trim=Trim(start_s=1, end_s=5),
        meta_file_ids={"master": "file_1"},
    )
    repo.merge_trial_paths(trial.id, {"master": "a/master.mp4"})
    loaded = repo.get_trial(trial.id)

    assert loaded.status == "ready"
    assert loaded.trim.end_s == 5
    assert loaded.paths == {"original": "a/original.mp4", "master": "a/master.mp4"}
    assert repo.latest_trial(exp.id, "fit", status="ready").id == trial.id
    assert repo.latest_trial(exp.id, "prove") is None


def test_tracks_and_trajectories_upsert(repo):
    exp = make_experiment(repo)
    trial = repo.create_trial(experiment_id=exp.id, role="fit")

    repo.upsert_track(TrackRecord(trial_id=trial.id, object_key="bob", stats={"frames": 1}))
    repo.upsert_track(TrackRecord(trial_id=trial.id, object_key="bob", stats={"frames": 2}))
    repo.upsert_trajectory(TrajectoryRecord(trial_id=trial.id, summary={"n": 1}))
    repo.upsert_trajectory(TrajectoryRecord(trial_id=trial.id, summary={"n": 2}))

    assert [t.stats for t in repo.list_tracks([trial.id])] == [{"frames": 2}]
    assert repo.list_trajectories([trial.id])[0].summary == {"n": 2}


def test_models_fits_and_reviews(repo):
    exp = make_experiment(repo)
    trial = repo.create_trial(experiment_id=exp.id, role="fit")
    repo.upsert_model(ModelRecord(experiment_id=exp.id, id="P1", name="Nonlinear", prior_rank=1))
    repo.upsert_model(ModelRecord(experiment_id=exp.id, id="P0", name="Small angle", prior_rank=2))
    repo.upsert_model(
        ModelRecord(experiment_id=exp.id, id="P0", name="Small angle v2", prior_rank=2)
    )
    repo.set_model_status(exp.id, "P0", "rejected", "held-out error 5x larger")

    fit = repo.insert_fit(
        FitRecord(
            experiment_id=exp.id,
            model_id="P1",
            trial_id=trial.id,
            params=[FitParam(name="g", value=9.8, ci_low=9.7, ci_high=9.9, unit="m/s^2")],
            heldout_rmse=0.002,
            heldout_ci=[0.001, 0.003],
        )
    )
    review = repo.insert_review(ReviewRecord(experiment_id=exp.id, verdict="challenge"))
    repo.set_review_response(review.id, {"text": "rebuttal"})

    models = repo.list_models(exp.id)
    assert [m.id for m in models] == ["P1", "P0"]
    assert models[1].name == "Small angle v2"
    assert models[1].status == "rejected"
    assert repo.get_model(exp.id, "P1").prior_rank == 1
    assert repo.list_fits(exp.id)[0].params[0].value == 9.8
    assert fit.heldout_ci == [0.001, 0.003]
    assert repo.list_reviews(exp.id)[0].response == {"text": "rebuttal"}


def test_fits_need_a_model_that_exists(repo):
    import psycopg

    exp = make_experiment(repo)
    trial = repo.create_trial(experiment_id=exp.id, role="fit")

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        repo.insert_fit(FitRecord(experiment_id=exp.id, model_id="nope", trial_id=trial.id))


def test_ghost_versions_count_up_per_experiment(repo):
    a, b = make_experiment(repo), make_experiment(repo)
    trial_a = repo.create_trial(experiment_id=a.id, role="prove")
    trial_b = repo.create_trial(experiment_id=b.id, role="prove")

    versions = [
        repo.add_ghost_render(GhostRender(experiment_id=a.id, trial_id=trial_a.id)).version
        for _ in range(3)
    ]
    other = repo.add_ghost_render(GhostRender(experiment_id=b.id, trial_id=trial_b.id))
    explicit = repo.add_ghost_render(
        GhostRender(experiment_id=b.id, trial_id=trial_b.id, version=7)
    )

    assert versions == [1, 2, 3]
    assert other.version == 1
    assert explicit.version == 7
    assert repo.next_ghost_version(a.id) == 4
    assert [g.version for g in repo.list_ghosts(a.id)] == [1, 2, 3]


def test_concurrent_ghost_inserts_get_distinct_versions(repo):
    exp = make_experiment(repo)
    trial = repo.create_trial(experiment_id=exp.id, role="prove")

    threads = [
        threading.Thread(
            target=repo.add_ghost_render,
            args=(GhostRender(experiment_id=exp.id, trial_id=trial.id),),
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [g.version for g in repo.list_ghosts(exp.id)] == [1, 2, 3, 4, 5, 6]


def test_benchmark_upsert_marks_experiment_and_joins_in_listing(repo):
    exp = make_experiment(repo)
    entry = BenchmarkEntry(system="free_fall", quantity="g", truth_value=9.8, truth_sigma=0.01)

    repo.upsert_benchmark(exp.id, entry)
    repo.upsert_benchmark(exp.id, entry.model_copy(update={"truth_value": 9.79}))

    rows = repo.list_benchmark()
    assert len(rows) == 1 and rows[0]["truth_value"] == 9.79 and rows[0]["title"] == "Drop test"
    assert repo.get_experiment(exp.id).is_benchmark is True


def test_spend_ledger_sums_by_day_and_experiment_and_survives_delete(repo):
    a, b = make_experiment(repo), make_experiment(repo)

    repo.add_spend(amount_usd=1.25, provider="meta", experiment_id=a.id)
    repo.add_spend(amount_usd=0.5, provider="meta", experiment_id=a.id, category="sam")
    repo.add_spend(amount_usd=2.0, provider="modal", experiment_id=b.id)

    assert repo.spend_today() == pytest.approx(3.75)
    assert repo.spend_for_experiment(a.id) == pytest.approx(1.75)
    assert repo.get_experiment(a.id).cost_usd == pytest.approx(1.75)

    repo.delete_experiment(b.id)
    assert repo.spend_today() == pytest.approx(3.75)


def test_llm_call_stats(repo):
    exp = make_experiment(repo)
    repo.insert_llm_call(
        LlmCall(experiment_id=exp.id, provider="meta", model="wonder", cost_usd=0.1)
    )
    repo.insert_llm_call(
        LlmCall(experiment_id=exp.id, provider="meta", model="wonder", cache_hit=True)
    )

    assert repo.llm_stats_today() == {"calls": 2, "cache_hits": 1}


def test_deleting_an_experiment_removes_everything_under_it(repo, db):
    exp = make_experiment(repo)
    trial = repo.create_trial(experiment_id=exp.id, role="fit")
    repo.upsert_model(ModelRecord(experiment_id=exp.id, id="P1", name="n"))
    repo.insert_fit(FitRecord(experiment_id=exp.id, model_id="P1", trial_id=trial.id))
    repo.insert_review(ReviewRecord(experiment_id=exp.id, verdict="accept"))
    repo.add_ghost_render(GhostRender(experiment_id=exp.id, trial_id=trial.id))
    repo.append_event(exp.id, actor="system", kind="observation", title="x")
    repo.create_job(experiment_id=exp.id, kind="analyze_experiment")
    repo.create_upload(
        experiment_id=exp.id, role="fit", filename="a", size_bytes=1, chunk_size=1, n_chunks=1
    )

    assert repo.delete_experiment(exp.id) is True

    for table in (
        "trials",
        "models",
        "fits",
        "reviews",
        "ghost_renders",
        "events",
        "jobs",
        "uploads",
    ):
        assert db.fetch_one(f"select count(*) as n from {table}")["n"] == 0
    assert repo.delete_experiment(exp.id) is False


# --- jobs -----------------------------------------------------------------------------------
def test_job_claim_runs_once(repo):
    exp = make_experiment(repo)
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment", params={"a": 1})

    first = repo.claim_job(job.id)
    second = repo.claim_job(job.id)

    assert first.status == "running" and first.attempt == 1 and first.started_at is not None
    assert second is None


def test_one_active_job_per_experiment(repo):
    exp = make_experiment(repo)
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment")

    with pytest.raises(ActiveJobError) as exc:
        repo.create_job(experiment_id=exp.id, kind="revise")
    assert exc.value.job.id == job.id

    repo.finish_job(job.id, status="succeeded")
    assert repo.create_job(experiment_id=exp.id, kind="revise").kind == "revise"


def test_racing_job_creation_lets_exactly_one_through(repo):
    exp = make_experiment(repo)
    outcomes = []

    def attempt():
        try:
            repo.create_job(experiment_id=exp.id, kind="analyze_experiment")
            outcomes.append("created")
        except ActiveJobError:
            outcomes.append("active")

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == ["active"] * 7 + ["created"]


def test_job_phase_progress_never_goes_backwards_and_needs_running(repo):
    exp = make_experiment(repo)
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment")

    assert repo.set_job_phase(job.id, "tracking", 0.3) is False  # still queued
    repo.claim_job(job.id)
    assert repo.set_job_phase(job.id, "tracking", 0.3) is True
    repo.set_job_progress(job.id, 0.2)
    assert repo.get_job(job.id).progress == pytest.approx(0.3)
    repo.set_job_phase(job.id, "fitting", 0.6)
    loaded = repo.get_job(job.id)
    assert loaded.phase == "fitting" and loaded.progress == pytest.approx(0.6)


def test_finish_job_is_final_and_success_fills_the_bar(repo):
    exp = make_experiment(repo)
    job = repo.create_job(experiment_id=exp.id, kind="ingest_trial")
    repo.claim_job(job.id)

    done = repo.finish_job(job.id, status="succeeded")
    again = repo.finish_job(job.id, status="failed", error="late")

    assert done.progress == 1.0 and done.finished_at is not None
    assert again is None
    assert repo.get_job(job.id).status == "succeeded"
    assert repo.heartbeat(job.id) is False


def test_stale_jobs_are_found_by_heartbeat_age(repo, db):
    exp_a, exp_b, exp_c = (make_experiment(repo) for _ in range(3))
    running = repo.create_job(experiment_id=exp_a.id, kind="analyze_experiment")
    repo.claim_job(running.id)
    healthy = repo.create_job(experiment_id=exp_b.id, kind="analyze_experiment")
    repo.claim_job(healthy.id)
    queued = repo.create_job(experiment_id=exp_c.id, kind="analyze_experiment")
    db.execute(
        "update jobs set heartbeat_at = now() - interval '90 seconds' where id = %s", (running.id,)
    )
    db.execute(
        "update jobs set heartbeat_at = now() - interval '90 seconds' where id = %s", (queued.id,)
    )

    stale = repo.stale_jobs(running_timeout_s=60, queued_timeout_s=180)
    assert [j.id for j in stale] == [running.id]

    stale = repo.stale_jobs(running_timeout_s=60, queued_timeout_s=30)
    assert {j.id for j in stale} == {running.id, queued.id}
    assert [
        j.id
        for j in repo.stale_jobs(running_timeout_s=60, queued_timeout_s=30, experiment_id=exp_c.id)
    ] == [queued.id]


def test_recent_analyses_count_by_client_and_ignore_retries(repo):
    exp = make_experiment(repo)
    first = repo.create_job(experiment_id=exp.id, kind="analyze_experiment", client_ip_hash="aaa")
    repo.finish_job(first.id, status="failed", error="x")
    retry = repo.create_job(
        experiment_id=exp.id, kind="analyze_experiment", client_ip_hash="aaa", retry_of=first.id
    )
    repo.finish_job(retry.id, status="failed", error="x")
    repo.create_job(experiment_id=exp.id, kind="revise", client_ip_hash="aaa")

    assert repo.count_recent_analyses("aaa") == 1
    assert repo.count_recent_analyses("bbb") == 0


def test_transaction_rolls_back_on_error(repo):
    exp = make_experiment(repo)

    with pytest.raises(RuntimeError):
        with repo.transaction() as tx:
            tx.create_trial(experiment_id=exp.id, role="fit")
            raise RuntimeError("boom")

    assert repo.list_trials(exp.id) == []

    with repo.transaction() as tx:
        tx.create_trial(experiment_id=exp.id, role="fit")
    assert len(repo.list_trials(exp.id)) == 1
