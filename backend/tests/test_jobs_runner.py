import threading
import time

import pytest

from ghostline.jobs import lifecycle, stages
from ghostline.jobs.errors import StageError, TransientStageError
from ghostline.jobs.runner import run_job
from ghostline.jobs.spawn import dispatch
from ghostline.store.records import GhostRender, ModelRecord, Trim
from ghostline.store.volume import NullVolume


class CountingVolume(NullVolume):
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


@pytest.fixture
def volume():
    return CountingVolume()


@pytest.fixture
def run(repo, paths, settings, volume):
    def go(job):
        return run_job(job.id, repo=repo, paths=paths, volume=volume, settings=settings)

    return go


def new_trial(repo, paths, exp, role="fit", status="uploaded"):
    trial = repo.create_trial(experiment_id=exp.id, role=role)
    original = paths.original(exp.id, trial.id, "mp4")
    original.parent.mkdir(parents=True)
    original.write_bytes(b"fake video bytes")
    repo.update_trial(trial.id, status=status, paths={"original": paths.rel(exp.id, original)})
    return repo.get_trial(trial.id)


def ready_trial(repo, paths, exp, role="fit"):
    trial = new_trial(repo, paths, exp, role, status="ready")
    repo.update_trial(
        trial.id,
        trim=Trim(start_s=0.5, end_s=9.5),
        n_frames=600,
        duration_s=10.0,
        paths={**trial.paths, "master": f"trials/{trial.id}/master.mp4"},
    )
    return repo.get_trial(trial.id)


def titles(repo, exp):
    return [e.title for e in repo.list_events(exp.id)]


# --- ingest ---------------------------------------------------------------------------------
def test_ingest_job_fills_in_the_trial_and_readies_the_experiment(
    repo, paths, run, volume, fake_stages
):
    exp = repo.create_experiment(title="drop")
    trial = new_trial(repo, paths, exp)
    job = repo.create_job(
        experiment_id=exp.id, kind="ingest_trial", params={"trial_id": str(trial.id)}
    )

    done = run(job)

    assert done.status == "succeeded" and done.progress == 1.0 and done.phase == "normalizing"
    loaded = repo.get_trial(trial.id)
    assert loaded.status == "ready"
    assert loaded.n_frames == 600 and loaded.duration_s == 10.0
    assert loaded.trim.suggested_start_s == 0.5 and loaded.trim.source == "auto"
    assert set(loaded.paths) == {"original", "master", "poster"}  # merged, not replaced
    assert loaded.meta_file_ids["master"].startswith("file_")
    assert repo.get_experiment(exp.id).status == "ready"
    assert titles(repo, exp) == ["Normalized the clip", "Clip warning"]
    assert volume.commits >= 2  # at the phase change and at the end


def test_failed_ingest_marks_the_trial_failed_and_returns_the_experiment_to_draft(
    repo, paths, run, fake_stages
):
    exp = repo.create_experiment(title="drop")
    trial = new_trial(repo, paths, exp)
    job = repo.create_job(
        experiment_id=exp.id, kind="ingest_trial", params={"trial_id": str(trial.id)}
    )
    fake_stages.fail_on["ingest"] = StageError(
        "That file has no video stream.", code="no_video", retryable=False
    )

    done = run(job)

    assert (done.status, done.error_code, done.retryable) == ("failed", "no_video", False)
    assert repo.get_trial(trial.id).status == "failed"
    assert repo.get_trial(trial.id).error == "That file has no video stream."
    assert repo.get_experiment(exp.id).status == "draft"
    [event] = repo.list_events(exp.id)
    assert (event.kind, event.actor, event.title) == (
        "warning",
        "system",
        "Preparing the clip failed",
    )
    assert event.body_md == "That file has no video stream."


# --- analyze --------------------------------------------------------------------------------
def start_analysis(repo, paths, system_type="pendulum"):
    exp = repo.create_experiment(title="swing", system_type=system_type)
    trial = ready_trial(repo, paths, exp)
    job = repo.create_job(
        experiment_id=exp.id, kind="analyze_experiment", params={"trial_id": str(trial.id)}
    )
    return exp, trial, job


def test_analysis_runs_every_stage_and_stores_the_result(
    repo, paths, run, fake_stages, monkeypatch
):
    exp, trial, job = start_analysis(repo, paths, system_type="auto")
    phases = []
    original = repo.set_job_phase
    monkeypatch.setattr(
        repo, "set_job_phase", lambda *args: (phases.append(args[1]), original(*args))[1]
    )

    done = run(job)

    assert done.status == "succeeded"
    assert fake_stages.calls == [
        "analyze_scene",
        "track_objects",
        "calibrate",
        "extract_trajectory",
        "run_physicist",
        "run_review",
    ]
    assert phases == [
        "watching",
        "tracking",
        "calibrating",
        "hypothesizing",
        "fitting",
        "peer_review",
    ]

    loaded = repo.get_experiment(exp.id)
    assert loaded.status == "complete"
    assert loaded.system_type == "pendulum"  # taken from the scene because the hint was "auto"
    assert loaded.scene.proceed is True
    assert loaded.winner_model_id == "P1" and loaded.confidence == "high"
    assert loaded.review_status == "accepted"
    assert loaded.conclusion.quantities[0].name == "g"
    assert repo.get_trial(trial.id).calibration["mm_per_px"] == 0.8
    assert repo.get_trial(trial.id).paths["calib_overlay"].endswith("overlay.png")
    assert [t.object_key for t in repo.list_tracks([trial.id])] == ["bob"]
    assert repo.list_trajectories([trial.id])[0].summary == {"n_points": 580}
    assert {m.id for m in repo.list_models(exp.id)} == {"P0", "P1"}
    assert len(repo.list_fits(exp.id)) == 1
    assert len(repo.list_reviews(exp.id)) == 1
    [event] = [e for e in repo.list_events(exp.id) if e.kind == "prediction"]
    assert event.phase == "hypothesizing"  # stamped with the phase the job was in


def test_a_user_chosen_system_hint_is_kept(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths, system_type="ramp")

    run(job)

    assert repo.get_experiment(exp.id).system_type == "ramp"


def test_progress_follows_the_phases(repo, paths, run, fake_stages, monkeypatch):
    exp, _, job = start_analysis(repo, paths)
    seen = []
    original = repo.set_job_phase
    monkeypatch.setattr(
        repo,
        "set_job_phase",
        lambda job_id, phase, progress: (
            seen.append((phase, progress)),
            original(job_id, phase, progress),
        )[1],
    )

    run(job)

    assert [p for p, _ in seen][0] == "watching"
    values = [v for _, v in seen]
    assert values == sorted(values)
    assert values[0] == 0.0 and 0.75 < values[-1] < 1.0
    assert repo.get_job(job.id).progress == 1.0


def test_a_scene_that_cannot_be_measured_fails_without_retry(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)
    scene = fake_stages.analyze_scene

    def refuse(ctx, experiment, trial):
        result = scene(ctx, experiment, trial)
        return result.model_copy(update={"proceed": False, "reason_if_not": "The sheet is hidden."})

    stages.analyze_scene = refuse  # replaced for this test; the fixture restores it

    done = run(job)

    assert (done.status, done.error_code, done.retryable) == ("failed", "cannot_measure", False)
    assert done.error == "The sheet is hidden."
    loaded = repo.get_experiment(exp.id)
    assert loaded.status == "failed" and loaded.error == "The sheet is hidden."
    assert "track_objects" not in fake_stages.calls


def test_unwired_stages_fail_the_job_with_the_wave_2_message(repo, paths, run, monkeypatch):
    from conftest import FakeStages

    # Every analysis stage is wired now; fake measurement and put a stub in the physicist's place.
    fake = FakeStages()
    for name in ("analyze_scene", "track_objects", "calibrate", "extract_trajectory"):
        monkeypatch.setattr(stages, name, getattr(fake, name))

    def stub(*_args):
        raise NotImplementedError(stages._WIRED_LATER)

    monkeypatch.setattr(stages, "run_physicist", stub)
    exp, _, job = start_analysis(repo, paths)

    done = run(job)

    assert (done.status, done.error_code) == ("failed", "not_implemented")
    assert "wired in wave 2" in done.error
    assert repo.get_experiment(exp.id).status == "failed"


def test_an_unexpected_exception_fails_the_job_and_can_be_retried(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)
    fake_stages.fail_on["calibrate"] = ZeroDivisionError("division by zero")

    done = run(job)

    assert (done.status, done.error_code, done.retryable) == ("failed", "internal_error", True)
    assert done.error == "ZeroDivisionError: division by zero"
    assert done.phase == "calibrating"  # where it stopped
    assert repo.get_experiment(exp.id).status == "failed"


def test_transient_stage_errors_are_retried_inside_the_job(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)
    failures = iter([TransientStageError("429"), TransientStageError("503")])
    real = fake_stages.track_objects

    def flaky(ctx, trial, scene):
        error = next(failures, None)
        if error:
            raise error
        return real(ctx, trial, scene)

    stages.track_objects = flaky

    done = run(job)

    assert done.status == "succeeded"


def test_transient_errors_that_keep_coming_fail_the_job(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)
    fake_stages.fail_on["track_objects"] = TransientStageError("503 again")

    done = run(job)

    assert (done.status, done.error, done.retryable) == ("failed", "503 again", True)
    assert fake_stages.calls.count("track_objects") == 3


def test_a_job_is_only_run_once(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)

    first = run(job)
    second = run(job)

    assert first.status == second.status == "succeeded"
    assert fake_stages.calls.count("analyze_scene") == 1
    assert first.attempt == 1


def test_the_heartbeat_keeps_a_long_stage_alive(repo, paths, run, fake_stages, db):
    exp, _, job = start_analysis(repo, paths)
    beats = []
    real = fake_stages.track_objects

    def slow(ctx, trial, scene):
        started = repo.get_job(job.id).heartbeat_at
        time.sleep(0.4)  # eight heartbeat intervals at 0.05 s
        beats.append(repo.get_job(job.id).heartbeat_at - started)
        return real(ctx, trial, scene)

    stages.track_objects = slow

    run(job)

    assert beats[0].total_seconds() > 0.2


def test_a_job_failed_by_the_reaper_stops_quietly(repo, paths, run, fake_stages, settings, db):
    exp, _, job = start_analysis(repo, paths)
    real = fake_stages.track_objects
    reaped = []

    def stall_then_continue(ctx, trial, scene):
        # Pretend the worker froze: its heartbeat goes stale and the API reaps it.
        db.execute(
            "update jobs set heartbeat_at = now() - interval '5 minutes' where id = %s", (job.id,)
        )
        reaped.extend(lifecycle.reap_stale_jobs(repo, settings, exp.id))
        deadline = time.time() + 2
        while not ctx.abandoned.is_set() and time.time() < deadline:
            time.sleep(0.01)  # the heartbeat thread notices the job is no longer running
        return real(ctx, trial, scene)

    stages.track_objects = stall_then_continue

    final = run(job)

    assert [j.id for j in reaped] == [job.id]
    assert final.status == "failed" and final.error_code == "heartbeat_timeout"
    assert "calibrate" not in fake_stages.calls  # the worker stopped at the next phase
    loaded = repo.get_experiment(exp.id)
    assert loaded.status == "failed"
    assert [e.title for e in repo.list_events(exp.id)].count("Analysis failed") == 1


def test_the_reaper_fails_silent_running_jobs_and_never_started_ones(repo, settings, db):
    running_exp = repo.create_experiment(title="a")
    queued_exp = repo.create_experiment(title="b")
    fresh_exp = repo.create_experiment(title="c")
    running = repo.create_job(experiment_id=running_exp.id, kind="analyze_experiment")
    repo.claim_job(running.id)
    queued = repo.create_job(experiment_id=queued_exp.id, kind="analyze_experiment")
    fresh = repo.create_job(experiment_id=fresh_exp.id, kind="analyze_experiment")
    repo.claim_job(fresh.id)
    db.execute(
        "update jobs set heartbeat_at = now() - interval '61 seconds' where id = %s", (running.id,)
    )
    db.execute(
        "update jobs set heartbeat_at = now() - interval '181 seconds' where id = %s", (queued.id,)
    )

    reaped = lifecycle.reap_stale_jobs(repo, settings)

    assert {j.id: j.error_code for j in reaped} == {
        running.id: "heartbeat_timeout",
        queued.id: "queue_timeout",
    }
    assert "no heartbeat for 60 s" in repo.get_job(running.id).error
    assert repo.get_job(running.id).retryable is True
    assert repo.get_job(fresh.id).status == "running"
    assert lifecycle.reap_stale_jobs(repo, settings) == []  # nothing left to reap


def test_failing_a_finished_job_changes_nothing(repo):
    exp = repo.create_experiment(title="a")
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment")
    repo.claim_job(job.id)
    repo.finish_job(job.id, status="succeeded")

    assert lifecycle.fail_job(repo, job, error="late") is None
    assert repo.get_job(job.id).status == "succeeded"
    assert repo.list_events(exp.id) == []


def test_unknown_phase_names_are_caught(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)

    def bad_phase(ctx, experiment, trial):
        ctx.phase("daydreaming")

    stages.analyze_scene = bad_phase

    done = run(job)

    assert done.error_code == "internal_error" and "unknown phase" in done.error


def test_spawn_failure_fails_the_job_so_it_can_be_retried(repo, spawner):
    exp = repo.create_experiment(title="a")
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment")
    spawner.fail_with = RuntimeError("modal is down")

    result = dispatch(repo, spawner, job)

    assert (result.status, result.error_code, result.retryable) == ("failed", "spawn_failed", True)
    assert "modal is down" in result.error
    assert repo.active_job(exp.id) is None


def test_dispatch_records_the_modal_call_id(repo, spawner, db):
    exp = repo.create_experiment(title="a")
    job = repo.create_job(experiment_id=exp.id, kind="analyze_experiment")

    result = dispatch(repo, spawner, job)

    assert result.status == "queued"
    assert db.fetch_one("select call_id from jobs where id = %s", (job.id,))["call_id"] == "call-1"


# --- prove, revise, rerun, export check -----------------------------------------------------
def completed_experiment(repo, paths, run, fake_stages):
    exp, _, job = start_analysis(repo, paths)
    run(job)
    fake_stages.calls.clear()
    return repo.get_experiment(exp.id)


def test_prove_measures_the_new_clip_and_saves_a_ghost(repo, paths, run, fake_stages):
    exp = completed_experiment(repo, paths, run, fake_stages)
    prove = ready_trial(repo, paths, exp, role="prove")
    job = repo.create_job(
        experiment_id=exp.id, kind="prove_trial", params={"trial_id": str(prove.id)}
    )

    done = run(job)

    assert done.status == "succeeded"
    assert fake_stages.calls == [
        "track_objects",
        "calibrate",
        "extract_trajectory",
        "predict_ghost",
    ]
    [ghost] = repo.list_ghosts(exp.id)
    assert (ghost.version, ghost.trial_id, ghost.divergence_onset_s) == (1, prove.id, 2.8)
    assert paths.experiment_dir(exp.id).joinpath(ghost.video_path).read_bytes() == b"ghost-video"
    assert [t.trial_id for t in repo.list_tracks([prove.id])] == [prove.id]
    assert repo.get_trial(prove.id).calibration is not None
    verdict = [e for e in repo.list_events(exp.id) if e.kind == "ghost"][0]
    assert verdict.title == "Ghost verdict" and "2.80 s" in verdict.body_md
    assert verdict.data["max_error_mm"] == 14.0
    assert repo.get_experiment(exp.id).status == "complete"  # proving doesn't reopen the analysis


def test_prove_without_a_scene_fails_cleanly(repo, paths, run, fake_stages):
    exp = repo.create_experiment(title="a")
    prove = ready_trial(repo, paths, exp, role="prove")
    job = repo.create_job(
        experiment_id=exp.id, kind="prove_trial", params={"trial_id": str(prove.id)}
    )

    done = run(job)

    assert (done.status, done.error_code, done.retryable) == ("failed", "no_scene", False)


def test_a_failed_prove_leaves_the_finished_analysis_alone(repo, paths, run, fake_stages):
    exp = completed_experiment(repo, paths, run, fake_stages)
    prove = ready_trial(repo, paths, exp, role="prove")
    job = repo.create_job(
        experiment_id=exp.id, kind="prove_trial", params={"trial_id": str(prove.id)}
    )
    fake_stages.fail_on["predict_ghost"] = RuntimeError("render crashed")

    done = run(job)

    assert done.status == "failed"
    assert repo.get_experiment(exp.id).status == "complete"
    assert repo.list_ghosts(exp.id) == []


def test_revise_logs_the_diagnosis_and_adds_a_ghost_only_when_adopted(
    repo, paths, run, fake_stages
):
    exp = completed_experiment(repo, paths, run, fake_stages)
    prove = ready_trial(repo, paths, exp, role="prove")
    ghost = repo.add_ghost_render(GhostRender(experiment_id=exp.id, trial_id=prove.id))
    job = repo.create_job(experiment_id=exp.id, kind="revise", params={"ghost_id": str(ghost.id)})

    run(job)

    assert len(repo.list_ghosts(exp.id)) == 1
    [event] = [e for e in repo.list_events(exp.id) if e.kind == "revision"]
    assert event.title == "The revision didn't help"
    assert event.body_md == "Damping was underestimated."

    from ghostline.jobs.stage_types import GhostResult, RevisionResult

    def adopt(ctx, experiment, ghost):
        v2 = ctx.repo.next_ghost_version(experiment.id)
        return RevisionResult(
            diagnosis_md="Add damping.",
            adopted=True,
            revised_model=ModelRecord(experiment_id=experiment.id, id="P2", name="Damped"),
            ghost=GhostResult(
                version=v2, video_path="ghost/v2.mp4", track_path="ghost/v2_track.json"
            ),
        )

    stages.diagnose_revision = adopt
    run(repo.create_job(experiment_id=exp.id, kind="revise", params={}))  # newest ghost by default

    assert [g.version for g in repo.list_ghosts(exp.id)] == [1, 2]
    assert repo.get_model(exp.id, "P2").name == "Damped"
    assert [e.title for e in repo.list_events(exp.id) if e.kind == "revision"][
        -1
    ] == "Revision adopted"


def test_revise_with_no_ghost_fails_without_retry(repo, paths, run, fake_stages):
    exp = completed_experiment(repo, paths, run, fake_stages)
    job = repo.create_job(experiment_id=exp.id, kind="revise", params={})

    done = run(job)

    assert (done.error_code, done.retryable) == ("no_ghost", False)


def test_rerun_inserts_a_new_fit_and_logs_it(repo, paths, run, fake_stages):
    exp = completed_experiment(repo, paths, run, fake_stages)
    params = {"model_id": "P1", "split_pct": 0.5, "initial_guesses": {"g": 9.0}}
    job = repo.create_job(experiment_id=exp.id, kind="rerun", params=params)

    done = run(job)

    assert done.status == "succeeded" and done.phase == "fitting"
    fits = repo.list_fits(exp.id)
    assert len(fits) == 2 and fits[-1].split == {"pct": 0.5}
    [event] = [e for e in repo.list_events(exp.id) if e.title.startswith("Re-ran")]
    assert event.title == "Re-ran Nonlinear pendulum with a 50% split"
    assert event.kind == "fit" and event.data["initial_guesses"] == {"g": 9.0}
    assert {(r.kind, r.id) for r in event.refs} == {("model", "P1"), ("fit", str(fits[-1].id))}


def test_export_check_stores_the_replay_result(repo, paths, run, fake_stages):
    exp = completed_experiment(repo, paths, run, fake_stages)

    run(repo.create_job(experiment_id=exp.id, kind="export_verify"))

    assert repo.get_experiment(exp.id).replay_check["passed"] is True
    assert [e.title for e in repo.list_events(exp.id)][-1] == "Reproduced in PyBullet"


def test_volume_commit_errors_fail_the_job(repo, paths, settings, fake_stages):
    exp, _, job = start_analysis(repo, paths)

    class BrokenVolume(NullVolume):
        def commit(self):
            raise OSError("volume busy")

    done = run_job(job.id, repo=repo, paths=paths, volume=BrokenVolume(), settings=settings)

    assert done.status == "failed" and "volume busy" in done.error


def test_two_jobs_for_different_experiments_run_side_by_side(repo, paths, run, fake_stages):
    jobs = [start_analysis(repo, paths)[2] for _ in range(2)]
    results = []

    threads = [threading.Thread(target=lambda j=j: results.append(run(j))) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [r.status for r in results] == ["succeeded", "succeeded"]
