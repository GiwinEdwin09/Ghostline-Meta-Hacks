"""What the web reads: the experiment view with fits, confidence reasons, review and ghosts; the
explain route; the benchmark table; the media paths that are never served (Parts 18, 24, 25, 28)."""

from types import SimpleNamespace

import pytest

from ghostline.llm import DailyBudgetExceeded
from ghostline.store.core_schemas import ParamSpec
from ghostline.store.records import (
    FitParam,
    FitRecord,
    GhostRender,
    ModelRecord,
    ReviewIssue,
    ReviewRecord,
)

CONCLUSION = {
    "winner_id": "S2",
    "quantities": [
        {"name": "c", "value": 1.5, "ci_low": 1.3, "ci_high": 1.7, "unit": "rad/s^2"},
    ],
    "rejected": [{"id": "S0", "reason": "worse held-out error", "evidence": "0.4 vs 0.1"}],
    "confidence": "low",
    "caveats": ["short clip"],
    "narrative_md": "The spinner slows linearly. More text.",
}


@pytest.fixture
def analysed(client, repo):
    """An experiment as the physicist leaves it: models, fits, a Confidence event, a review."""
    eid = client.post("/api/experiments", json={"title": "Spinner"}).json()["id"]
    trial = repo.create_trial(experiment_id=eid, role="fit")
    spec = {
        "c": ParamSpec(init=1, lower=0, upper=2, unit="rad/s^2", kind="physical"),
        "phi0": ParamSpec(init=0, lower=-1, upper=1, unit="rad", kind="initial_condition"),
    }
    for model_id, name in (("S0", "Viscous"), ("S2", "Coulomb and viscous")):
        repo.upsert_model(
            ModelRecord(
                experiment_id=eid, id=model_id, name=name, params_spec=spec, library_ref=model_id,
                prior_rank=1 if model_id == "S0" else 2,
                status="rejected" if model_id == "S0" else "winner",
            )
        )  # fmt: skip
    rmse = {
        "S0": {"0.50": 0.4, "0.65": 0.5, "0.80": 0.5},
        "S2": {"0.50": 0.1, "0.65": 0.2, "0.80": 0.6},
    }
    for model_id, scores in rmse.items():
        repo.insert_fit(
            FitRecord(
                experiment_id=eid, model_id=model_id, trial_id=trial.id,
                split={"pct": 0.65, "kind": "selection"},
                params=[FitParam(name="c", value=1.5, ci_low=1.3, ci_high=1.7, unit="rad/s^2"),
                        FitParam(name="phi0", value=1.0, ci_low=0.9, ci_high=1.1, unit="rad")],
                train_rmse=0.01, heldout_rmse=scores["0.65"], heldout_ci=[0.1, 0.9],
                plot_paths={"fit": f"fits/{model_id}/fit.png"},
                diagnostics={"reduced_chi2": 1.2, "nrmse": 0.03, "systematic_residual": True,
                             "heldout_rmse_by_split": scores},
            )
        )  # fmt: skip
    repo.insert_fit(
        FitRecord(
            experiment_id=eid, model_id="S2", trial_id=trial.id, split={"kind": "final"},
            params=[FitParam(name="c", value=1.5, ci_low=1.3, ci_high=1.7, unit="rad/s^2")],
            diagnostics={"budget": {"c": {"statistical": 0.05, "scale_rel": 0.01}}},
        )
    )  # fmt: skip
    repo.update_experiment(
        eid, status="complete", conclusion=CONCLUSION, confidence="low", winner_model_id="S2",
        review_status="contested",
    )  # fmt: skip
    repo.append_event(
        eid, actor="harness", kind="acceptance", title="Confidence: low",
        data={"cant_tell": True, "fixes": ["Film the whole spin-down."],
              "reasons": [{"code": "tied", "level": "low", "text": "S0 and S2 are tied."},
                          {"code": "flag", "level": "medium", "text": "One flag."}]},
    )  # fmt: skip
    repo.insert_review(
        ReviewRecord(
            experiment_id=eid, round=1, verdict="challenge", summary="Too bold.",
            issues=[ReviewIssue(id="i1", severity="major", claim="c", evidence="e",
                                requested_action="refit", detail="d")],
            response={"response_md": "Refitted."},
        )
    )  # fmt: skip
    return eid


def test_the_view_joins_fits_the_budget_and_the_confidence_reasons(client, analysed):
    view = client.get(f"/api/experiments/{analysed}").json()

    by_id = {m["id"]: m for m in view["models"]}
    assert by_id["S2"]["fit"]["heldout_rmse"] == 0.2  # the selection fit, not the final refit
    assert by_id["S2"]["fit"]["diagnostics"]["systematic_residual"] is True
    assert by_id["S2"]["fit"]["plot_urls"]["fit"] == f"{analysed}/fits/S2/fit.png"
    assert [p["kind"] for p in by_id["S2"]["fit"]["params"]] == ["physical", "initial_condition"]
    assert by_id["S2"]["fit"]["params"][1]["at_bound"] is True  # phi0 = 1.0 is at its upper bound
    assert by_id["S2"]["status"] == "winner" and by_id["S0"]["rejection_reason"] is None
    # S2 predicted best at 50% and 65%, S0 at 80%
    assert (by_id["S2"]["fit"]["splits_won"], by_id["S0"]["fit"]["splits_won"]) == (2, 1)

    conclusion = view["conclusion"]
    assert conclusion["confidence"] == "low"
    assert conclusion["confidence_reasons"] == ["S0 and S2 are tied.", "One flag."]
    assert conclusion["cant_tell"] == {
        "reason": "S0 and S2 are tied.",
        "fixes": ["Film the whole spin-down."],
    }
    budget = {b["label"]: b["sigma"] for b in conclusion["quantities"][0]["budget"]}
    assert budget == {"Statistical": 0.05, "Scale rel": 0.01}
    assert conclusion["rejected"] == [
        {"id": "S0", "reason": "worse held-out error", "evidence": "0.4 vs 0.1"}
    ]
    assert (
        view["headline"] == "c = 1.5 ± 0.2 rad/s^2" and view["winner_name"] == "Coulomb and viscous"
    )
    assert view["blurb"] == "The spinner slows linearly."
    assert view["export_formats"] == ["json", "pybullet"]  # S2 has simulator parameters

    [round1] = view["review"]["rounds"]
    assert view["review"]["status"] == "contested" and round1["response_md"] == "Refitted."
    assert round1["issues"][0]["requested_action"] == "refit"


def test_ghosts_carry_the_threshold_and_the_revision(client, repo, analysed, paths):
    trial = repo.list_trials(analysed)[0]
    track = paths.ghost_track(analysed, 2)
    track.parent.mkdir(parents=True, exist_ok=True)
    track.write_text('{"t": [0.0, 2.0, 5.0], "tau_mm": 6.5, "launch_window_s": 0.5}')
    repo.add_ghost_render(GhostRender(
        experiment_id=analysed, trial_id=trial.id, model_id="S2", video_path="ghost/v1.mp4",
        track_path="ghost/v1_track.json", threshold_mm=4.0, predicted_ahead_s=3.0,
        divergence_onset_s=2.0, max_error_mm=9.0,
    ))  # fmt: skip
    second = repo.add_ghost_render(GhostRender(
        experiment_id=analysed, trial_id=trial.id, model_id="S2", video_path="ghost/v2.mp4",
        track_path="ghost/v2_track.json",
    ))  # fmt: skip
    repo.set_ghost_revision(
        second.id, {"diagnosis": "Damping.", "adopted": True, "note": "Better."}
    )

    first, later = client.get(f"/api/experiments/{analysed}").json()["ghosts"]

    assert (first["threshold_mm"], first["predicted_ahead_s"]) == (4.0, 3.0)
    assert first["track_url"] == f"{analysed}/ghost/v1_track.json" and first["revision"] is None
    # v2 has no stored threshold, so it is read from its track file
    assert (later["threshold_mm"], later["predicted_ahead_s"]) == (6.5, 4.5)
    assert later["model_name"] == "Coulomb and viscous"
    assert later["revision"] == {"diagnosis": "Damping.", "adopted": True, "note": "Better."}


def test_events_carry_media_urls_and_a_failed_clip_that_cannot_be_measured_is_blocked(
    client, repo, analysed
):
    from ghostline.store.records import Attachment

    repo.append_event(
        analysed, actor="harness", kind="observation", title="Plot",
        attachments=[Attachment(path="calib/calib_overlay.png", kind="image", caption="Grid")],
    )  # fmt: skip
    page = client.get(f"/api/experiments/{analysed}/events", params={"after": 1}).json()
    assert page["events"][-1]["attachments"] == [
        {"kind": "image", "url": f"{analysed}/calib/calib_overlay.png", "caption": "Grid"}
    ]

    other = client.post("/api/experiments", json={"title": "Blank wall"}).json()["id"]
    job = repo.create_job(experiment_id=other, kind="analyze_experiment")
    repo.claim_job(job.id)
    repo.update_experiment(other, status="failed", error="nothing to measure")
    repo.finish_job(job.id, status="failed", error="nothing", error_code="cannot_measure")
    assert client.get(f"/api/experiments/{other}").json()["status"] == "blocked"


# --- explain --------------------------------------------------
class FakeMeta:
    calls = 0

    def __init__(self, runtime):
        pass

    def respond(self, **kwargs):
        FakeMeta.calls += 1
        FakeMeta.last = kwargs
        return SimpleNamespace(output_text=" Gravity is weaker on Mars, so the swing is slower. ")


def test_explain_asks_wonder_once_and_caches_the_answer(client, analysed, monkeypatch):
    monkeypatch.setattr("ghostline.api.explain.MetaClient", FakeMeta)
    FakeMeta.calls = 0
    url = f"/api/experiments/{analysed}/explain"
    body = {"summary": "Pendulum on Mars", "numbers": {"period_ratio": 1.63, "g": "3.71"}}

    first = client.post(url, json=body)
    again = client.post(url, json={"numbers": body["numbers"], "summary": body["summary"]})
    other = client.post(url, json={**body, "numbers": {"period_ratio": 2.0}})

    assert first.json() == {
        "text": "Gravity is weaker on Mars, so the swing is slower.",
        "cached": False,
    }
    assert again.json()["cached"] is True and FakeMeta.calls == 2  # the first and the other
    assert other.json()["cached"] is False
    assert (
        "period_ratio: 1.63" in FakeMeta.last["input"]
        or "period_ratio: 2.0" in FakeMeta.last["input"]
    )
    assert "never state, estimate or invent a number" in FakeMeta.last["instructions"]


def test_explain_needs_the_passcode_a_result_and_a_budget(
    client, anon, repo, analysed, monkeypatch
):
    url = f"/api/experiments/{analysed}/explain"
    assert anon.post(url, json={"summary": "x"}).status_code == 401
    draft = client.post("/api/experiments", json={"title": "x"}).json()["id"]
    assert (
        client.post(f"/api/experiments/{draft}/explain", json={"summary": "x"}).status_code == 409
    )

    class OverBudget(FakeMeta):
        def respond(self, **kwargs):
            raise DailyBudgetExceeded(20, 20)

    monkeypatch.setattr("ghostline.api.explain.MetaClient", OverBudget)
    assert client.post(url, json={"summary": "x"}).json()["code"] == "daily_budget_reached"
    assert client.post(url, json={"summary": "x" * 3000}).status_code == 422


# --- media --------------------------------------------------
def test_media_refuses_the_physicists_session_and_raw_uploads(client, paths, analysed):
    folder = paths.experiment_dir(analysed)
    for rel in ("agent/session.json", "trials/t1/original.mov", "trials/t1/original.bin"):
        target = folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"secret")
        assert client.get(f"/api/media/{analysed}/{rel}").status_code == 404, rel
    (folder / "fits").mkdir()
    (folder / "fits" / "fit.png").write_bytes(b"png")
    assert client.get(f"/api/media/{analysed}/fits/fit.png").status_code == 200
    # a look-alike name is fine; only the folder and the raw upload are internal
    (folder / "fits" / "agent_plot.png").write_bytes(b"png")
    assert client.get(f"/api/media/{analysed}/fits/agent_plot.png").status_code == 200
    assert client.get(f"/api/media/{analysed}/..%2Fagent%2Fsession.json").status_code == 404


# --- benchmark --------------------------------------------------
def test_the_benchmark_scores_coverage_selection_and_failure_flags(client, repo, anon):
    def experiment(title, quantities, winner, confidence, status="complete"):
        eid = client.post("/api/experiments", json={"title": title}).json()["id"]
        repo.update_experiment(
            eid, status=status, winner_model_id=winner, confidence=confidence,
            conclusion={"winner_id": winner, "quantities": quantities, "rejected": [],
                        "confidence": confidence, "caveats": [], "narrative_md": ""},
        )  # fmt: skip
        return eid

    g = lambda v, lo, hi: [{"name": "g", "value": v, "ci_low": lo, "ci_high": hi, "unit": "m/s^2"}]  # noqa: E731
    hit = experiment("hit", g(9.8, 9.6, 10.0), "FF0", "high")
    miss = experiment("miss", g(9.0, 8.9, 9.1), "FF2", "medium")
    bad = experiment("bad", g(5.0, 1.0, 9.0), "FF0", "low")
    base = {"system": "free_fall", "quantity": "g", "truth_value": 9.8, "truth_sigma": 0.01,
            "unit": "m/s^2"}  # fmt: skip
    client.put(f"/api/benchmark/{hit}", json={**base, "clip_id": "ff-1", "expected_model": "FF0"})
    client.put(f"/api/benchmark/{miss}", json={**base, "clip_id": "ff-2", "expected_model": "FF0"})
    client.put(
        f"/api/benchmark/{bad}",
        json={**base, "clip_id": "bad-1", "is_failure_case": True, "failure_expected": "tilt"},
    )

    table = anon.get("/api/benchmark").json()
    rows = {r["clip_id"]: r for r in table["rows"]}

    assert rows["ff-1"]["covered"] is True and rows["ff-2"]["covered"] is False
    assert rows["ff-2"]["z"] < -3 and rows["ff-1"]["model_correct"] is True
    assert rows["ff-2"]["model_correct"] is False
    assert rows["bad-1"]["is_failure_case"] and rows["bad-1"]["failure_flagged"] is True
    assert rows["bad-1"]["measured"] is None  # a failure case isn't scored on its number
    summary = table["summary"]
    assert (summary["n_clips"], summary["coverage"], summary["model_selection_accuracy"]) == (
        3,
        0.5,
        0.5,
    )
    assert summary["failure_cases_flagged"] == "1/1"
