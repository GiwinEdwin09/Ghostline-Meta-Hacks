"""Library-only mode (Part 13, Failures): the analysis finishes with Wonder disabled."""

import dataclasses

from fixtures.agent.support import make_world

from ghostline.agent.session import load_session
from ghostline.agent.stage import rerun_model, run_physicist


def test_it_completes_with_wonder_disabled(repo, paths, settings):
    off = dataclasses.replace(settings, physicist_model="off")
    world = make_world(repo, paths, off)

    conclusion = run_physicist(
        world.ctx, world.experiment, world.trial, world.scene, world.trajectory, world.calibration
    )

    # the truth is viscous plus Coulomb; the held-out comparison finds it
    assert conclusion.winner_id == "S2"
    names = {q.name: q for q in conclusion.quantities}
    assert names["c"].ci_low < 1.5 < names["c"].ci_high
    assert names["b"].ci_low < 0.15 < names["b"].ci_high
    assert {r["id"] for r in conclusion.rejected} == {"S0", "S1", "S3"}
    assert "template" in conclusion.narrative_md
    assert conclusion.confidence == "medium"  # no review yet caps a clean result at medium

    events = repo.list_events(world.experiment.id)
    assert any("Wonder was unavailable" in e.title for e in events)
    assert [e.kind for e in events].count("fit") == 4
    assert {e.actor for e in events if e.kind == "rejection"} == {"harness"}
    pinned = [e.data.get("pinned") for e in events if e.data.get("pinned")]
    assert pinned == ["rejection"]  # no prediction was made, so none is pinned

    models = {m.id: m for m in repo.list_models(world.experiment.id)}
    assert models["S2"].status == "winner" and models["S0"].status == "rejected"
    assert set(models["S2"].params_spec) >= {"b", "c", "omega0"}
    fits = repo.list_fits(world.experiment.id)
    assert {f.model_id for f in fits} == {"S0", "S1", "S2", "S3"}
    assert len([f for f in fits if f.model_id == "S2"]) == 2  # selection fit and final fit
    final = [f for f in fits if f.model_id == "S2"][-1]
    assert final.split["kind"] == "final" and final.heldout_rmse is not None
    assert (paths.experiment_dir(world.experiment.id) / final.plot_paths["fit"]).is_file()

    session = load_session(paths, world.experiment.id)
    assert session.mode == "library_only" and session.selection.selected_id == "S2"


def test_a_rerun_refits_one_model_with_a_new_split(repo, paths, settings):
    off = dataclasses.replace(settings, physicist_model="off")
    world = make_world(repo, paths, off)
    run_physicist(
        world.ctx, world.experiment, world.trial, world.scene, world.trajectory, world.calibration
    )
    model = repo.get_model(world.experiment.id, "S2")

    fit = rerun_model(world.ctx, world.experiment, model, 0.8, {})
    guessed = rerun_model(world.ctx, world.experiment, model, 0.5, {"omega0": 38.0})

    assert fit.split["pct"] == 0.8 and fit.heldout_rmse and fit.heldout_ci
    assert abs({p.name: p.value for p in fit.params}["c"] - 1.5) < 0.2
    assert guessed.split["initial_guesses"] == {"omega0": 38.0} and guessed.heldout_rmse
    assert (paths.experiment_dir(world.experiment.id) / fit.plot_paths["fit"]).is_file()
