"""The spin-down run of Part 13's Done-when, replayed from a live recording.

A real Muse Wonder analysed synthetic spin-down data (viscous plus Coulomb friction, and a user
who expects an exponential decay). Its calls are saved in fixtures/agent/spin_down/cache and the
harness results (and the plots they carry) in fixtures/agent/spin_down/harness, so this test
replays the whole run offline with the same numbers on every machine. To record it again (this
spends about a dollar of Wonder):

    GHOSTLINE_RECORD=1 uv run --env-file .env pytest backend/tests/test_agent_replay.py -s
"""

import json
import time
from pathlib import Path

from fixtures.agent.support import (
    RECORDING,
    make_world,
    pin_training_plot,
    recorded_clients,
    runner_factory,
)

from ghostline.agent.clients import use_clients
from ghostline.agent.numbers import matched_numbers, sorted_known
from ghostline.agent.runners import use_runner_factory
from ghostline.agent.session import load_session
from ghostline.agent.stage import run_physicist

FOLDER = Path(__file__).parent / "fixtures" / "agent" / "spin_down"
DESCRIPTION = "A spinner I flicked. I think it slows down exponentially, like a paddle in honey."


def test_wonder_names_a_favorite_and_the_held_out_data_refutes_it(
    repo, paths, settings, monkeypatch
):
    pin_training_plot(monkeypatch, FOLDER)
    world = make_world(repo, paths, settings, description=DESCRIPTION)
    meta, reviewer, runtime, store = recorded_clients(FOLDER)

    started = time.monotonic()
    with use_clients(meta, reviewer), use_runner_factory(runner_factory(FOLDER / "harness")):
        conclusion = run_physicist(
            world.ctx,
            world.experiment,
            world.trial,
            world.scene,
            world.trajectory,
            world.calibration,
        )
    wall_s = time.monotonic() - started

    session = load_session(paths, world.experiment.id)
    assert session.mode == "wonder" and conclusion.winner_id == "S2"
    events = repo.list_events(world.experiment.id)
    kinds = [e.kind for e in events]

    # the prediction is recorded verbatim, before any fit
    prediction = next(e for e in events if e.kind == "prediction")
    assert prediction.actor == "wonder" and prediction.data["pinned"] == "prediction"
    assert session.favorite_why in prediction.body_md
    favorite = session.models[session.favorite_id]
    assert favorite.refuting_signature in prediction.body_md
    assert kinds.index("prediction") < kinds.index("fit")

    # every losing model is rejected on a held-out number from its own fit
    fits = {f.model_id: f for f in repo.list_fits(world.experiment.id)}
    rejections = [e for e in events if e.kind == "rejection"]
    assert {e.data["model_id"] for e in rejections} == set(session.models) - {"S2"}
    for event in rejections:
        known = sorted_known([fits[event.data["model_id"]].heldout_rmse])
        assert matched_numbers(event.body_md, known), event.body_md
    if favorite.id != "S2":
        assert favorite.id in {e.data["model_id"] for e in rejections}
    assert [e.data.get("pinned") for e in rejections].count("rejection") == 1

    # the harness numbers, not Wonder's, are in the conclusion
    truth = {"b": 0.15, "c": 1.5}
    for q in conclusion.quantities:
        if q.name in truth:
            assert q.ci_low < truth[q.name] < q.ci_high
    assert conclusion.confidence == "medium"  # a clean result, but not yet reviewed

    cost = sum(r.cost_usd for r in runtime.store.records())
    if RECORDING:
        stats = {
            "wall_s": round(wall_s, 1),
            "cost_usd": round(cost, 4),
            "wonder_calls": len(runtime.store.records()),
            "models": sorted(session.models),
            "favorite": session.favorite_id,
            "winner": conclusion.winner_id,
        }
        (FOLDER / "run.json").write_text(json.dumps(stats, indent=1))
        print("RUN", stats)
    else:
        assert all(r.cache_hit for r in store.records()) and cost == 0  # replay is free
