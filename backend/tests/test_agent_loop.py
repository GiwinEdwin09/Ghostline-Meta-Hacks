"""The Wonder loop on a scripted Wonder: phases, budgets, validation, notebook and fallbacks.

The scripted model reads the numbers the harness prints and quotes them back, as a careful Wonder
would. The harness is real (in-process); no model is called.
"""

import dataclasses
import json

import pytest
from fixtures.agent.support import (
    call_item,
    conclude_message_json,
    find_outputs,
    make_world,
    message_output,
    scripted_clients,
    tool_texts,
)

from ghostline.agent.clients import use_clients
from ghostline.agent.session import load_session
from ghostline.agent.stage import run_physicist
from ghostline.llm import ModelNotFoundError

PROPOSAL = {
    "favorite_id": "S0",
    "favorite_why": "A spinner slowed by air should lose speed in proportion to speed.",
    "models": [
        {
            "id": ref,
            "name": ref,
            "mechanism": "m",
            "equations_latex": "x",
            "library_ref": ref,
            "code": None,
            "params": [],
            "observables": [],
            "state": [],
            "constants": [],
            "prior_rank": rank,
            "refuting_signature": sig,
        }
        for rank, (ref, sig) in enumerate(
            [
                ("S0", "If the spin rate falls on a straight line, not a curve, S0 is wrong."),
                ("S1", "If the spin never stops linearly, S1 is wrong."),
                ("S2", "If a single friction term fits as well, S2 is wrong."),
            ],
            start=1,
        )
    ],
}


def table_of(items):
    return next(o for o in find_outputs(items) if "models" in o and "verdict" in o)


def conclusion_args(items, narrative, *, bad_quantity=False):
    final = conclude_message_json(items)
    table = table_of(items)
    rmse = {r["model_id"]: r["heldout_rmse"] for r in table["models"]}
    quantities = [
        {
            "name": q["name"],
            "value": q["value"],
            "ci_low": q["ci95"][0],
            "ci_high": q["ci95"][1],
            "unit": q["unit"],
        }
        for q in final["final_fit"]
    ]
    if bad_quantity:
        quantities[0]["value"] *= 1.5
    return {
        "winner_id": final["winner"],
        "quantities": quantities,
        "rejected": [
            {
                "id": i,
                "reason": "It forecasts the held-out motion worse.",
                "evidence": f"Held-out RMSE {rmse_i} rad.",
            }
            for i, rmse_i in rmse.items()
            if i != final["winner"]
        ],
        "confidence": "high",
        "caveats": [],
        "narrative_md": narrative,
        "issue_responses": [],
    }


def happy_policy(items, n):
    if n == 1:
        return [call_item("propose_models", PROPOSAL, n)]
    if n == 2:
        return [call_item("run_fits", {"model_ids": ["S0", "S1", "S2"], "split": "selection"}, n)]
    if n == 3:
        return [message_output("Both friction terms are needed; one alone forecasts badly.")]
    if n == 4:  # an invented number, and a quantity that doesn't match the harness
        args = conclusion_args(items, "Friction is 7.77 units.", bad_quantity=True)
        return [call_item("submit_conclusion", args, n)]
    args = conclusion_args(items, "Two friction terms, constant and speed-dependent, explain it.")
    return [call_item("submit_conclusion", args, n)]


@pytest.fixture
def run_loop(repo, paths, settings, tmp_path):
    def go(policy, **overrides):
        world = make_world(repo, paths, dataclasses.replace(settings, **overrides))
        meta, reviewer, sdk, _ = scripted_clients(policy, tmp_path)
        with use_clients(meta, reviewer):
            conclusion = run_physicist(
                world.ctx,
                world.experiment,
                world.trial,
                world.scene,
                world.trajectory,
                world.calibration,
            )
        return world, conclusion, sdk

    return go


def test_the_loop_runs_the_phases_in_order_and_writes_the_notebook(run_loop, repo, paths):
    world, conclusion, sdk = run_loop(happy_policy)

    assert conclusion.winner_id == "S2" and len(sdk.calls) == 5
    efforts = [c["reasoning"]["effort"] for c in sdk.calls]
    assert efforts == ["high", "low", "medium", "high", "high"]
    assert all(c["max_output_tokens"] >= 8000 for c in sdk.calls)
    assert len({json.dumps(c["tools"]) for c in sdk.calls}) == 1  # one tool list, for the cache
    assert len({c["instructions"] for c in sdk.calls}) == 1
    assert sdk.calls[0]["instructions"].startswith("You are the physicist in Ghostline")
    assert "S3" in sdk.calls[0]["instructions"]  # the library catalog is in the prefix

    # an invented number and a wrong quantity are sent back, then the corrected call is accepted
    problems = json.loads(tool_texts(sdk.calls[4]["input"])[-1])["problems"]
    assert any("7.77" in p for p in problems)
    assert any("does not match the harness" in p for p in problems)

    events = repo.list_events(world.experiment.id)
    kinds = [e.kind for e in events]
    assert kinds.count("hypothesis") == 3 and kinds.count("fit") == 3
    prediction = next(e for e in events if e.kind == "prediction")
    assert prediction.actor == "wonder" and prediction.data["pinned"] == "prediction"
    assert PROPOSAL["favorite_why"] in prediction.body_md
    assert "If the spin rate falls on a straight line" in prediction.body_md  # verbatim
    assert kinds.index("prediction") < kinds.index("fit")  # recorded before any fit
    rejection = next(e for e in events if e.kind == "rejection")
    assert rejection.data["pinned"] == "rejection" and "Held-out RMSE" in rejection.body_md
    assert any(e.kind == "observation" and e.actor == "wonder" for e in events)
    assert [e.title for e in events if e.kind == "acceptance"][-1].startswith("Confidence:")

    # the rules, not Wonder, set the confidence: it asked for high, no review yet caps it
    assert conclusion.confidence == "medium"
    assert {m.id: m.status for m in repo.list_models(world.experiment.id)} == {
        "S0": "rejected",
        "S1": "rejected",
        "S2": "winner",
    }
    session = load_session(paths, world.experiment.id)
    assert session.mode == "wonder" and session.favorite_id == "S0"
    assert any(i.get("type") == "function_call" for i in session.history)


def test_wonder_never_gets_held_out_data(run_loop, paths):
    def policy(items, n):
        steps = {
            1: ("get_training_data", {"t_start": None, "t_end": None, "max_points": 400}),
            2: ("propose_models", PROPOSAL),
            3: ("run_fits", {"model_ids": ["S0", "S1", "S2"], "split": "selection"}),
            4: ("view_plot", {"plot_id": "S0.fit"}),  # the reviewer's plot: held-out points
            5: ("view_plot", {"plot_id": "S0.fit_train"}),
        }
        if n in steps:
            return [call_item(*steps[n], n)]
        if n == 6:
            return [message_output("Both friction terms are needed.")]
        return [call_item("submit_conclusion", conclusion_args(items, "Two friction terms."), n)]

    world, conclusion, sdk = run_loop(policy)

    t_split = load_session(paths, world.experiment.id).t_split
    texts = tool_texts(sdk.calls[-1]["input"])
    data = json.loads(texts[0])
    assert max(data["series"]["t"]) <= t_split < world.case.t[-1] - 1
    assert not any("heldout_errors" in t or "heldout_draws" in t for t in texts)
    assert "Unknown plot id 'S0.fit'" in texts[3]  # no plot with held-out points
    plot = next(
        i
        for i in sdk.calls[-1]["input"]
        if i.get("type") == "function_call_output" and i["call_id"] == "call_5"
    )
    assert any(p.get("type") == "input_image" for p in plot["output"])  # image inside the output
    assert conclusion.winner_id == "S2"


def test_a_failing_wonder_finishes_in_library_only_mode(run_loop, repo):
    def policy(items, n):
        raise ModelNotFoundError("muse-wonder-1.0 is not available to this key")

    world, conclusion, sdk = run_loop(policy)

    assert conclusion.winner_id == "S2" and "template" in conclusion.narrative_md
    titles = [e.title for e in repo.list_events(world.experiment.id)]
    assert any("Wonder was unavailable" in t for t in titles)


CUSTOM = {
    "id": "W1",
    "name": "Exponential spin-down",
    "mechanism": "Drag proportional to spin rate.",
    "equations_latex": "\\dot\\omega=-b\\omega",
    "library_ref": None,
    "code": "",
    "params": [
        {"name": "phi0", "init": 0.0, "lower": -10.0, "upper": 10.0, "unit": "rad",
         "kind": "initial_condition"},
        {"name": "omega0", "init": 30.0, "lower": 0.0, "upper": 100.0, "unit": "rad/s",
         "kind": "initial_condition"},
        {"name": "b", "init": 0.2, "lower": 0.01, "upper": 5.0, "unit": "1/s", "kind": "physical"},
    ],
    "observables": ["phi"],
    "state": [],
    "constants": [],
    "prior_rank": 4,
    "refuting_signature": "If the spin rate falls linearly, W1 is wrong.",
}  # fmt: skip
GOOD_CODE = (
    "def closed_form(t, p):\n"
    "    return np.array([p['phi0'] + p['omega0'] * (1 - np.exp(-p['b'] * t)) / p['b']])\n"
)


def test_custom_code_is_checked_then_fitted_through_the_runner(run_loop, repo):
    def proposal(code):
        models = [*PROPOSAL["models"][:2], {**CUSTOM, "code": code}]
        return {**PROPOSAL, "models": models}

    def policy(items, n):
        if n == 1:  # reaches for os: refused before anything runs
            return [call_item("propose_models", proposal("import os\n" + GOOD_CODE), n)]
        if n == 2:
            return [call_item("propose_models", proposal(GOOD_CODE), n)]
        if n == 3:
            return [
                call_item("run_fits", {"model_ids": ["S0", "S1", "W1"], "split": "selection"}, n)
            ]
        if n == 4:
            return [message_output("The exponential fits as well as the library one.")]
        return [call_item("submit_conclusion", conclusion_args(items, "It is a tie."), n)]

    world, conclusion, sdk = run_loop(policy)

    first = json.loads(tool_texts(sdk.calls[1]["input"])[0])
    refused = next(m for m in first["models"] if m["id"] == "W1")
    assert not refused["accepted"] and "import of 'os' is not allowed" in refused["errors"][0]
    assert "at least 3 models" in first["error"]
    model = repo.get_model(world.experiment.id, "W1")
    assert model.source == "wonder" and "np.exp" in model.code
    assert {m.id for m in repo.list_models(world.experiment.id)} == {"S0", "S1", "W1"}
    assert conclusion.winner_id in {"S0", "S1", "W1"}
