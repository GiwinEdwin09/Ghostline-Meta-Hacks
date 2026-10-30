"""The Part 14 debate on a scripted Wonder and a scripted referee: rounds, the respond turn,
statuses, the packet's contents and the outage path. No model is called."""

import dataclasses
import json

import pytest
from fixtures.agent.support import call_item, make_world, scripted_clients, tool_texts
from fixtures.fakes import chat_dict
from test_agent_loop import PROPOSAL, conclusion_args, happy_policy

from ghostline.agent.clients import use_clients
from ghostline.agent.session import load_session
from ghostline.agent.stage import run_physicist
from ghostline.llm import PermanentLLMError
from ghostline.review.stage import run_review


def review_json(verdict, issues=(), summary="Summary."):
    return chat_dict(
        json.dumps(
            {
                "verdict": verdict,
                "summary": summary,
                "issues": [
                    {
                        "id": i,
                        "severity": sev,
                        "claim": f"claim {i}",
                        "evidence": f"evidence {i}",
                        "requested_action": action,
                        "detail": f"detail {i}",
                    }
                    for i, sev, action in issues
                ],
            }
        )
    )


def wonder_policy(items, n):
    """The happy path to a conclusion, then an answer to the referee's issue I1."""
    if n <= 5:
        return happy_policy(items, n)
    args = conclusion_args(items, "Both friction terms stay.")
    table = next(
        o
        for o in (json.loads(t.split("\n\n")[0]) for t in tool_texts(items) if t.startswith("{"))
        if "models" in o and "verdict" in o
    )
    rmse = next(r["heldout_rmse"] for r in table["models"] if r["model_id"] == "S0")
    args["confidence"] = "medium"
    args["issue_responses"] = [
        {
            "issue_id": "I1",
            "stance": "rebut",
            "response": f"S0 forecasts worse, with held-out RMSE {rmse}.",
        }
    ]
    return [call_item("submit_conclusion", args, n)]


@pytest.fixture
def analysis(repo, paths, settings, tmp_path):
    def go(chats, reviewer_configured=True):
        world = make_world(repo, paths, settings)
        meta, reviewer, sdk, referee = scripted_clients(
            wonder_policy, tmp_path, chats, reviewer_configured
        )
        with use_clients(meta, reviewer):
            conclusion = run_physicist(
                world.ctx,
                world.experiment,
                world.trial,
                world.scene,
                world.trajectory,
                world.calibration,
            )
            experiment = repo.get_experiment(world.experiment.id)
            outcome = run_review(world.ctx, experiment, conclusion)
        return world, conclusion, outcome, sdk, referee

    return go


def test_a_challenge_goes_to_wonder_then_back_to_the_reviewer(analysis, repo, paths):
    chats = [
        review_json("challenge", [("I1", "major", "run_model"), ("I2", "minor", "add_caveat")]),
        review_json("accept_with_caveats", [("I3", "minor", "add_caveat")]),
    ]

    world, before, outcome, sdk, referee = analysis(chats)

    assert outcome.review_status == "accepted_with_caveats"
    assert len(sdk.calls) == 6  # five for the analysis, one answer to the referee
    first, second = referee.chat.completions.calls
    assert first["extra_headers"] == {"Modal-Routing-Affinity-Key": str(world.experiment.id)}
    assert first["response_format"]["type"] == "json_schema"
    assert first["response_format"]["json_schema"]["name"] == "Review"
    content = first["messages"][1]["content"]
    assert [p["type"] for p in content] == ["text", "text", "image_url", "text", "image_url"]
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")
    packet = content[0]["text"]
    assert "comparison_table" in packet and "data_quality_flags" in packet
    assert "calibration_and_timing_uncertainty_one_sigma" in packet
    # the referee sees artifacts, never the author's conversation or reasoning
    texts = " ".join(
        part["text"]
        for m in first["messages"]
        for part in (m["content"] if isinstance(m["content"], list) else [])
        if part["type"] == "text"
    )
    assert PROPOSAL["favorite_why"] not in texts and "If the spin rate falls" not in texts
    assert '"type": "reasoning"' not in json.dumps(first["messages"])
    # round two carries the history and the author's answer
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert "[rebut]" in second["messages"][3]["content"][0]["text"]

    reviews = repo.list_reviews(world.experiment.id)
    assert [r.verdict for r in reviews] == ["challenge", "accept_with_caveats"]
    assert reviews[0].response["issue_responses"][0]["issue_id"] == "I1"
    events = repo.list_events(world.experiment.id)
    assert [e.actor for e in events if e.kind == "review"] == ["spark_open", "spark_open"]
    challenge = next(e for e in events if e.kind == "review")
    assert challenge.data["pinned"] == "challenge" and len(challenge.data["issues"]) == 2
    answer = next(e for e in events if e.kind == "response")
    assert answer.actor == "wonder" and "held-out RMSE" in answer.body_md
    assert outcome.conclusion.confidence == "medium"  # the rules with a caveated review
    assert any(c.startswith("Reviewer:") for c in outcome.conclusion.caveats)
    assert load_session(paths, world.experiment.id).review_status == "accepted_with_caveats"


def test_a_challenge_that_survives_two_rounds_is_contested(analysis):
    chats = [
        review_json("challenge", [("I1", "major", "refit")]),
        review_json("challenge", [("I1", "major", "refit")]),
    ]

    world, before, outcome, sdk, _ = analysis(chats)

    assert outcome.review_status == "contested"
    assert outcome.conclusion.confidence == "low"
    assert any(c.startswith("Contested by the reviewer") for c in outcome.conclusion.caveats)
    assert len(sdk.calls) == 6  # Wonder answered once; nothing more after round two


def test_an_accepted_claim_can_reach_high_confidence(analysis):
    world, before, outcome, _, _ = analysis([review_json("accept")])

    assert before.confidence == "medium" and outcome.review_status == "accepted"
    assert outcome.conclusion.confidence == "high"  # Wonder asked for high and the rules agree


def test_a_challenge_with_only_minor_issues_is_a_caveat(analysis):
    world, _, outcome, sdk, _ = analysis(
        [review_json("challenge", [("I1", "minor", "add_caveat")])]
    )

    assert outcome.review_status == "accepted_with_caveats" and len(sdk.calls) == 5


def test_an_unreachable_reviewer_leaves_the_conclusion_unchanged(analysis, repo):
    world, before, outcome, _, _ = analysis([PermanentLLMError("503 from the endpoint")])

    assert outcome.review_status == "not_reviewed" and outcome.conclusion == before
    assert "Not reviewed (reviewer unavailable)" in [
        e.title for e in repo.list_events(world.experiment.id)
    ]


def test_no_endpoint_configured_means_not_reviewed(analysis):
    world, before, outcome, _, referee = analysis([], reviewer_configured=False)

    assert outcome.review_status == "not_reviewed" and outcome.conclusion == before
    assert referee.chat.completions.calls == []


def test_library_only_analyses_are_reviewed_too(repo, paths, settings, tmp_path):
    off = dataclasses.replace(settings, physicist_model="off")
    world = make_world(repo, paths, off)
    chats = [review_json("challenge", [("I1", "major", "lower_confidence")])]
    meta, reviewer, _, referee = scripted_clients(None, tmp_path, chats)
    with use_clients(meta, reviewer):
        conclusion = run_physicist(
            world.ctx, world.experiment, world.trial, world.scene, world.trajectory,
            world.calibration,
        )  # fmt: skip
        outcome = run_review(world.ctx, repo.get_experiment(world.experiment.id), conclusion)

    # nobody can answer, so the challenge stands after one round
    assert outcome.review_status == "contested" and len(referee.chat.completions.calls) == 1
