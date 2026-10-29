"""Part 14's Done-when: the reviewer challenges two planted bad claims with the right issue.

Muse Spark Open reviewed each packet live; its replies are saved in fixtures/agent/review/<case>/
and the harness results behind the packets next to them, so this replays offline. To record again
(Spark Open costs about a cent a call):

    GHOSTLINE_RECORD=1 uv run --env-file .env pytest backend/tests/test_review_planted.py -s

The cases run on a library-only analysis (so there is nobody to answer a challenge, and one
round is enough), whose conclusion is then tampered with the way a careless author would:

- overfit: spin-down data from a single viscous term, and a claim that crowns the model with an
  extra free parameter (S2) because it has the lowest training error;
- retimed: a clip whose time base is uncertain (the `time_base_uncertain` flag), and a claim of
  high confidence with tight intervals and no caveat;
- honest: the untouched library-only conclusion, to show the reviewer does not challenge
  everything.
"""

import dataclasses
from pathlib import Path

import pytest
from fixtures.agent.support import (
    RECORDING,
    REVIEWER_MODEL,
    TIMING,
    make_world,
    recorded_clients,
    runner_factory,
    spin_case,
)

from ghostline.agent.clients import use_clients
from ghostline.agent.lab import Lab
from ghostline.agent.rerun import load_inputs
from ghostline.agent.runners import RunnerPool, use_runner_factory
from ghostline.agent.session import load_session, save_session
from ghostline.agent.stage import run_physicist
from ghostline.review.stage import run_review
from ghostline.store.core_schemas import Conclusion, Quantity

FOLDER = Path(__file__).parent / "fixtures" / "agent" / "review"
HELDOUT_WORDS = ("held-out", "heldout", "held out", "forecast", "predict")
SIMPLER_WORDS = ("extra", "simpler", "parsimon", "overfit", "over-fit", "additional parameter")
TIME_WORDS = ("time_base", "re-timed", "retimed", "time base", "timing", "slow", "frame rate")


def issue_text(review_issues):
    return " ".join(f"{i.claim} {i.evidence} {i.detail}" for i in review_issues).lower()


def analysed(repo, paths, settings, **world):
    """A finished library-only analysis, ready to be reviewed (and tampered with)."""
    off = dataclasses.replace(settings, physicist_model="off")
    w = make_world(repo, paths, off, **world)
    conclusion = run_physicist(w.ctx, w.experiment, w.trial, w.scene, w.trajectory, w.calibration)
    return w, conclusion


def final_for(w, repo, paths, model_id):
    """The all-data fit of another model, as the author of the claim would have run it."""
    session = load_session(paths, w.experiment.id)
    with RunnerPool() as pool:
        lab = Lab(
            w.ctx, w.experiment, w.trial, load_inputs(w.ctx, w.experiment, w.trial), session, pool
        )
        lab.ensure_final(model_id)
    return load_session(paths, w.experiment.id)


def review(w, case, reviewer_clients):
    meta, reviewer, _, _ = reviewer_clients
    with use_clients(meta, reviewer):
        return run_review(w.ctx, w.experiment, w.conclusion)


@pytest.fixture
def case_env(monkeypatch):
    """Pin the harness (recorded) and the clients (recorded) for one case folder."""

    def setup(name):
        folder = FOLDER / name
        clients = recorded_clients(folder)
        if RECORDING:
            assert clients[2].settings.reviewer_model == REVIEWER_MODEL
        return folder, clients

    return setup


def run_case(name, repo, paths, settings, case_env, make):
    folder, clients = case_env(name)
    with use_runner_factory(runner_factory(folder / "harness")):
        w, conclusion = make()
        w.conclusion = conclusion
        outcome = review(w, name, clients)
    return w, outcome, clients


def test_a_forced_overfit_is_challenged_on_its_extra_parameter(repo, paths, settings, case_env):
    def make():
        case = spin_case("S0", dict(phi0=0.0, omega0=40.0, b=0.25), seed=11)
        w, honest = analysed(repo, paths, settings, case=case)
        assert honest.winner_id == "S0"  # the harness picks the true, simplest model
        session = final_for(w, repo, paths, "S2")
        sel = session.selection
        qs = session.quantities["S2"]
        train = {m.id: m.train_rmse for m in sel.models}
        claim = Conclusion(
            winner_id="S2",
            quantities=[
                Quantity(
                    name=q.name, value=q.value, ci_low=q.ci_low, ci_high=q.ci_high, unit=q.unit
                )
                for q in qs
            ],
            rejected=[
                {
                    "id": m,
                    "reason": "It leaves a larger training error than S2.",
                    "evidence": f"Training RMSE {train[m]:.4g} rad against {train['S2']:.4g} rad.",
                }
                for m in ("S0", "S1", "S3")
            ],
            confidence="high",
            caveats=[],
            narrative_md=(
                "S2, viscous plus Coulomb friction, fits the spin-down best: it has the lowest "
                "training error of the four models, so both friction terms are needed."
            ),
        )
        save_session(paths, w.experiment.id, session)
        return w, claim

    w, outcome, _ = run_case("overfit", repo, paths, settings, case_env, make)

    reviews = repo.list_reviews(w.experiment.id)
    first = reviews[0]
    major = [i for i in first.issues if i.severity == "major"]
    assert first.verdict == "challenge" and major
    text = issue_text(major)
    assert any(w_ in text for w_ in HELDOUT_WORDS)  # it points at the held-out comparison
    assert any(w_ in text for w_ in SIMPLER_WORDS) or "s0" in text  # the extra parameter
    assert outcome.review_status == "contested"
    assert outcome.conclusion.confidence == "low"


def test_a_retimed_clip_is_challenged_on_its_time_base(repo, paths, settings, case_env):
    retimed = {**TIMING, "retimed": True, "needs_confirmation": True}

    def make():
        w, honest = analysed(repo, paths, settings, timing=retimed)
        assert honest.confidence == "low"  # the rules already see the flag
        q = {x.name: x for x in honest.quantities}
        claim = honest.model_copy(
            update={
                "confidence": "high",
                "caveats": [],
                "narrative_md": (
                    f"The spinner loses speed to a constant friction of {q['c'].value:.4g} "
                    f"rad/s^2 and a viscous rate of {q['b'].value:.4g} per second. The "
                    "intervals are tight, so these values are reliable."
                ),
            }
        )
        return w, claim

    w, outcome, _ = run_case("retimed", repo, paths, settings, case_env, make)

    first = repo.list_reviews(w.experiment.id)[0]
    major = [i for i in first.issues if i.severity == "major"]
    assert first.verdict == "challenge" and major
    assert any(w_ in issue_text(major) for w_ in TIME_WORDS)
    assert any(i.requested_action in ("lower_confidence", "add_caveat", "refit") for i in major)
    assert outcome.review_status == "contested" and outcome.conclusion.confidence == "low"


def test_an_honest_claim_is_not_challenged(repo, paths, settings, case_env):
    def make():
        w, honest = analysed(repo, paths, settings)
        return w, honest

    w, outcome, _ = run_case("honest", repo, paths, settings, case_env, make)

    first = repo.list_reviews(w.experiment.id)[0]
    assert first.verdict in ("accept", "accept_with_caveats")
    assert outcome.review_status in ("accepted", "accepted_with_caveats")
    assert outcome.conclusion.winner_id == "S2"
