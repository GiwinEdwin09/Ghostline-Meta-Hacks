"""Stage entry point for the reviewer (the signature of the stub in jobs/stages.py).

    run_review(ctx, experiment, conclusion) -> ReviewOutcome

The debate (build plan Part 14):

1. Spark Open reviews the packet (packet.py): accept, accept_with_caveats or challenge.
2. On a challenge with any major issue, Wonder gets the issues as a new turn with tools enabled
   (it can run a model, refit, lower its confidence or rebut with numbers), then the reviewer
   gets round 2 with the history.
3. The final status is accepted, accepted_with_caveats, or contested when the reviewer still
   challenges after two rounds. If the endpoint is down the analysis keeps its conclusion and the
   status is not_reviewed.

Wonder's side continues the conversation saved by `run_physicist` in
`experiments/{id}/agent/session.json` (see agent/session.py), and that file is also where the
harness results in the packet come from. Without a saved session, or in library-only mode, there
is nobody to answer a challenge and it stands as contested.
"""

from __future__ import annotations

import logging
from typing import Any

from ghostline.agent import events
from ghostline.agent.clients import get_clients
from ghostline.agent.lab import Lab
from ghostline.agent.loop import PhysicistLoop, WonderUnavailable, wonder_disabled
from ghostline.agent.rerun import load_inputs
from ghostline.agent.runners import RunnerPool
from ghostline.agent.session import Session, load_session, save_session
from ghostline.confidence import assess_confidence
from ghostline.jobs.context import JobContext
from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import ReviewOutcome
from ghostline.store.core_schemas import Conclusion, Review, ReviewIssue
from ghostline.store.records import Experiment, Ref, ReviewRecord

from .packet import Packet, build_packet
from .reviewer import ReviewerUnavailable, next_round, opening_messages, review_round

log = logging.getLogger("ghostline.review")
MAX_ROUNDS = 2


def run_review(ctx: JobContext, experiment: Experiment, conclusion: Conclusion) -> ReviewOutcome:
    session = load_session(ctx.paths, experiment.id)
    if session is None or session.selection is None or session.selection.selected_id is None:
        return _not_reviewed(ctx, conclusion, "The analysis session was not saved.")
    meta, reviewer = get_clients(ctx)
    runtime = reviewer.runtime
    if runtime.mode != "replay" and not runtime.settings.reviewer_base_url:
        return _not_reviewed(ctx, conclusion, "No reviewer endpoint is configured.")

    exp_id = str(experiment.id)
    current, status, issues, answers = conclusion, None, [], []
    messages: list[dict[str, Any]] = []
    previous = None
    try:
        for round_ in range(1, MAX_ROUNDS + 1):
            packet = _packet(ctx, experiment, session, current, round_, answers)
            messages = (
                opening_messages(packet) if round_ == 1 else next_round(messages, previous, packet)
            )
            review, previous = review_round(reviewer, messages, exp_id)
            record = _save_review(ctx, experiment, review, round_)
            major = [i for i in review.issues if i.severity == "major"]
            issues = review.issues
            ctx.progress(round_ / (MAX_ROUNDS + 1))
            if review.verdict == "accept":
                status = "accepted"
            elif review.verdict == "accept_with_caveats" or not major:
                status = "accepted_with_caveats"
            elif round_ == MAX_ROUNDS:
                status = "contested"
            else:
                answered = _wonder_answers(ctx, experiment, session, meta, review, record, major)
                if answered is None:
                    status = "contested"  # nobody to answer the challenge
                else:
                    current, answers = answered
                    continue
            break
    except ReviewerUnavailable as exc:
        log.warning("reviewer unavailable: %s", exc)
        if status is None and previous is None:
            return _not_reviewed(ctx, current, str(exc))
        status = status or "contested"
    assert status is not None
    return _finish(ctx, session, current, status, issues)


def _packet(
    ctx: JobContext,
    experiment: Experiment,
    session: Session,
    conclusion: Conclusion,
    round_: int,
    answers: list[Any],
) -> Packet:
    winner = session.selection.selected_id  # type: ignore[union-attr]
    plots = {}
    for kind in ("fit", "resid"):
        relative = session.plot_files.get(f"{winner}.{kind}")
        if relative:
            plots[kind] = ctx.paths.resolve_media(experiment.id, relative).read_bytes()
    if round_ == 1:
        return build_packet(session, conclusion, plots)
    lines = ["The author answered your issues:"]
    lines += [f"- {a.issue_id} [{a.stance}]: {a.response}" for a in answers]
    heading = (
        "\n".join(lines)
        + "\n\nHere is the revised packet. Review it again with the same checklist."
    )
    return build_packet(session, conclusion, plots, heading=heading)


def _save_review(ctx: JobContext, experiment: Experiment, review: Review, round_: int):
    record = ctx.repo.insert_review(
        ReviewRecord(
            experiment_id=experiment.id,
            round=round_,
            verdict=review.verdict,
            issues=review.issues,
            summary=review.summary,
        )
    )
    lines = [review.summary] if review.summary else []
    lines += [
        f"- **{i.severity}:** {i.claim} {i.evidence} "
        f"*(asks: {i.requested_action.replace('_', ' ')})*"
        for i in review.issues
    ]
    first_challenge = review.verdict == "challenge" and round_ == 1
    ctx.notebook.append(
        actor="spark_open",
        kind="review",
        title=f"Reviewer, round {round_}: {review.verdict.replace('_', ' ')}",
        body_md="\n\n".join(lines) if lines else "No issues.",
        data={
            "verdict": review.verdict,
            "issues": [i.model_dump() for i in review.issues],
            "round": round_,
            "pinned": "challenge" if first_challenge else None,
        },
        refs=[Ref(kind="review", id=str(record.id))],
    )
    return record


def _issues_message(major: list[ReviewIssue], all_issues: list[ReviewIssue]) -> str:
    def line(i: ReviewIssue) -> str:
        return (
            f"- {i.id} ({i.severity}): {i.claim} Evidence: {i.evidence} "
            f"Asks: {i.requested_action}. {i.detail}"
        )

    return (
        "Phase: RESPOND. The independent referee challenged your conclusion.\n"
        + "\n".join(line(i) for i in all_issues)
        + "\n\nYou may use the tools, for example run_fits (after propose_models) to test a model "
        "the referee asks for. Then call submit_conclusion with your conclusion, revised if the "
        "referee is right, and issue_responses answering every major issue "
        f"({', '.join(i.id for i in major)}): concede, rebut or partial, with the harness numbers "
        "that support you. Ask for a lower confidence if the evidence calls for it. Do not weaken "
        "a claim the harness supports just to please the referee."
    )


def _wonder_answers(
    ctx: JobContext,
    experiment: Experiment,
    session: Session,
    meta,
    review: Review,
    record: ReviewRecord,
    major: list[ReviewIssue],
) -> tuple[Conclusion, list[Any]] | None:
    """Wonder's turn: its revised conclusion and its answers, or None when it can't answer."""
    model = ctx.settings.physicist_model
    if session.mode != "wonder" or not session.history or wonder_disabled(model):
        return None
    trial = ctx.repo.latest_trial(experiment.id, "fit", status="ready")
    if trial is None or experiment.scene is None:
        return None
    try:
        inputs = load_inputs(ctx, experiment, trial)
    except StageError:
        return None
    with RunnerPool() as pool:
        lab = Lab(ctx, experiment, trial, inputs, session, pool)
        loop = PhysicistLoop(lab, meta, model=model, scene=experiment.scene, hint="")
        try:
            revised = loop.respond(_issues_message(major, review.issues), [i.id for i in major])
        except WonderUnavailable as exc:
            log.warning("Wonder could not answer the review: %s", exc)
            return None
        lab.save()
    answers = loop.responses
    body = "\n\n".join(f"**{a.issue_id}, {a.stance}:** {a.response}" for a in answers)
    ctx.notebook.append(
        actor="wonder",
        kind="response",
        title=f"Wonder answers round {record.round}",
        body_md=body or "Wonder revised its conclusion.",
        data={
            "round": record.round,
            "issue_responses": [a.model_dump() for a in answers],
            "winner_id": revised.winner_id,
            "confidence_requested": session.requested_confidence,
        },
        refs=[Ref(kind="review", id=str(record.id))],
    )
    ctx.repo.set_review_response(
        record.id,
        {
            "issue_responses": [a.model_dump() for a in answers],
            "conclusion": revised.model_dump(),
        },
    )
    return revised, answers


def _finish(
    ctx: JobContext,
    session: Session,
    conclusion: Conclusion,
    status: str,
    issues: list[ReviewIssue],
) -> ReviewOutcome:
    """The final conclusion for a review status: confidence from the rules with the review in,
    the reviewer's open issues as caveats."""
    assert session.selection is not None
    assessment = assess_confidence(
        session.selection,
        system=session.system,
        flags=session.flags,
        review=status,  # type: ignore[arg-type]
    )
    level = Lab.lower_of(assessment.level, session.requested_confidence)
    caveats = list(conclusion.caveats)
    if status == "accepted_with_caveats":
        caveats += [f"Reviewer: {i.claim} {i.detail}".strip() for i in issues]
    elif status == "contested":
        major = [i for i in issues if i.severity == "major"]
        caveats += [f"Contested by the reviewer: {i.claim} {i.detail}".strip() for i in major]
    for reason in assessment.reasons:
        if reason.code in ("review_caveats", "contested") and reason.text not in caveats:
            caveats.append(reason.text)
    final = conclusion.model_copy(
        update={"confidence": level, "caveats": list(dict.fromkeys(caveats))}
    )
    session.review_status, session.conclusion, session.assessment = status, final, assessment
    save_session(ctx.paths, ctx.experiment_id, session)
    if final.confidence != conclusion.confidence or status != "accepted":
        events.confidence(ctx.notebook, final.confidence, assessment)
    ctx.commit()
    return ReviewOutcome(review_status=status, conclusion=final)


def _not_reviewed(ctx: JobContext, conclusion: Conclusion, why: str) -> ReviewOutcome:
    ctx.notebook.append(
        actor="system",
        kind="warning",
        title="Not reviewed (reviewer unavailable)",
        body_md=f"{why}\n\nThe analysis is complete, but no independent model checked it.",
        data={"review_status": "not_reviewed"},
    )
    return ReviewOutcome(review_status="not_reviewed", conclusion=conclusion)
