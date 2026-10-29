"""submit_conclusion: checking Wonder's conclusion against the harness, then recording it.

The rules (Part 13): every number must match a harness result within rounding, and Wonder can't
introduce one the harness didn't produce. Concretely:

- `winner_id` is the model the held-out selection picked, and its quantities are the final-fit
  quantities. A value or interval that differs from the harness by more than rounding rejects the
  call; the stored conclusion uses the harness's exact numbers either way.
- Every other fitted model has a rejection whose evidence quotes that model's own held-out score
  or residual diagnostic.
- Every number in the narrative, the caveats, the rejections and the review responses is on the
  list of numbers the orchestrator showed Wonder (numbers.py).
- The confidence level comes from the rules in confidence.py. Wonder may ask for lower.

`check_conclusion` returns the problems (sent back to Wonder as the tool result); `build_conclusion`
turns an accepted call into the `Conclusion` that goes to the experiment row.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ghostline.confidence import ConfidenceAssessment, ReviewStatus
from ghostline.physics.select import FinalQuantity, ModelScore
from ghostline.store.core_schemas import Conclusion, Quantity

from . import events
from .lab import Lab
from .numbers import (
    close_enough,
    matched_numbers,
    round_tree,
    sorted_known,
    unmatched_numbers,
)
from .schemas import ConclusionArgs

MAX_NARRATIVE_CHARS = 1200
_MECHANICAL = {"not_reviewed", "all_clear"}


def _model_numbers(score: ModelScore) -> np.ndarray:
    """The held-out and residual numbers of one model, as Wonder was shown them."""
    values: list[float] = list(score.heldout_rmse.values())
    for lo, hi in score.heldout_ci.values():
        values += [lo, hi]
    values += [x for x in (score.nrmse, score.divergence_time, score.train_rmse) if x is not None]
    diag = score.train_diagnostics
    if diag is not None:
        values += [
            x
            for x in (diag.lag1, diag.durbin_watson, diag.runs_p, diag.chi2_red, diag.systematic_r2)
            if x is not None
        ]
    return sorted_known(round_tree(values))


def _texts(args: ConclusionArgs) -> Iterable[tuple[str, str]]:
    yield "narrative_md", args.narrative_md
    for i, caveat in enumerate(args.caveats):
        yield f"caveats[{i}]", caveat
    for r in args.rejected:
        yield f"rejected {r.id} reason", r.reason
        yield f"rejected {r.id} evidence", r.evidence
    for r in args.issue_responses:
        yield f"response to {r.issue_id}", r.response


def check_conclusion(
    lab: Lab, args: ConclusionArgs, *, required_issue_ids: Iterable[str] = ()
) -> list[str]:
    """Everything wrong with the call, as sentences Wonder can act on. Empty means accepted."""
    session, errors = lab.session, []
    sel = session.selection
    if sel is None or sel.selected_id is None:
        return ["No model has been fitted yet, so there is nothing to conclude."]
    if args.winner_id != sel.selected_id:
        errors.append(
            f"winner_id must be {sel.selected_id}, the model the harness selected on held-out "
            f"data ({sel.message}). You may not name another winner."
        )
    finals = session.quantities.get(sel.selected_id)
    if finals is None:
        return [*errors, "The final numbers for the winner are not available yet."]

    by_name = {q.name: q for q in finals}
    given = {q.name: q for q in args.quantities}
    for name, q in given.items():
        actual = by_name.get(name)
        if actual is None:
            errors.append(
                f"Quantity {name!r} is not a harness quantity. Use: {', '.join(by_name)}."
            )
        elif not (
            close_enough(q.value, actual.value)
            and close_enough(q.ci_low, actual.ci_low)
            and close_enough(q.ci_high, actual.ci_high)
        ):
            errors.append(
                f"Quantity {name!r} does not match the harness: it gave value {actual.value:.5g}, "
                f"95% interval {actual.ci_low:.5g} to {actual.ci_high:.5g} {actual.unit}."
            )
    missing = [q.name for q in finals if not q.derived and q.name not in given]
    if missing:
        errors.append(f"Report every fitted physical quantity; missing: {', '.join(missing)}.")
    if not args.quantities:
        errors.append("quantities is empty.")

    ok_ids = [m.id for m in sel.models if m.status == "ok" and m.id != sel.selected_id]
    rejected = {r.id: r for r in args.rejected}
    absent = [i for i in ok_ids if i not in rejected]
    if absent:
        errors.append(
            f"Add a rejection for every other fitted model; missing: {', '.join(absent)}."
        )
    unknown = [i for i in rejected if i not in session.models or i == sel.selected_id]
    if unknown:
        errors.append(f"rejected lists ids that are not losing models: {', '.join(unknown)}.")
    for model_id, r in rejected.items():
        if model_id not in ok_ids:
            continue
        evidence_known = _model_numbers(sel.score(model_id))
        if not matched_numbers(r.evidence, evidence_known):
            errors.append(
                f"The evidence for rejecting {model_id} must quote its own held-out RMSE (or a "
                "residual diagnostic) exactly as the comparison table shows it."
            )

    if not args.narrative_md.strip():
        errors.append("narrative_md is empty.")
    elif len(args.narrative_md) > MAX_NARRATIVE_CHARS:
        errors.append(
            f"narrative_md is {len(args.narrative_md)} characters; keep it to a few short "
            f"sentences (under {MAX_NARRATIVE_CHARS})."
        )

    known = lab.known()
    for where, text in _texts(args):
        bad = unmatched_numbers(text, known)
        if bad:
            errors.append(
                f"In {where}: the number(s) {', '.join(bad)} did not come from the harness or "
                "the measurements. Quote numbers exactly as shown, or leave them out."
            )

    wanted = set(required_issue_ids)
    if wanted:
        answered = {r.issue_id for r in args.issue_responses}
        if wanted - answered:
            errors.append(
                f"Answer every major issue in issue_responses; missing: "
                f"{', '.join(sorted(wanted - answered))}."
            )
    return errors


def build_conclusion(
    lab: Lab, args: ConclusionArgs, review: ReviewStatus | None = None
) -> tuple[Conclusion, ConfidenceAssessment]:
    """The stored conclusion for an accepted call: harness numbers, rule-based confidence."""
    session = lab.session
    sel = session.selection
    assert sel is not None and sel.selected_id is not None
    finals: list[FinalQuantity] = session.quantities[sel.selected_id]
    given = {q.name for q in args.quantities}
    quantities = [
        Quantity(name=q.name, value=q.value, ci_low=q.ci_low, ci_high=q.ci_high, unit=q.unit)
        for q in finals
        if q.name in given or not q.derived
    ]
    reasons = {r.id: r for r in args.rejected}
    rejected = [
        {"id": m.id, "reason": reasons[m.id].reason, "evidence": reasons[m.id].evidence}
        if m.id in reasons
        else {"id": m.id, "reason": m.error or "It could not be fitted.", "evidence": ""}
        for m in sel.models
        if m.id != sel.selected_id
    ]
    assessment = lab.assess(review)
    level = lab.lower_of(assessment.level, args.confidence)
    caveats = list(dict.fromkeys(c.strip() for c in args.caveats if c.strip()))
    for reason in assessment.reasons:
        if reason.code not in _MECHANICAL and reason.level != "high" and reason.text not in caveats:
            caveats.append(reason.text)
    if assessment.level == "low":
        caveats += [f"To tighten this: {fix}" for fix in assessment.fixes if fix not in caveats]
    conclusion = Conclusion(
        winner_id=sel.selected_id,
        quantities=quantities,
        rejected=rejected,
        confidence=level,  # type: ignore[arg-type]
        caveats=caveats,
        narrative_md=args.narrative_md.strip(),
    )
    return conclusion, assessment


def record_conclusion(
    lab: Lab,
    conclusion: Conclusion,
    assessment: ConfidenceAssessment,
    *,
    actor: str = "wonder",
) -> None:
    """Write the conclusion to the notebook and the model rows: a rejection for each model the
    harness beat, the winner, and the confidence with its reasons. `actor` is wonder for its own
    conclusion and harness for a templated one."""
    sel = lab.session.selection
    assert sel is not None and sel.selected_id is not None
    nb = lab.ctx.notebook
    tied = set(sel.tied_with) if sel.status == "tied" else set()
    first = True
    for item in conclusion.rejected:
        model_id = item["id"]
        entry = lab.session.models[model_id]
        score = sel.score(model_id)
        if score.status != "ok":
            continue
        if model_id in tied:
            lab.set_status(model_id, "tied", item["reason"])
            continue
        events.rejection(
            nb, entry, item["reason"], item["evidence"], score, first=first, actor=actor
        )
        lab.set_status(model_id, "rejected", item["reason"])
        first = False
    winner = lab.session.models[sel.selected_id]
    lab.set_status(winner.id, "winner")
    events.winner(
        nb, winner, conclusion.narrative_md, lab.session.quantities[winner.id], sel, actor=actor
    )
    events.confidence(nb, conclusion.confidence, assessment)
    lab.session.conclusion = conclusion
    lab.session.assessment = assessment
    lab.save()
