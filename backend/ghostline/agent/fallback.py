"""Library-only mode (build plan Part 13, Failures).

When Wonder can't be used (disabled, unreachable, over the experiment budget, or it broke the
protocol), the analysis still finishes: fit every library model for the system, select by
held-out error, refit the winner for the final numbers, and write the summary from templates.
The notebook says Wonder was unavailable. No model proposes, judges or words anything here, and
every number still comes from the harness.

Models Wonder already fitted are reused, so a failure late in the loop loses nothing. Custom
models it wrote are left out: library-only means the curated library.
"""

from __future__ import annotations

import logging

from ghostline.confidence import ConfidenceAssessment
from ghostline.jobs.errors import StageError
from ghostline.physics.select import ModelScore, Selection
from ghostline.store.core_schemas import Conclusion, Quantity

from . import events
from .conclusion import record_conclusion
from .events import fmt
from .lab import Lab, LabError

log = logging.getLogger("ghostline.agent")

NOTICE = "Wonder was unavailable, so this summary was written from a template."


def run_library_only(lab: Lab, reason: str) -> Conclusion:
    session = lab.session
    session.mode, session.unavailable_reason = "library_only", reason
    session.history.clear()
    events.library_only(lab.ctx.notebook, reason)

    # Custom models are not part of a library-only analysis.
    for model_id, entry in list(session.models.items()):
        if entry.spec.code is not None:
            session.results.pop(model_id, None)
            lab.set_status(model_id, "rejected", "Not used: Wonder became unavailable.")
            session.models.pop(model_id)
    for rank, ref in enumerate(lab.library_ids(), start=1):
        if ref not in session.models:
            entry = lab.library_entry(ref, rank)
            lab.add_model(entry)
            events.hypothesis(lab.ctx.notebook, entry)
    lab.ctx.phase("fitting")
    lab.ctx.progress(0.2)
    lab.reselect()  # drop any custom model from a selection made before the fallback
    lab.fit_selection(list(session.models))
    if session.selection is None or session.selection.selected_id is None:
        raise StageError(
            "None of the library models could be fitted to this motion.",
            code="no_fit",
            retryable=False,
        )
    lab.ctx.progress(0.7)
    try:
        lab.ensure_final()
    except LabError as exc:
        raise StageError(str(exc), code="no_fit", retryable=False) from exc
    assessment = lab.assess(None)
    conclusion = templated_conclusion(lab, assessment)
    record_conclusion(lab, conclusion, assessment, actor="harness")
    log.info("library-only analysis finished", extra={"winner": conclusion.winner_id})
    return conclusion


def _unit(lab: Lab) -> str:
    units = set(lab.inputs.units.values())
    return units.pop() if len(units) == 1 else ""


def _heldout(score: ModelScore, sel: Selection) -> tuple[float | None, tuple[float, float] | None]:
    key = f"{sel.primary_split:.2f}"
    return score.heldout_rmse.get(key), score.heldout_ci.get(key)


def templated_conclusion(lab: Lab, assessment: ConfidenceAssessment) -> Conclusion:
    sel = lab.session.selection
    assert sel is not None and sel.selected_id is not None
    winner_id = sel.selected_id
    entries = lab.session.models
    unit = _unit(lab)
    pct = round((1 - sel.primary_split) * 100)
    win_score = sel.score(winner_id)
    win_rmse, _ = _heldout(win_score, sel)
    tied = set(sel.tied_with) if sel.status == "tied" else set()
    quantities = lab.session.quantities[winner_id]

    rejected = []
    for score in sel.models:
        if score.id == winner_id:
            continue
        entry = entries[score.id]
        if score.status != "ok":
            rejected.append(
                {"id": score.id, "reason": score.error or "It could not be fitted.", "evidence": ""}
            )
            continue
        rmse, ci = _heldout(score, sel)
        ci_text = f" (95% interval {fmt(ci[0], 3)} to {fmt(ci[1], 3)})" if ci else ""
        evidence = (
            f"Held-out RMSE {fmt(rmse)} {unit}{ci_text}, against {fmt(win_rmse)} {unit} for "
            f"{winner_id}."
        )
        diag = score.train_diagnostics
        if diag is not None and diag.systematic and diag.lag1 is not None:
            evidence += (
                f" Its residuals still show structure (lag-1 autocorrelation {fmt(diag.lag1, 3)})."
            )
        if score.id in tied:
            reason = (
                f"{entry.name} predicts the held-out motion as well as {winner_id} within the "
                "uncertainty, so the data can't separate them."
            )
        else:
            reason = f"{entry.name} predicts the last {pct}% of the motion worse than {winner_id}."
        rejected.append({"id": score.id, "reason": reason, "evidence": evidence})

    head = quantities[0] if quantities else None
    value = (
        f" It gives {head.name} = {fmt(head.value)} ({fmt(head.ci_low, 3)} to "
        f"{fmt(head.ci_high, 3)}) {head.unit}."
        if head
        else ""
    )
    winner = entries[winner_id]
    if sel.status == "decided":
        lead = (
            f"**{winner.name}** predicts the last {pct}% of the motion best, with a held-out "
            f"RMSE of {fmt(win_rmse)} {unit}, and it beats every other model on the held-out "
            f"data.{value}"
        )
    elif sel.status == "tied":
        rivals = ", ".join(entries[i].name for i in sel.tied_with)
        lead = (
            f"The data can't tell **{winner.name}** apart from {rivals}; {winner.name} is "
            f"reported because it is the simplest.{value}"
        )
    else:
        lead = (
            f"The held-out splits disagree, so Ghostline can't tell which model is right. "
            f"**{winner.name}** came closest.{value}"
        )
    level = assessment.level
    caveats = [NOTICE]
    caveats += [
        r.text
        for r in assessment.reasons
        if r.level != "high" and r.code not in ("not_reviewed", "all_clear")
    ]
    if assessment.level == "low":
        caveats += [f"To tighten this: {fix}" for fix in assessment.fixes]
    return Conclusion(
        winner_id=winner_id,
        quantities=[
            Quantity(name=q.name, value=q.value, ci_low=q.ci_low, ci_high=q.ci_high, unit=q.unit)
            for q in quantities
        ],
        rejected=rejected,
        confidence=level,
        caveats=list(dict.fromkeys(caveats)),
        narrative_md=f"{lead} {NOTICE}",
    )
