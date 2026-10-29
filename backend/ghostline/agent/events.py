"""Lab notebook entries for the physicist (build plan Part 20).

The orchestrator writes these, never the model. Every number in `data` and in a harness entry's
body comes straight from the harness results; Wonder's own words (its favorite and why, the
refuting signatures, its rejection reasons and narrative) are attached verbatim as body text.
`data.pinned` marks the four key moments the notebook UI pins to the top.
"""

from __future__ import annotations

from ghostline.confidence import ConfidenceAssessment
from ghostline.notebook import NotebookWriter
from ghostline.physics.select import FinalQuantity, ModelScore, Selection
from ghostline.store.records import Ref

from .session import ModelEntry, Session


def fmt(x: float | None, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}g}"


def hypothesis(nb: NotebookWriter, entry: ModelEntry) -> None:
    body = (
        f"{entry.mechanism}\n\n$${entry.equations_latex}$$\n\n"
        f"**Refuted if:** {entry.refuting_signature}"
    )
    nb.append(
        actor="wonder",
        kind="hypothesis",
        title=f"Hypothesis {entry.id}: {entry.name}",
        body_md=body,
        data={
            "model_id": entry.id,
            "prior_rank": entry.prior_rank,
            "round": entry.round,
            "source": entry.source,
            "library_ref": entry.spec.library_ref,
        },
        refs=[Ref(kind="model", id=entry.id)],
    )


def prediction(
    nb: NotebookWriter, session: Session, favorite: ModelEntry, *, round_: int, training_pct: int
) -> None:
    """Wonder's favorite and what would refute each model, before any fit of this round."""
    others = [m for m in session.models.values() if m.round == round_ and m.id != favorite.id]
    lines = [
        session.favorite_why.strip(),
        "",
        f"**I will count {favorite.name} refuted if this happens:** {favorite.refuting_signature}",
    ]
    if others:
        lines += ["", "What would refute the others:"]
        lines += [f"- **{m.id} {m.name}:** {m.refuting_signature}" for m in others]
    lines += [
        "",
        f"*The harness scores every model on the last {100 - training_pct}% of the motion, "
        "which I never see.*",
    ]
    first = round_ == 0
    nb.append(
        actor="wonder",
        kind="prediction",
        title=(
            f"Prediction made before any fit: {favorite.name}"
            if first
            else f"Prediction for revision round {round_}: {favorite.name}"
        ),
        body_md="\n".join(lines),
        data={
            "model_id": favorite.id,
            "round": round_,
            "pinned": "prediction" if first else None,
            "signatures": {m.id: m.refuting_signature for m in session.models.values()},
        },
        refs=[Ref(kind="model", id=favorite.id)],
    )


def fit(
    nb: NotebookWriter,
    entry: ModelEntry,
    score: ModelScore,
    selection: Selection,
    unit: str,
    fit_id: str,
) -> None:
    prim = f"{selection.primary_split:.2f}"
    pct = round(selection.primary_split * 100)
    if score.status != "ok":
        nb.append(
            actor="harness",
            kind="fit",
            title=f"Fit {entry.id}: {entry.name} could not be fitted",
            body_md=score.error or "The harness could not fit this model.",
            data={"model_id": entry.id, "status": score.status, "fit_id": fit_id},
            refs=[Ref(kind="model", id=entry.id), Ref(kind="fit", id=fit_id)],
        )
        return
    physical = [p for p in score.params if p.kind == "physical" or p.derived]
    params = "\n".join(
        f"- {p.name}: **{fmt(p.value)}** ({fmt(p.ci_low, 3)} to {fmt(p.ci_high, 3)}) {p.unit}"
        for p in physical
    )
    heldout = score.heldout_rmse.get(prim)
    ci = score.heldout_ci.get(prim)
    ci_text = f" ({fmt(ci[0], 3)} to {fmt(ci[1], 3)})" if ci else ""
    diag = score.train_diagnostics
    body = (
        f"Fitted on the first {pct}% of the motion, then forecast the rest.\n\n{params}\n\n"
        f"Training RMSE **{fmt(score.train_rmse)} {unit}**. "
        f"Held-out RMSE **{fmt(heldout)} {unit}**{ci_text}."
    )
    nb.append(
        actor="harness",
        kind="fit",
        title=f"Fit {entry.id}: {entry.name}",
        body_md=body,
        data={
            "model_id": entry.id,
            "fit_id": fit_id,
            "split": selection.primary_split,
            "rerunnable": True,
            "fit": [p.model_dump() for p in physical],
            "train_rmse": score.train_rmse,
            "heldout_rmse": heldout,
            "heldout_rmse_by_split": score.heldout_rmse,
            "heldout_ci95": list(ci) if ci else None,
            "diagnostics": diag.model_dump() if diag else None,
        },
        refs=[Ref(kind="model", id=entry.id), Ref(kind="fit", id=fit_id)],
    )


def verdict(nb: NotebookWriter, selection: Selection, names: dict[str, str]) -> None:
    """The harness's own reading of the held-out comparison, after a batch of fits."""
    if selection.selected_id is None:
        return
    nb.append(
        actor="harness",
        kind="observation",
        title="Held-out comparison",
        body_md=selection.message,
        data={
            "status": selection.status,
            "selected_id": selection.selected_id,
            "splits_won": selection.splits_won,
            "n_splits": selection.n_splits,
            "tied_with": selection.tied_with,
            "splits": [s.model_dump() for s in selection.splits],
        },
        refs=[Ref(kind="model", id=selection.selected_id)],
    )


def analysis(nb: NotebookWriter, text: str) -> None:
    nb.append(actor="wonder", kind="observation", title="Wonder reads the fits", body_md=text)


def rejection(
    nb: NotebookWriter,
    entry: ModelEntry,
    reason: str,
    evidence: str,
    score: ModelScore | None,
    *,
    first: bool,
    actor: str = "wonder",
) -> None:
    body = f"{reason}\n\n**Evidence:** {evidence}" if evidence else reason
    nb.append(
        actor=actor,  # type: ignore[arg-type]
        kind="rejection",
        title=f"Rejected {entry.id}: {entry.name}",
        body_md=body,
        data={
            "model_id": entry.id,
            "pinned": "rejection" if first else None,
            "heldout_rmse": score.heldout_rmse if score else None,
            "heldout_ci95": {k: list(v) for k, v in score.heldout_ci.items()} if score else None,
        },
        refs=[Ref(kind="model", id=entry.id)],
    )


def winner(
    nb: NotebookWriter,
    entry: ModelEntry,
    narrative: str,
    quantities: list[FinalQuantity],
    selection: Selection,
    *,
    actor: str = "wonder",
) -> None:
    nb.append(
        actor=actor,  # type: ignore[arg-type]
        kind="acceptance",
        title=f"Winner: {entry.name}",
        body_md=narrative,
        data={
            "model_id": entry.id,
            "status": selection.status,
            "quantities": [q.model_dump() for q in quantities],
        },
        refs=[Ref(kind="model", id=entry.id)],
    )


def confidence(nb: NotebookWriter, level: str, assessment: ConfidenceAssessment) -> None:
    lines = [f"- {r.text}" for r in assessment.reasons]
    if assessment.fixes:
        lines += ["", "**To tighten this:**", *[f"- {fix}" for fix in assessment.fixes]]
    nb.append(
        actor="harness",
        kind="acceptance",
        title=f"Confidence: {level}",
        body_md="\n".join(lines),
        data={
            "confidence": level,
            "cant_tell": assessment.cant_tell,
            "reasons": [r.model_dump() for r in assessment.reasons],
            "fixes": assessment.fixes,
        },
    )


def library_only(nb: NotebookWriter, reason: str) -> None:
    nb.append(
        actor="system",
        kind="warning",
        title="Wonder was unavailable, so this analysis uses the library models only",
        body_md=(
            f"{reason}\n\nGhostline fitted every library model for this kind of motion, chose "
            "the one with the best held-out prediction, and wrote the summary from templates. "
            "The numbers come from the same harness; no model proposed or judged them."
        ),
        data={"library_only": True, "reason": reason},
    )
