"""propose_models: checking Wonder's candidate models and registering the ones that pass.

A library model is just a reference. A custom model is Python source, so it is checked in three
steps: the AST rules of Part 11 (in this process, without running anything), then a dry run in the
runner (a Sandbox in production; never in this process), and at most two fixes after the first
failure. At most four custom models in all. The notebook gets one hypothesis entry per accepted
model and one prediction entry with Wonder's favorite and every refuting signature, verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ghostline.physics.harness.custom import CUSTOM_MODEL_CONTRACT, spec_issues
from ghostline.physics.harness.schema import ModelSpec
from ghostline.physics.library import ParamSpec as LibraryParamSpec

from . import events
from .lab import Lab
from .schemas import ProposeArgs, ProposedModel
from .session import ModelEntry

MIN_MODELS = 3
MAX_PER_CALL = 5
MAX_MODELS = 10
MAX_CUSTOM = 4
MAX_CUSTOM_SUBMISSIONS = 3  # the first try and two fixes
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,23}$")


@dataclass
class Outcome:
    accepted: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    ok: bool = False


def _check(lab: Lab, m: ProposedModel, seen: set[str]) -> tuple[ModelSpec | None, list[str]]:
    """The harness spec for one proposed model, or the problems with it."""
    session = lab.session
    errors: list[str] = []
    if not m.refuting_signature.strip():
        errors.append("refuting_signature is required: say what result would refute this model.")
    if not m.mechanism.strip():
        errors.append("mechanism is required.")
    if (m.library_ref is None) == (m.code is None):
        return None, [*errors, "Give exactly one of library_ref and code."]

    if m.library_ref is not None:
        ids = lab.library_ids()
        if m.library_ref not in ids:
            return None, [
                *errors,
                f"{m.library_ref!r} is not a library model for this system. "
                f"Use one of: {', '.join(ids)}.",
            ]
        if m.id != m.library_ref:
            errors.append(f"A library model's id is its library id: use {m.library_ref!r}.")
        if m.library_ref in session.models or m.library_ref in seen:
            errors.append(f"{m.library_ref} was already proposed.")
        if errors:
            return None, errors
        return ModelSpec(id=m.library_ref, library_ref=m.library_ref), []

    # Custom model.
    if not _ID.match(m.id):
        errors.append("id must be letters, digits and underscores, starting with a letter.")
    if m.id in session.models or m.id in seen:
        errors.append(f"id {m.id!r} is already taken.")
    attempts = session.custom_attempts.get(m.id, 0) + 1
    session.custom_attempts[m.id] = attempts
    if attempts > MAX_CUSTOM_SUBMISSIONS:
        errors.append(f"{m.id} failed its checks {MAX_CUSTOM_SUBMISSIONS} times; it is dropped.")
    if not m.params:
        errors.append("params is required for a custom model.")
    unknown = [c for c in m.constants if c not in lab.inputs.constants]
    if unknown:
        have = ", ".join(lab.inputs.constants) or "none"
        errors.append(f"constants {unknown} are not available (available: {have}).")
    if errors:
        return None, errors
    try:
        params = {
            p.name: LibraryParamSpec(
                init=p.init, lower=p.lower, upper=p.upper, unit=p.unit, kind=p.kind
            )
            for p in m.params
        }
        spec = ModelSpec(
            id=m.id,
            code=m.code,
            name=m.name,
            mechanism=m.mechanism,
            equations_latex=m.equations_latex,
            params=params,
            observables=m.observables or None,
            state=m.state or None,
            constants=list(m.constants),
        )
    except (ValidationError, ValueError) as error:
        return None, [f"The parameter spec is invalid: {error}"]
    problems = spec_issues(spec)
    return (None, problems) if problems else (spec, [])


def process(lab: Lab, args: ProposeArgs, *, revising: bool) -> Outcome:
    """Validate the call and register every model that passes. `revising` is true after the
    first fits, when the new models form a revision round."""
    session = lab.session
    out = Outcome()
    if not args.models:
        out.payload = {"error": "models is empty."}
        return out
    if len(args.models) > MAX_PER_CALL:
        out.payload = {"error": f"Propose at most {MAX_PER_CALL} models per call."}
        return out

    seen: set[str] = set()
    specs: dict[str, tuple[ProposedModel, ModelSpec]] = {}
    results: list[dict[str, Any]] = []
    for m in args.models:
        spec, errors = _check(lab, m, seen)
        if spec is not None:
            seen.add(spec.id)
            specs[spec.id] = (m, spec)
        else:
            results.append({"id": m.id, "accepted": False, "errors": errors})
    if specs and any(s.code for _, s in specs.values()):
        custom = [s for _, s in specs.values() if s.code]
        failures = lab.validate_custom(custom)
        for model_id, errors in failures.items():
            _, spec = specs.pop(model_id)
            results.append({"id": model_id, "accepted": False, "errors": errors})
            seen.discard(model_id)

    if len(session.models) + len(specs) > MAX_MODELS:
        out.payload = {"error": f"At most {MAX_MODELS} models in all."}
        return out
    n_custom = sum(1 for _, s in specs.values() if s.code)
    if session.custom_count + n_custom > MAX_CUSTOM:
        out.payload = {"error": f"At most {MAX_CUSTOM} custom models are allowed in all."}
        return out
    round_ = session.rounds + 1 if revising else 0
    new: list[ModelEntry] = []
    for model_id, (m, spec) in specs.items():
        if spec.library_ref:
            entry = lab.library_entry(spec.library_ref, m.prior_rank, round_=round_)
            entry.refuting_signature = m.refuting_signature.strip()
        else:
            entry = ModelEntry(
                spec=spec,
                name=m.name.strip() or model_id,
                mechanism=m.mechanism.strip(),
                equations_latex=m.equations_latex.strip(),
                refuting_signature=m.refuting_signature.strip(),
                prior_rank=m.prior_rank,
                round=round_,
                source="wonder",
            )
        new.append(entry)
        results.append({"id": model_id, "accepted": True, "errors": []})

    total = len(session.models) + len(new)
    favorite = next((e for e in new if e.id == args.favorite_id), None)
    if not revising and total < MIN_MODELS:
        # Nothing is registered until the first set is big enough, so Wonder can resend it whole.
        out.payload = {
            "models": results,
            "error": f"Propose at least {MIN_MODELS} models with different mechanisms in one call "
            f"({len(new)} passed the checks).",
        }
        return out
    if favorite is None and not revising:
        out.payload = {
            "models": results,
            "error": f"favorite_id {args.favorite_id!r} is not one of the accepted models.",
        }
        return out

    for entry in new:
        lab.add_model(entry)
        events.hypothesis(lab.ctx.notebook, entry)
    if revising and new:
        session.rounds += 1
    if favorite is not None:
        session.favorite_id, session.favorite_why = favorite.id, args.favorite_why.strip()
        events.prediction(
            lab.ctx.notebook,
            session,
            favorite,
            round_=round_,
            training_pct=round(lab.inputs.primary_split * 100),
        )
    elif revising and new:
        session.favorite_why = args.favorite_why.strip()
        events.prediction(
            lab.ctx.notebook, session, new[0], round_=round_,
            training_pct=round(lab.inputs.primary_split * 100),
        )  # fmt: skip
    lab.save()
    out.accepted = [e.id for e in new]
    out.ok = bool(new)
    out.payload = {
        "models": results,
        "registered": out.accepted,
        "next": "Call run_fits with split 'selection' for the new models."
        if new
        else "None passed; fix the errors and propose again.",
    }
    return out


def contract_text() -> str:
    return CUSTOM_MODEL_CONTRACT
