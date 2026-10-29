"""Wonder's seven tools (build plan Part 13): their definitions and what each one does.

    list_library_models, get_training_data, view_frames, propose_models, run_fits, view_plot,
    submit_conclusion

`tool_specs()` is the list sent with every request (the same list in every phase, so the prompt
cache holds); `ToolBox.call` runs one call. Which tools work in which phase is enforced here, not
by changing the list. Images (frames, plots) come back as `input_image` parts inside the
function output, which the Day 0 smoke test showed works. Everything Wonder is shown goes through
`Lab.show` or `Lab.remember`, so `submit_conclusion` can be checked against it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from ghostline.llm import data_url, function_tool, input_image, input_text
from ghostline.llm.structured import validation_message
from ghostline.physics.library import catalog

from . import proposals
from .frames import render_frames
from .inputs import MAX_TOOL_POINTS
from .lab import Lab, LabError
from .numbers import round_tree
from .schemas import (
    ConclusionArgs,
    ListLibraryArgs,
    ProposeArgs,
    RunFitsArgs,
    TrainingDataArgs,
    ViewFramesArgs,
    ViewPlotArgs,
)

MAX_REVISION_ROUNDS = 2

TOOLS: dict[str, tuple[str, type[BaseModel]]] = {
    "list_library_models": (
        "List the library models for a kind of motion: id, name, mechanism, equations, the "
        "signature each predicts that its rivals don't, parameters and their bounds.",
        ListLibraryArgs,
    ),
    "get_training_data": (
        "Numeric arrays of the measured motion. Only the training window exists for you; "
        "the held-out part cannot be requested.",
        TrainingDataArgs,
    ),
    "view_frames": (
        "Up to six still frames of the video at the given times, optionally with the tracked "
        "object's mask outlined or its trail drawn. Times in the held-out part are refused.",
        ViewFramesArgs,
    ),
    "propose_models": (
        "Register competing models before any fit: name your favorite and, for each model, what "
        "would refute it. Library models by reference, or custom Python code with a parameter "
        "spec. Returns the validation result for each model.",
        ProposeArgs,
    ),
    "run_fits": (
        "Run the harness. split 'selection': fit each model on the first 65% of the motion and "
        "score its forecast of the rest; you get the comparison table and the harness's verdict. "
        "split 'final': refit the selected winner on all the data for the final numbers.",
        RunFitsArgs,
    ),
    "view_plot": (
        "A plot from the harness, by id from the comparison table. Training-window plots only.",
        ViewPlotArgs,
    ),
    "submit_conclusion": (
        "Finish: the winner, its quantities with 95% intervals, a rejection for every other "
        "model, your confidence, caveats and a short narrative. Every number is checked "
        "against the harness.",
        ConclusionArgs,
    ),
}

ALLOWED: dict[str, set[str]] = {
    "HYPOTHESIZE": {"list_library_models", "get_training_data", "view_frames", "propose_models"},
    "FIT": {"run_fits", "view_plot", "get_training_data", "view_frames", "list_library_models"},
    "ANALYZE": {
        "submit_conclusion",
        "view_plot",
        "get_training_data",
        "view_frames",
        "list_library_models",
        "propose_models",
        "run_fits",
    },
    "REVISE": {
        "propose_models",
        "run_fits",
        "view_plot",
        "get_training_data",
        "list_library_models",
    },
    "CONCLUDE": {"submit_conclusion", "view_plot"},
    "RESPOND": {
        "submit_conclusion",
        "run_fits",
        "propose_models",
        "view_plot",
        "get_training_data",
        "view_frames",
        "list_library_models",
    },
}


def tool_specs() -> list[dict[str, Any]]:
    return [function_tool(name, desc, model) for name, (desc, model) in TOOLS.items()]


@dataclass
class ToolOutput:
    """One tool result. `output` is the `function_call_output` payload (text, or text and image
    parts). `effect` tells the driver what changed: proposed, fitted, final, conclusion."""

    output: str | list[dict[str, Any]]
    ok: bool = True
    effect: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


def error(message: str, **detail: Any) -> ToolOutput:
    return ToolOutput(json.dumps({"error": message}), ok=False, detail=detail)


class ToolBox:
    """Runs Wonder's tool calls against a `Lab`."""

    def __init__(self, lab: Lab) -> None:
        self.lab = lab
        self.on_conclusion: Callable[[ConclusionArgs], ToolOutput] | None = None

    def call(self, name: str, arguments: str, phase: str) -> ToolOutput:
        spec = TOOLS.get(name)
        if spec is None:
            return error(f"Unknown tool {name!r}. Tools: {', '.join(TOOLS)}.")
        if name not in ALLOWED[phase]:
            return error(
                f"{name} is not available in the {phase} phase. Available now: "
                f"{', '.join(sorted(ALLOWED[phase]))}."
            )
        try:
            args = spec[1].model_validate_json(arguments or "{}")
        except ValidationError as exc:
            return error(f"Invalid arguments for {name}: {validation_message(exc)}")
        try:
            return getattr(self, f"_{name}")(args, phase)
        except LabError as exc:
            return error(str(exc))

    # --- read-only tools --------------------------------------------------------------------
    def _list_library_models(self, args: ListLibraryArgs, phase: str) -> ToolOutput:
        entries = catalog(args.system)
        if not entries:
            return error(
                f"No library models for {args.system!r}. This experiment is "
                f"{self.lab.inputs.system}."
            )
        return ToolOutput(self.lab.show({"system": args.system, "models": entries}))

    def _get_training_data(self, args: TrainingDataArgs, phase: str) -> ToolOutput:
        n = min(max(args.max_points, 2), MAX_TOOL_POINTS)
        t, obs = self.lab.inputs.training().window(args.t_start, args.t_end, n)
        if t.size == 0:
            raise LabError(
                f"No training data between those times. The training window runs from "
                f"{self.lab.inputs.data.t[0]:.3f} s to {self.lab.session.t_split:.3f} s."
            )
        arrays = {"t": t, **obs}
        summary = {
            "n_points": int(t.size),
            "t_first": float(t[0]),
            "t_last": float(t[-1]),
            "training_window_ends_s": self.lab.session.t_split,
            **{f"{k}_min": float(v.min()) for k, v in obs.items()},
            **{f"{k}_max": float(v.max()) for k, v in obs.items()},
        }
        self.lab.remember(summary)
        body = {
            "units": {"t": "s", **self.lab.inputs.units},
            **summary,
            "series": {k: [round(float(x), 6) for x in v] for k, v in arrays.items()},
        }
        return ToolOutput(json.dumps(round_tree(body), separators=(",", ":")))

    def _view_frames(self, args: ViewFramesArgs, phase: str) -> ToolOutput:
        frames, notes = render_frames(self.lab, args.times, args.overlay)
        if not frames:
            raise LabError(" ".join(notes) or "No frame could be read.")
        parts: list[dict[str, Any]] = []
        for caption, jpeg in frames:
            parts.append(input_text(caption + (f" (overlay: {args.overlay})")))
            parts.append(input_image(jpeg, "image/jpeg"))
        if notes:
            parts.insert(0, input_text(" ".join(notes)))
        return ToolOutput(parts)

    def _view_plot(self, args: ViewPlotArgs, phase: str) -> ToolOutput:
        allowed = self.lab.agent_plot_ids()
        if args.plot_id not in allowed:
            raise LabError(
                f"Unknown plot id {args.plot_id!r}. Plots you can view: "
                f"{', '.join(sorted(allowed)) or 'none yet'}."
            )
        image = input_image(self.lab.plot_bytes(args.plot_id), "image/png")
        return ToolOutput([input_text(f"Plot {args.plot_id}: training window only."), image])

    # --- tools that change the experiment ---------------------------------------------------
    def _propose_models(self, args: ProposeArgs, phase: str) -> ToolOutput:
        revising = bool(self.lab.session.results)
        if revising and self.lab.session.rounds >= MAX_REVISION_ROUNDS:
            raise LabError(
                f"The {MAX_REVISION_ROUNDS} revision rounds are used up. Reply with your "
                "analysis, or conclude."
            )
        outcome = proposals.process(self.lab, args, revising=revising)
        if outcome.payload.get("error"):
            return ToolOutput(json.dumps(outcome.payload), ok=False)
        return ToolOutput(
            json.dumps(outcome.payload),
            ok=outcome.ok,
            effect="proposed" if outcome.ok else None,
            detail={"accepted": outcome.accepted},
        )

    def _run_fits(self, args: RunFitsArgs, phase: str) -> ToolOutput:
        lab = self.lab
        unknown = [i for i in args.model_ids if i not in lab.session.models]
        if unknown:
            raise LabError(
                f"Unknown model ids: {', '.join(unknown)}. "
                f"Proposed: {', '.join(lab.session.models)}."
            )
        if not args.model_ids:
            raise LabError("model_ids is empty.")
        if args.split == "final":
            sel = lab.session.selection
            if sel is None or sel.selected_id is None:
                raise LabError("Run the selection fits first.")
            if args.model_ids != [sel.selected_id]:
                raise LabError(
                    f"Only the selected winner can be refit for the final numbers: "
                    f"model_ids must be [{sel.selected_id!r}]."
                )
            quantities = lab.ensure_final()
            return ToolOutput(
                lab.show({"final_fit": sel.selected_id, "quantities": final_payload(quantities)}),
                effect="final",
            )
        fitted = lab.fit_selection(args.model_ids)
        return ToolOutput(lab.show(lab.table_payload(fitted)), effect="fitted")

    def _submit_conclusion(self, args: ConclusionArgs, phase: str) -> ToolOutput:
        if self.on_conclusion is None:
            return error("submit_conclusion is not available right now.")
        return self.on_conclusion(args)


def final_payload(quantities) -> list[dict[str, Any]]:
    """The final quantities as Wonder sees them, with the uncertainty budget of each."""
    return [
        {
            "name": q.name,
            "value": q.value,
            "unit": q.unit,
            "ci95": [q.ci_low, q.ci_high],
            "uncertainty_budget": q.budget,
            "derived": q.derived,
            "at_bound": q.at_bound,
        }
        for q in quantities
    ]


__all__ = ["ALLOWED", "TOOLS", "ToolBox", "ToolOutput", "data_url", "tool_specs"]
