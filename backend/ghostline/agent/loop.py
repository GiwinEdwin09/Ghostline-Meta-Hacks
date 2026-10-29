"""The Part 13 orchestrator: HYPOTHESIZE -> FIT -> ANALYZE -> REVISE -> CONCLUDE.

Wonder acts only through the seven tools. This module owns the state machine around them:

- the context, with the system prompt and the library catalog first (prompt caching), then the
  scene analysis, the training window (about 60 points and a plot), the calibration and the hint;
- reasoning effort per phase (high to hypothesize and conclude, low to fit, medium to analyze),
  `max_output_tokens` of at least 8,000 (reasoning shares it), always streaming;
- budgets: at most 8 tool calls per phase (a call with bad arguments counts), 2 revision rounds,
  4 custom models;
- the conversation, kept as Responses API items: Wonder's `output` goes back in as it came, with
  each `function_call_output` after it, so no `previous_response_id` is needed.

If Wonder can't be reached or stops following the protocol, `WonderUnavailable` is raised and the
caller finishes in library-only mode (fallback.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import ghostline
from ghostline.llm import (
    BudgetExceeded,
    MetaClient,
    PermanentLLMError,
    StructuredOutputError,
    TransientLLMError,
    function_call_output,
    input_image,
    input_text,
    user_message,
)
from ghostline.llm.meta import FunctionCall
from ghostline.store.core_schemas import Conclusion, SceneAnalysis

from . import conclusion as concl
from .lab import Lab, LabError
from .numbers import round_tree
from .plots import training_plot
from .schemas import ConclusionArgs
from .tools import ToolBox, ToolOutput, error, final_payload, tool_specs

log = logging.getLogger("ghostline.agent")

PROMPT_PATH = Path(ghostline.__file__).parent / "prompts" / "physicist.md"
MAX_CALLS_PER_PHASE = 8
MAX_ROUNDS = 2
MAX_TURNS = 40
MAX_IDLE_TURNS = 2  # replies without a tool call in a row, before giving up
EFFORT = {
    "HYPOTHESIZE": "high",
    "FIT": "low",
    "ANALYZE": "medium",
    "REVISE": "low",
    "CONCLUDE": "high",
    "RESPOND": "high",
}
MAX_TOKENS = {"high": 12000}
DEFAULT_MAX_TOKENS = 8000
NO_WONDER = {"", "off", "none", "disabled", "library-only", "library_only"}


class WonderUnavailable(Exception):
    """Wonder failed (API error, budget, or it broke the protocol). Finish in library-only mode."""


def wonder_disabled(model: str) -> bool:
    return model.strip().lower() in NO_WONDER


def load_prompt() -> tuple[str, str]:
    """(prompt text, version): the physicist prompt and a short hash of it."""
    import hashlib

    text = PROMPT_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]


def build_instructions(lab: Lab) -> str:
    """System prompt, the custom-model contract and the library catalog for this system. This
    is the stable prefix of every request in the session."""
    from .proposals import contract_text

    text, _ = load_prompt()
    catalog = json.dumps(round_tree(lab.catalog()), separators=(",", ":"))
    constants = json.dumps(lab.inputs.constants) if lab.inputs.constants else "none"
    lab.remember(lab.catalog(), lab.inputs.constants)
    return (
        f"{text}\n\n## Custom model contract\n{contract_text()}\n"
        f"Constants available to a custom model through p: {constants}.\n\n"
        f"## Library catalog for a {lab.inputs.system} system\n{catalog}\n"
    )


def first_message(lab: Lab, scene: SceneAnalysis, hint: str) -> dict[str, Any]:
    """The HYPOTHESIZE context: scene analysis, training-window summary, plot, calibration."""
    inputs = lab.inputs
    view = inputs.training()
    t, obs = view.sample(60)
    names = list(obs)
    rows = ["t_s," + ",".join(names)]
    rows += [f"{ti:.4f}," + ",".join(f"{obs[n][i]:.5g}" for n in names) for i, ti in enumerate(t)]
    scene_view = scene.model_dump(include={
        "system_type", "description", "tracked_objects", "object_properties",
        "initial_hypotheses", "quality_issues",
    })  # fmt: skip
    lab.remember(inputs.context)
    text = (
        "Phase: HYPOTHESIZE.\n\n"
        f"The user's hint: {hint or 'none'}\n\n"
        "## Your earlier scene analysis (before any measurement)\n"
        f"{json.dumps(scene_view, separators=(',', ':'))}\n\n"
        "## Measurements and calibration (training window only)\n"
        f"{json.dumps(round_tree(inputs.context), separators=(',', ':'))}\n\n"
        f"## About {len(t)} evenly spaced points of the training window (CSV; for reading the "
        "shape, not for quoting)\n"
        + "\n".join(rows)
        + "\n\nThe plot shows the training window; the shaded stretch is held out and hidden from "
        "you.\n\nPropose 3 to 5 competing models with propose_models. Look at the data or frames "
        "first if it helps."
    )
    return user_message(text, input_image(training_plot(inputs), "image/png"))


class PhysicistLoop:
    def __init__(self, lab: Lab, meta: MetaClient, *, model: str, scene: SceneAnalysis, hint: str):
        self.lab = lab
        self.meta = meta
        self.model = model
        self.scene = scene
        self.hint = hint
        self.toolbox = ToolBox(lab)
        self.toolbox.on_conclusion = self._on_conclusion
        self.tools = tool_specs()
        self.phase = "HYPOTHESIZE"
        self.used: dict[str, int] = {}
        self.conclusion: Conclusion | None = None
        self.turns = 0
        self.idle = 0
        self.required_issues: list[str] = []
        self.responses: list = []
        self.history = lab.session.history

    # --- model calls ------------------------------------------------------------------------
    def _ask(self, phase: str | None = None):
        phase = phase or self.phase
        effort = EFFORT[phase]
        try:
            response = self.meta.respond(
                model=self.model,
                instructions=self.instructions,
                input=self.history,
                tools=self.tools,
                reasoning_effort=effort,
                max_output_tokens=MAX_TOKENS.get(effort, DEFAULT_MAX_TOKENS),
                stream=True,
                experiment_id=str(self.lab.exp_id),
                phase=phase.lower(),
            )
        except (
            PermanentLLMError,
            TransientLLMError,
            StructuredOutputError,
            BudgetExceeded,
        ) as error_:
            raise WonderUnavailable(f"{type(error_).__name__}: {error_}") from error_
        self.history.extend(response.output)
        return response

    # --- the state machine ------------------------------------------------------------------
    def start(self) -> None:
        self.instructions = build_instructions(self.lab)
        self.history.clear()
        self.history.append(first_message(self.lab, self.scene, self.hint))

    def run(self) -> Conclusion:
        if not self.history:
            self.start()
        else:
            self.instructions = build_instructions(self.lab)
        while self.conclusion is None:
            self.lab.ctx.check_alive()
            self.turns += 1
            if self.turns > MAX_TURNS:
                raise WonderUnavailable("Wonder used its whole turn budget without concluding.")
            response = self._ask()
            calls = response.function_calls
            if not calls:
                self._idle(response.output_text, response.truncated)
                continue
            self.idle = 0
            for call in calls:
                self.history.append(self._run_call(call))
                if self.conclusion is not None:
                    break
            self.lab.save()
        return self.conclusion

    def _idle(self, text: str, truncated: bool) -> None:
        """A reply with no tool call: the analysis in ANALYZE, otherwise a nudge."""
        text = text.strip()
        if self.phase == "ANALYZE" and text and not truncated:
            self.lab.session.analysis_notes.append(text)
            from . import events

            events.analysis(self.lab.ctx.notebook, text)
            self._enter_conclude()
            self.history.append(user_message(self._conclude_message()))
            return
        self.idle += 1
        if self.idle > MAX_IDLE_TURNS:
            raise WonderUnavailable(f"Wonder stopped calling tools in the {self.phase} phase.")
        hint = {
            "HYPOTHESIZE": "Call propose_models with 3 to 5 competing models.",
            "FIT": "Call run_fits with split 'selection' for your models.",
            "ANALYZE": "Reply with your 2 to 4 sentence reading of the fits, or call a tool.",
            "REVISE": "Call run_fits for the new models.",
            "CONCLUDE": "Call submit_conclusion.",
            "RESPOND": "Answer the reviewer's issues by calling submit_conclusion.",
        }.get(self.phase, "Use the tools.")
        self.history.append(user_message(f"Phase: {self.phase}. Use the tools. {hint}"))

    def _run_call(self, call: FunctionCall) -> dict[str, Any]:
        self.used[self.phase] = self.used.get(self.phase, 0) + 1
        if self.used[self.phase] > MAX_CALLS_PER_PHASE:
            if self.phase in ("HYPOTHESIZE", "CONCLUDE", "RESPOND"):
                raise WonderUnavailable(
                    f"Wonder used all {MAX_CALLS_PER_PHASE} tool calls of the {self.phase} phase."
                )
            out = self._move_on(call)
        else:
            out = self.toolbox.call(call.name, call.arguments, self.phase)
            guidance = self._after(call.name, out)
            if guidance:
                out = _with_guidance(out, guidance)
        return function_call_output(call.call_id, out.output)

    def _move_on(self, call: FunctionCall) -> ToolOutput:
        """The tool budget of a middle phase ran out: go to the conclusion with what exists."""
        self._enter_conclude()
        return error(
            f"The tool budget of this phase is spent. {self._conclude_message()}",
        )

    def _after(self, name: str, out: ToolOutput) -> str | None:
        """Phase transitions after a tool call, and the guidance Wonder gets with the result."""
        lab, sess = self.lab, self.lab.session
        if out.effect == "proposed" and self.phase == "HYPOTHESIZE":
            self.phase = "FIT"
            lab.ctx.phase("fitting")
            lab.ctx.progress(0.1)
            return (
                f"Phase: FIT. Call run_fits with split 'selection' for: {', '.join(sess.models)}."
            )
        if out.effect == "proposed" and self.phase in ("ANALYZE", "REVISE"):
            self.phase = "REVISE"
            lab.ctx.progress(0.4)
            new_ids = ", ".join(out.detail.get("accepted", []))
            return f"Phase: REVISE. Call run_fits with split 'selection' for: {new_ids}."
        if out.effect == "fitted" and self.phase in ("FIT", "REVISE"):
            self.phase = "ANALYZE"
            lab.ctx.progress(0.6 if self.phase == "ANALYZE" else 0.4)
            left = MAX_ROUNDS - sess.rounds
            more = (
                f"To test another mechanism call propose_models ({left} revision round(s) left). "
                if left > 0
                else "No revision rounds are left. "
            )
            return (
                "Phase: ANALYZE. Read the table and the verdict; view_plot shows the training "
                f"window. {more}Otherwise reply in 2 to 4 sentences saying what the fits show."
            )
        return None

    def _enter_conclude(self) -> None:
        lab = self.lab
        try:
            lab.ensure_final()
        except LabError as exc:
            raise WonderUnavailable(f"The final fit failed: {exc}") from exc
        self.phase = "CONCLUDE"
        lab.ctx.progress(0.85)

    def _conclude_message(self) -> str:
        lab = self.lab
        sel = lab.session.selection
        assert sel is not None and sel.selected_id is not None
        quantities = lab.session.quantities[sel.selected_id]
        assessment = lab.assess(None)
        reasons = [
            r.text for r in assessment.reasons if r.code not in ("not_reviewed", "all_clear")
        ]
        need = [m.id for m in sel.models if m.id != sel.selected_id and m.status == "ok"]
        body = lab.show(
            {
                "winner": sel.selected_id,
                "final_fit": final_payload(quantities),
                "confidence_level_from_rules": assessment.level,
                "confidence_reasons": reasons,
            }
        )
        return (
            "Phase: CONCLUDE. The harness refit the winner on all the data. Final numbers:\n"
            f"{body}\n\nCall submit_conclusion. winner_id is {sel.selected_id}. Add a rejection "
            f"for: {', '.join(need) or 'no other model'}. Quote numbers exactly as printed."
        )

    # --- submit_conclusion ------------------------------------------------------------------
    def _on_conclusion(self, args: ConclusionArgs) -> ToolOutput:
        if self.phase == "ANALYZE":
            self._enter_conclude()
            return ToolOutput(
                json.dumps({"submitted": False, "message": self._conclude_message()}), ok=False
            )
        if self.phase not in ("CONCLUDE", "RESPOND"):
            return error(f"submit_conclusion is not available in the {self.phase} phase.")
        responding = self.phase == "RESPOND"
        if responding:
            try:
                self.lab.ensure_final()  # a re-run may have changed the winner
            except LabError as exc:
                return error(str(exc))
        problems = concl.check_conclusion(
            self.lab, args, required_issue_ids=self.required_issues if responding else ()
        )
        if problems:
            return ToolOutput(json.dumps({"submitted": False, "problems": problems}), ok=False)
        conclusion, assessment = concl.build_conclusion(self.lab, args, review=None)
        self.lab.session.requested_confidence = args.confidence
        self.responses = args.issue_responses
        if not responding:
            concl.record_conclusion(self.lab, conclusion, assessment, actor="wonder")
        self.conclusion = conclusion
        return ToolOutput(json.dumps({"submitted": True}), effect="conclusion")

    # --- the review's respond round ---------------------------------------------------------
    def respond(self, message: str, required_issue_ids: list[str]) -> Conclusion:
        """Continue the conversation with the reviewer's issues as a new turn, tools enabled.
        Returns the (re-validated) conclusion; `self.responses` holds the answer to each issue."""
        self.phase = "RESPOND"
        self.required_issues = required_issue_ids
        self.conclusion, self.responses = None, []
        self.used["RESPOND"], self.idle, self.turns = 0, 0, 0
        self.history.append(user_message(message))
        return self.run()


def _with_guidance(out: ToolOutput, guidance: str) -> ToolOutput:
    if isinstance(out.output, str):
        out.output = f"{out.output}\n\n{guidance}"
    else:
        out.output = [*out.output, input_text(guidance)]
    return out
