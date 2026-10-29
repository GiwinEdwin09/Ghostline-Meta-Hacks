"""The physicist's session: everything `run_review` needs to pick the analysis back up.

`run_physicist` and `run_review` are separate job phases (and may run in different containers), so
the state passes through the Volume: `experiments/{id}/agent/session.json`, written through
`ctx.paths` at the end of every state change that matters (proposals, fits, the conclusion). It
holds

- the models Wonder proposed (and the specs to refit them),
- the harness results (select and final), the selection, the final quantities with their
  uncertainty budgets, the data-quality flags and the calibration budget,
- the plot files, as experiment-relative paths,
- Wonder's conversation (`history`, the Responses API items, with images inline), so the respond
  round continues the same conversation, and
- the numbers Wonder has been shown (`known_numbers`), which `submit_conclusion` is checked against.

Rebuilding it from the stored models and fits would lose the held-out errors the selection needs
and Wonder's conversation, so the file is the source of truth. The reviewer never reads
`history`: its packet is built from the results only (review/packet.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ghostline.confidence import ConfidenceAssessment, QualityFlag
from ghostline.physics.harness.schema import ModelResult, ModelSpec
from ghostline.physics.select import CalibrationBudget, FinalQuantity, Selection
from ghostline.store.core_schemas import Conclusion
from ghostline.store.volume import VolumePaths

SESSION_VERSION = 1
SESSION_REL = "agent/session.json"


class ModelEntry(BaseModel):
    """One candidate model: what Wonder said about it and the spec the harness fits."""

    spec: ModelSpec
    name: str
    mechanism: str = ""
    equations_latex: str = ""
    refuting_signature: str = ""
    prior_rank: int = 0
    round: int = 0
    source: Literal["library", "wonder"] = "library"
    status: str = "proposed"
    rejection_reason: str | None = None

    @property
    def id(self) -> str:
        return self.spec.id


class Session(BaseModel):
    version: int = SESSION_VERSION
    mode: Literal["wonder", "library_only"] = "wonder"
    unavailable_reason: str | None = None
    system: str
    trial_id: str = ""
    primary_split: float = 0.65
    t_split: float = 0.0
    duration_s: float = 0.0
    flags: list[QualityFlag] = Field(default_factory=list)
    budget: CalibrationBudget = Field(default_factory=CalibrationBudget)
    constants: dict[str, float] = Field(default_factory=dict)
    sigma: float = 0.0
    motion_range: float | None = None

    models: dict[str, ModelEntry] = Field(default_factory=dict)
    results: dict[str, ModelResult] = Field(default_factory=dict)  # select-mode, by model id
    finals: dict[str, ModelResult] = Field(default_factory=dict)  # final-mode, by model id
    quantities: dict[str, list[FinalQuantity]] = Field(default_factory=dict)
    selection: Selection | None = None
    fit_ids: dict[str, str] = Field(default_factory=dict)  # model id -> latest FitRecord id
    plot_files: dict[str, str] = Field(default_factory=dict)  # plot id -> experiment-relative path

    favorite_id: str | None = None
    favorite_why: str = ""
    rounds: int = 0  # revision rounds used
    custom_attempts: dict[str, int] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)  # job ids, so they are deterministic
    analysis_notes: list[str] = Field(default_factory=list)  # Wonder's text between the tools

    history: list[dict[str, Any]] = Field(default_factory=list)
    known_numbers: list[float] = Field(default_factory=list)

    conclusion: Conclusion | None = None  # as returned by run_physicist
    assessment: ConfidenceAssessment | None = None
    review_status: str | None = None

    def entry(self, model_id: str) -> ModelEntry:
        return self.models[model_id]

    @property
    def custom_count(self) -> int:
        return sum(1 for m in self.models.values() if m.spec.code is not None)

    def winner(self) -> str | None:
        return self.selection.selected_id if self.selection else None


def session_file(paths: VolumePaths, experiment_id: Any) -> Path:
    return paths.experiment_dir(experiment_id) / SESSION_REL


def save_session(paths: VolumePaths, experiment_id: Any, session: Session) -> Path:
    """Write the session atomically, so a reader never sees half a file."""
    target = session_file(paths, experiment_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(session.model_dump_json(), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_session(paths: VolumePaths, experiment_id: Any) -> Session | None:
    target = session_file(paths, experiment_id)
    if not target.is_file():
        return None
    session = Session.model_validate_json(target.read_text(encoding="utf-8"))
    return session if session.version == SESSION_VERSION else None
