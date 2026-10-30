"""Entry points for the simulator export (build plan Part 19), called from `jobs/stages.py`.

`build_export(repo, experiment, fmt, mass_g)` serves `GET /export`. It takes one optional
extra, `inertia_kg_m2`, for a spinner whose moment of inertia the user knows. There is no replay
check: `verify_export` stays a stub.
"""

from __future__ import annotations

from typing import Literal

from ghostline.jobs.errors import StageError
from ghostline.store.core_schemas import SimExport
from ghostline.store.records import Experiment
from ghostline.store.repo import Repo

from .build import build_json
from .mapping import export_formats
from .measured import measure
from .snippet import render_pybullet


def build_export(
    repo: Repo,
    experiment: Experiment,
    fmt: Literal["json", "pybullet"],
    mass_g: float | None,
    inertia_kg_m2: float | None = None,
) -> SimExport | str:
    """The SimExport for fmt='json', or the PyBullet snippet text."""
    m = measure(repo, experiment)
    if fmt == "json":
        return build_json(m, mass_g, inertia_kg_m2)
    if fmt == "pybullet":
        return render_pybullet(m, mass_g / 1000 if mass_g else None, inertia_kg_m2)
    raise StageError(f"Unknown export format {fmt!r}.", code="bad_format", retryable=False)


def available_formats(repo: Repo, experiment: Experiment) -> list[str]:
    """Formats the export dialog can offer for this experiment's winner (empty hides the button)."""
    try:
        return export_formats(measure(repo, experiment).model_id)
    except StageError:
        return []
