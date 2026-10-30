"""What an export needs to know about a finished experiment (build plan Part 19).

`measure(repo, experiment)` reads the winner, its final quantities with their 95% intervals,
and the geometry and ranges the clip measured. Everything is optional except the winner and its
quantities; a missing piece shows up as a clear error only when something needs it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID

import numpy as np

from ghostline.jobs.errors import StageError
from ghostline.physics.library.ramp import STANDARD_GRAVITY
from ghostline.store.core_schemas import Quantity
from ghostline.store.records import Experiment, ModelRecord, TrajectoryRecord, Trial

from .mapping import EXPORTS, system_of

# Used when the scene analysis gives no size: a 4 cm ball or block (2 cm half-size).
DEFAULT_HALF_SIZE_M = 0.02
# Where an experiment page lives; set to the site's origin so the exported URL is absolute.
SITE_URL_ENV = "GHOSTLINE_SITE_URL"


class ExportRepo(Protocol):
    """The slice of `store.repo.Repo` the export reads."""

    def get_model(self, experiment_id: UUID, model_id: str) -> ModelRecord | None: ...
    def list_trials(self, experiment_id: UUID) -> list[Trial]: ...
    def list_trajectories(self, trial_ids: list[UUID]) -> list[TrajectoryRecord]: ...


@dataclass
class Measurement:
    experiment: Experiment
    model_id: str  # library id of the winner, for example "R1"
    system: str
    quantities: dict[str, Quantity]
    object_name: str
    surface_name: str | None
    half_size_m: float  # ball radius, block half-size
    half_size_known: bool
    geometry: dict[str, float] = field(default_factory=dict)  # angle_deg, length_m, ...
    ranges: dict[str, list[float]] = field(default_factory=dict)  # speed_m_s, ...
    impact_speeds: list[float] = field(default_factory=list)

    @property
    def gravity(self) -> float:
        """The gravity the fit used: fitted g where the model fits it, else standard gravity."""
        q = self.quantities.get("g")
        return q.value if q else STANDARD_GRAVITY

    def quantity(self, name: str) -> Quantity | None:
        return self.quantities.get(name)

    def experiment_url(self) -> str:
        return f"{os.environ.get(SITE_URL_ENV, '').rstrip('/')}/e/{self.experiment.id}"


def _library_id(repo: ExportRepo, experiment: Experiment) -> str:
    winner = experiment.winner_model_id or (
        experiment.conclusion.winner_id if experiment.conclusion else None
    )
    if winner is None:
        raise StageError("There is no winning model to export.", code="not_analyzed")
    record = repo.get_model(experiment.id, winner)
    library_id = (record.library_ref if record else None) or winner
    if library_id not in EXPORTS or (record is not None and record.source == "wonder"):
        raise StageError(
            "The winning law has no counterpart in a simulator, so there is nothing to export.",
            code="not_exportable",
        )
    return library_id


def _fit_trial(repo: ExportRepo, experiment: Experiment) -> tuple[Trial | None, dict[str, Any]]:
    trials = [t for t in repo.list_trials(experiment.id) if t.role == "fit"]
    trial = trials[-1] if trials else None
    if trial is None:
        return None, {}
    summaries = [t.summary for t in repo.list_trajectories([trial.id]) if t.summary]
    return trial, (summaries[0] if summaries else {})


def describe_objects(experiment: Experiment) -> tuple[str, str | None, float | None]:
    """(moving body, surface, size in cm) as the scene analysis named them."""
    scene = experiment.scene
    if scene is None:
        return experiment.title, None, None
    mover = next((o for o in scene.tracked_objects if o.role == "moving_body"), None)
    surface = next((o for o in scene.tracked_objects if o.role == "surface"), None)
    props = scene.object_properties
    name = mover.sam_prompt if mover else experiment.title
    material = str(props.get("material_guess") or "").strip()
    if material and material.lower() not in ("unknown", "unclear", name.lower()):
        name = f"{name} ({material})"
    size_cm = props.get("approx_size_cm")
    return name, surface.sam_prompt if surface else None, float(size_cm) if size_cm else None


def impact_speeds(summary: dict[str, Any], g: float, drop_height_m: float | None) -> list[float]:
    """Speeds into each impact, from the flight times: g x (flight time) / 2 after the first
    impact, sqrt(2 g h) for the first drop."""
    times = list((summary.get("events") or {}).get("impacts_t") or [])
    speeds = [g * (b - a) / 2 for a, b in zip(times[:-1], times[1:], strict=False)]
    if drop_height_m and drop_height_m > 0:
        speeds.insert(0, math.sqrt(2 * g * drop_height_m))
    return [s for s in speeds if s > 0]


def amplitude_range_deg(summary: dict[str, Any]) -> list[float] | None:
    theta = np.asarray((summary.get("samples") or {}).get("theta") or [], dtype=float)
    rng = (summary.get("ranges") or {}).get("theta")
    if theta.size >= 5:
        late = np.abs(theta[-max(theta.size // 5, 3) :])
        return [float(np.degrees(late.max())), float(np.degrees(np.abs(theta).max()))]
    if rng:
        return [0.0, float(np.degrees(max(abs(rng[0]), abs(rng[1]))))]
    return None


def measure(repo: ExportRepo, experiment: Experiment) -> Measurement:
    """Collect the winner, its quantities, and the measured geometry and ranges."""
    if experiment.conclusion is None:
        raise StageError("There is no result to export yet.", code="not_analyzed")
    model_id = _library_id(repo, experiment)
    quantities = {q.name: q for q in experiment.conclusion.quantities}
    trial, summary = _fit_trial(repo, experiment)
    calibration = (trial.calibration if trial else None) or {}
    object_name, surface_name, size_cm = describe_objects(experiment)
    system = system_of(model_id)

    half = size_cm / 200 if size_cm else DEFAULT_HALF_SIZE_M
    geometry: dict[str, float] = {}
    ranges: dict[str, list[float]] = {}
    speed_max = (summary.get("ranges") or {}).get("speed_max_m_s")
    if speed_max:
        ranges["speed_m_s"] = [0.0, float(speed_max)]

    ramp = calibration.get("ramp") or {}
    ramp_extra = summary.get("ramp") or {}
    angle = ramp.get("incline_deg") or ramp_extra.get("board_angle_deg")
    if angle is None and ramp_extra.get("path_angle_deg") is not None:
        angle = abs(ramp_extra["path_angle_deg"])
    if system == "ramp" and angle is not None:
        geometry["angle_deg"] = abs(float(angle))
        ranges["angle_deg"] = [abs(float(angle))] * 2

    pend = summary.get("pendulum") or {}
    tape = experiment.setup.pendulum_length_cm
    if system == "pendulum":
        length = pend.get("length_m") or (tape / 100 if tape else None)
        if length:
            geometry["length_m"] = float(length)
        radius = pend.get("bob_radius_m") or (half if size_cm else None)
        if radius:
            geometry["bob_radius_m"] = float(radius)
            half = float(radius)
        amplitude = amplitude_range_deg(summary)
        if amplitude:
            ranges["amplitude_deg"] = amplitude
            geometry["amplitude_rad"] = math.radians(amplitude[1])

    measurement = Measurement(
        experiment=experiment,
        model_id=model_id,
        system=system,
        quantities=quantities,
        object_name=object_name,
        surface_name=surface_name,
        half_size_m=half,
        half_size_known=size_cm is not None or "bob_radius_m" in geometry,
        geometry=geometry,
        ranges=ranges,
    )

    if system == "bounce":
        y_range = (summary.get("ranges") or {}).get("y")
        if y_range:
            geometry["drop_height_m"] = float(y_range[1] - y_range[0])
        speeds = impact_speeds(summary, measurement.gravity, geometry.get("drop_height_m"))
        measurement.impact_speeds = speeds
        if speeds:
            ranges["impact_speed_m_s"] = [min(speeds), max(speeds)]
    return measurement
