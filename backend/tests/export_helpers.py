"""Experiments for the export tests, with no database: a fake repo over records built by hand
or fitted with `physics.analysis` on synthetic data."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from ghostline.physics import analysis
from ghostline.physics.library import models_for
from ghostline.physics.select import CalibrationBudget
from ghostline.physics.synthetic import make_case
from ghostline.store.core_schemas import Conclusion, Quantity, SceneAnalysis, TrackedObject
from ghostline.store.records import (
    DeclaredCapture,
    Experiment,
    ModelRecord,
    Setup,
    TrajectoryRecord,
    Trial,
)

G = 9.80665


class FakeRepo:
    """The slice of `Repo` the export reads."""

    def __init__(self, experiment, models=(), trials=(), trajectories=(), fits=()):
        self.experiment = experiment
        self.fits = list(fits)
        self.models = {m.id: m for m in models}
        self.trials = list(trials)
        self.trajectories = list(trajectories)

    def get_model(self, experiment_id, model_id):
        return self.models.get(model_id)

    def list_trials(self, experiment_id):
        return list(self.trials)

    def list_models(self, experiment_id):
        return list(self.models.values())

    def list_fits(self, experiment_id):
        return list(self.fits)

    def list_trajectories(self, trial_ids):
        return [t for t in self.trajectories if t.trial_id in trial_ids]


def scene(system: str, mover: str, surface: str | None, size_cm: float | None) -> SceneAnalysis:
    objects = [TrackedObject(key="mover", sam_prompt=mover, role="moving_body", why="the mover")]
    if surface:
        objects.append(
            TrackedObject(key="surface", sam_prompt=surface, role="surface", why="the surface")
        )
    return SceneAnalysis(
        system_type=system,  # type: ignore[arg-type]
        description="",
        tracked_objects=objects,
        calibration={},
        object_properties={"shape": "block", "material_guess": "wood", "approx_size_cm": size_cm},
        initial_hypotheses=[],
        quality_issues=[],
        proceed=True,
    )


def quantity(name: str, value: float, rel: float = 0.02, unit: str = "") -> Quantity:
    half = abs(value) * rel
    return Quantity(name=name, value=value, ci_low=value - half, ci_high=value + half, unit=unit)


def build(
    system: str,
    winner: str,
    quantities: list[Quantity],
    *,
    calibration: dict | None = None,
    summary: dict | None = None,
    mover: str = "wooden block",
    surface: str | None = "painted board",
    size_cm: float | None = 6.0,
    confidence: str = "high",
    source: str = "library",
    setup: Setup | None = None,
) -> tuple[FakeRepo, Experiment]:
    eid = uuid4()
    conclusion = Conclusion(
        winner_id=winner,
        quantities=quantities,
        rejected=[],
        confidence=confidence,  # type: ignore[arg-type]
        caveats=["Measured on one clip."],
        narrative_md="",
    )
    experiment = Experiment(
        id=eid,
        created_at=datetime.now(UTC),
        title="Test experiment",
        system_type=system,
        status="complete",
        declared_capture=DeclaredCapture(),
        setup=setup or Setup(),
        scene=scene(system, mover, surface, size_cm),
        winner_model_id=winner,
        conclusion=conclusion,
        confidence=confidence,  # type: ignore[arg-type]
    )
    trial = Trial(
        id=uuid4(),
        experiment_id=eid,
        role="fit",
        status="ready",
        calibration=calibration,
        created_at=datetime.now(UTC),
    )
    trajectory = TrajectoryRecord(trial_id=trial.id, summary=summary or {})
    record = ModelRecord(
        experiment_id=eid, id=winner, name=winner, library_ref=winner, source=source
    )  # type: ignore[arg-type]
    return FakeRepo(experiment, [record], [trial], [trajectory]), experiment


def conclusion_quantities(result: analysis.Analysis) -> list[Quantity]:
    return [
        Quantity(name=q.name, value=q.value, ci_low=q.ci_low, ci_high=q.ci_high, unit=q.unit)
        for q in result.quantities
    ]


def fitted_ramp_block(mu_k: float = 0.2, angle_deg: float = 25.0, seed: int = 3):
    """A block sliding down a ramp, measured at 60 fps with 2 mm noise and fitted by the library."""
    alpha = math.radians(angle_deg)
    rng = np.random.default_rng(seed)
    t = np.arange(0, 1.4, 1 / 60)
    case = make_case(
        "ramp", "R1", dict(s0=0.0, v0=0.0, mu_k=mu_k), {"alpha": alpha, "g": G}, t, 0.002, rng
    )
    result = analysis.analyze(
        case.t, case.observed, case.sigma, system="ramp", constants=case.constants,
        library_ids=[m.id for m in models_for("ramp", "block")],  # a block does not roll
        budget=CalibrationBudget(sigma_angle=math.radians(0.2)), n_boot=80, n_perturb=24,
    )  # fmt: skip
    a_true = G * (math.sin(alpha) - mu_k * math.cos(alpha))
    repo, experiment = build(
        "ramp",
        result.selection.selected_id or "R1",
        conclusion_quantities(result),
        calibration={"ramp": {"angle_deg": angle_deg, "incline_deg": angle_deg, "sigma_deg": 0.2}},
        summary={"ranges": {"speed_max_m_s": a_true * 1.2}},
    )
    return repo, experiment, result, a_true


def fitted_bounce(e: float = 0.8, drop_m: float = 1.0, seed: int = 5):
    """A ball dropped from `drop_m`, filmed at 120 fps with 2 mm noise and fitted by the library."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 4.2, 1 / 120)
    params = dict(y0=drop_m, v0=0.0, y_floor=0.0, g=9.81, e=e)
    case = make_case("bounce", "B0", params, {}, t, 0.002, rng)
    result = analysis.analyze(
        case.t, case.observed, case.sigma, system="bounce", segment_id=case.segment_id,
        n_boot=80, n_perturb=24,
    )  # fmt: skip
    fall = math.sqrt(2 * drop_m / 9.81)
    impacts = [fall * (1 + 2 * sum(e**j for j in range(1, k + 1))) for k in range(0, 6)]
    repo, experiment = build(
        "bounce",
        result.selection.selected_id or "B0",
        conclusion_quantities(result),
        summary={
            "ranges": {"y": [0.0, drop_m], "speed_max_m_s": math.sqrt(2 * 9.81 * drop_m)},
            "events": {"impacts_t": impacts},
        },
        mover="steel ball",
        surface="concrete floor",
        size_cm=4.0,
    )
    return repo, experiment, result
