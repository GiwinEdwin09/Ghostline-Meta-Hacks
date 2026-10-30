"""The Ghostline JSON export, `SimExport` in Appendix C (build plan Part 19)."""

from __future__ import annotations

from typing import Any

from ghostline.jobs.errors import StageError
from ghostline.store.core_schemas import ExportedParam, SimExport

from .conversions import pendulum_inertia
from .expectations import median_impact_speed, restitution_triplet
from .mapping import EXPORTS, NEEDS_INERTIA
from .measured import Measurement

RANDOMIZATION_NOTE = (
    "The randomization ranges are the 95% measurement intervals. They cover measurement "
    "uncertainty only; part-to-part variation needs wider ranges."
)


def moment_of_inertia(
    m: Measurement, mass_kg: float | None, inertia_kg_m2: float | None
) -> float | None:
    """I in kg m^2: the user's value for a spinner, m (L^2 + (2/5) r^2) for a pendulum bob."""
    if inertia_kg_m2:
        return inertia_kg_m2
    length = m.geometry.get("length_m")
    if m.system == "pendulum" and mass_kg and length:
        return pendulum_inertia(mass_kg, length, m.geometry.get("bob_radius_m", 0.0))
    return None


def exported_parameters(m: Measurement) -> list[ExportedParam]:
    params: list[ExportedParam] = []
    if m.model_id == "B1":
        # PyBullet takes one restitution: e at the median measured impact speed.
        value, low, high = restitution_triplet(m)
        speed = median_impact_speed(m)
        params.append(
            ExportedParam(
                name="restitution",
                value=value,
                ci95=(low, high),
                unit="",
                definition=(
                    f"Coefficient of restitution at the median measured impact speed, "
                    f"{speed:.2f} m/s: e0 - e1 |v|. PyBullet takes this single value."
                ),
                from_model="B1",
            )
        )
    for mapping in EXPORTS[m.model_id]:
        q = m.quantity(mapping.source)
        if q is None:
            continue
        params.append(
            ExportedParam(
                name=mapping.field,
                value=q.value,
                ci95=(q.ci_low, q.ci_high),
                unit=mapping.unit,
                definition=mapping.definition,
                from_model=m.model_id,
            )
        )
    if not params:
        raise StageError(
            "The fit did not report the parameters this law exports.", code="missing_quantity"
        )
    return params


def _range_text(name: str, values: list[float], unit: str) -> str:
    low, high = values
    span = f"{low:.3g} {unit}" if abs(high - low) < 1e-9 else f"{low:.3g} to {high:.3g} {unit}"
    return f"{name} {span}"


def applicability_notes(m: Measurement) -> list[str]:
    pair = m.object_name + (f" on {m.surface_name}" if m.surface_name else " on its surface")
    notes = [
        f"Measured for {pair}. Friction, restitution and damping describe that pair of "
        "materials; a different object or surface is a different measurement.",
    ]
    measured = []
    labels = {
        "speed_m_s": ("speeds", "m/s"),
        "impact_speed_m_s": ("impact speeds", "m/s"),
        "angle_deg": ("ramp angle", "deg"),
        "amplitude_deg": ("swing amplitudes", "deg"),
    }
    for key, (name, unit) in labels.items():
        if key in m.ranges:
            measured.append(_range_text(name, m.ranges[key], unit))
    if measured:
        notes.append(
            f"Measured at {', '.join(measured)}. Using the numbers outside these ranges is "
            "extrapolation."
        )
    if m.model_id == "B1":
        notes.append(
            "Restitution fell with impact speed (e = e0 - e1 |v|). PyBullet takes one value, so "
            f"`restitution` is e at the median measured impact speed ({median_impact_speed(m):.2f}"
            " m/s); restitution.e0 and restitution.e1 keep both coefficients."
        )
    if m.system in NEEDS_INERTIA:
        notes.append(
            "Joint values are rates per unit of moment of inertia (1/s and rad/s^2). Multiply by "
            "the object's inertia I to get torques; for a pendulum bob I = m (L^2 + 2/5 r^2)."
        )
    notes.append(RANDOMIZATION_NOTE)
    return notes


def build_json(
    m: Measurement, mass_g: float | None = None, inertia_kg_m2: float | None = None
) -> SimExport:
    """The canonical export: measured values with 95% intervals, ranges and applicability."""
    mass_kg = mass_g / 1000 if mass_g else None
    params = exported_parameters(m)
    geometry: dict[str, Any] = {
        k: round(v, 6) for k, v in m.geometry.items() if k != "amplitude_rad"
    }
    inertia = moment_of_inertia(m, mass_kg, inertia_kg_m2)
    if inertia is not None:
        geometry["inertia_kg_m2"] = inertia
    conclusion = m.experiment.conclusion
    valid_for: dict[str, Any] = {"model": m.model_id, "system": m.system, **m.ranges}
    if geometry:
        valid_for["geometry"] = geometry
    if m.model_id == "B1":
        valid_for["median_impact_speed_m_s"] = median_impact_speed(m)
    valid_for["notes"] = applicability_notes(m)
    if conclusion is not None and conclusion.caveats:
        valid_for["caveats"] = list(conclusion.caveats)
    return SimExport(
        schema_version="ghostline.sim/1",
        experiment_url=m.experiment_url(),
        object=m.object_name,
        surface=m.surface_name,
        confidence=conclusion.confidence if conclusion else "low",
        mass_kg=mass_kg,
        parameters=params,
        randomization={p.name: p.ci95 for p in params},
        valid_for=valid_for,
        replay_check=m.experiment.replay_check,
    )
