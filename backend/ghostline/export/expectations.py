"""Numbers the export derives from the fit (build plan Part 19): the restitution PyBullet gets,
the effective pendulum length, and how long the snippet's demonstration swing has to run."""

from __future__ import annotations

import numpy as np

from ghostline.jobs.errors import StageError
from ghostline.physics.library import get_model

from .conversions import restitution_interval_at_speed
from .demo_math import half_life
from .measured import Measurement

TIMESTEP_HZ = 240
DEFAULT_IMPACT_SPEED_M_S = 2.0  # when the clip gives no impact speeds
PENDULUM_PARAM = {"P2": "gamma", "P3": "kappa", "P4": "mu_p"}


def median_impact_speed(m: Measurement) -> float:
    return float(np.median(m.impact_speeds)) if m.impact_speeds else DEFAULT_IMPACT_SPEED_M_S


def restitution_triplet(m: Measurement) -> tuple[float, float, float]:
    """(value, low, high) of the restitution PyBullet gets: e, or B1's e at the median speed."""
    if m.model_id == "B1":
        e0, e1 = m.quantity("e0"), m.quantity("e1")
        if e0 is None or e1 is None:
            raise StageError("The fit has no e0 and e1 to export.", code="missing_quantity")
        return restitution_interval_at_speed(
            (e0.value, e0.ci_low, e0.ci_high),
            (e1.value, e1.ci_low, e1.ci_high),
            median_impact_speed(m),
        )
    e = m.quantity("e")
    if e is None:
        raise StageError("The fit has no restitution e to export.", code="missing_quantity")
    return e.value, e.ci_low, e.ci_high


def pendulum_length_eff(m: Measurement) -> float:
    """The effective length the fit used: L + (2/5) r^2 / L."""
    length = m.geometry.get("length_m")
    if not length:
        raise StageError(
            "The pendulum's length was not measured, so it can't be rebuilt in PyBullet.",
            code="missing_geometry",
        )
    return length + 0.4 * m.geometry.get("bob_radius_m", 0.0) ** 2 / length


def _pendulum_param(m: Measurement):
    q = m.quantity(PENDULUM_PARAM[m.model_id])
    if q is None:
        raise StageError("The fit has no damping parameter to export.", code="missing_quantity")
    return q


def pendulum_curve(
    m: Measurement, value: float, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """The library model's swing on the snippet's 240 Hz grid, for one value of its damping."""
    t = np.arange(int(round(duration_s * TIMESTEP_HZ)) + 1) / TIMESTEP_HZ
    p = {
        "theta0": m.geometry.get("amplitude_rad", 0.5),
        "omega0": 0.0,
        "g": m.gravity,
        "L": pendulum_length_eff(m),
        PENDULUM_PARAM[m.model_id]: value,
    }
    return t, get_model(m.model_id, "pendulum").predict(t, p)[0]


def pendulum_duration_s(m: Measurement) -> float:
    """Long enough to see the swing fall to half: 2.2 half-lives of the fitted model."""
    q = _pendulum_param(m)
    t, theta = pendulum_curve(m, q.value, 120.0)
    hl = half_life(t, theta)
    return 30.0 if hl is None else float(min(max(2.2 * hl, 8.0), 90.0))
