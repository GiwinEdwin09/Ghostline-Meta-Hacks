"""What the replay is compared against (build plan Part 19): the measured value and its 95%
interval for each quantity the PyBullet run reports.

The measured side is built from the experiment's final quantities. For the pendulum, whose
half-life and period depend on the swing amplitude, the library model is integrated with the
fitted parameters, so the replay and the model are measured by the same functions
(`replay_math`), on the same 240 Hz grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ghostline.jobs.errors import StageError
from ghostline.physics.library import get_model

from .conversions import bounce_height_ratio, restitution_interval_at_speed
from .mapping import REPLAY_SCENARIOS
from .measured import Measurement
from .replay_math import half_life, mean_period

TIMESTEP_HZ = 240
PYBULLET_VERSION = "3.2.7"  # the version the replay image installs
DEFAULT_IMPACT_SPEED_M_S = 2.0  # when the clip gives no impact speeds
PENDULUM_PARAM = {"P2": "gamma", "P3": "kappa", "P4": "mu_p"}


@dataclass(frozen=True)
class Check:
    quantity: str  # shown to the user
    key: str  # the key in the replay's result
    unit: str
    measured: float
    low: float
    high: float


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


def pendulum_curve(m: Measurement, value: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """The library model's swing on the replay's 240 Hz grid, for one value of its damping."""
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


def _pendulum_checks(m: Measurement) -> list[Check]:
    q = _pendulum_param(m)
    duration = pendulum_duration_s(m)
    lives, periods = [], []
    for value in (q.ci_low, q.value, q.ci_high):
        t, theta = pendulum_curve(m, value, duration)
        lives.append(half_life(t, theta))
        periods.append(mean_period(t, theta))
    checks = []
    if all(v is not None for v in lives):
        checks.append(
            Check("swing half-life", "half_life_s", "s", lives[1], min(lives), max(lives))
        )
    if periods[1] is not None:
        # The period follows from g (and the length): scale the fitted period by g's interval.
        g = m.quantity("g")
        lo_g, hi_g = (g.ci_low, g.ci_high) if g else (m.gravity, m.gravity)
        factor = math.sqrt(m.gravity)
        checks.append(
            Check(
                "swing period",
                "period_s",
                "s",
                periods[1],
                periods[1] * factor / math.sqrt(hi_g),
                periods[1] * factor / math.sqrt(lo_g),
            )
        )
    return checks


def expected_checks(m: Measurement) -> list[Check]:
    """The comparisons the replay makes for the winner. Raises StageError if it can't replay."""
    scenario = REPLAY_SCENARIOS.get(m.model_id)
    if scenario is None:
        raise StageError(
            "This law can be exported, but the replay check covers sliding blocks, bounces, "
            "falls with drag and pendulums.",
            code="replay_unsupported",
            retryable=False,
        )
    if scenario == "ramp":
        a = m.quantity("acceleration")
        if a is not None:
            return [Check("acceleration down the ramp", "acceleration", "m/s²", a.value, a.ci_low, a.ci_high)]
        mu = m.quantity("mu_k")
        angle = m.geometry.get("angle_deg")
        if mu is None or angle is None:
            raise StageError("The fit has no acceleration to compare.", code="missing_quantity")
        sin, cos = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        accel = [m.gravity * (sin - x * cos) for x in (mu.value, mu.ci_high, mu.ci_low)]
        return [Check("acceleration down the ramp", "acceleration", "m/s²", accel[0], accel[1], accel[2])]
    if scenario == "bounce":
        e, lo, hi = restitution_triplet(m)
        return [
            Check(
                "ratio of successive bounce heights",
                "height_ratio",
                "",
                bounce_height_ratio(e),
                bounce_height_ratio(lo),
                bounce_height_ratio(hi),
            )
        ]
    if scenario == "free_fall":
        v = m.quantity("terminal_velocity")
        if v is not None:
            return [Check("terminal velocity", "terminal_velocity", "m/s", v.value, v.ci_low, v.ci_high)]
        name = "b" if m.model_id == "FF1" else "k"
        rate = m.quantity(name)
        if rate is None:
            raise StageError("The fit has no drag to compare.", code="missing_quantity")
        power = 1 if m.model_id == "FF1" else 2
        fn = (lambda r: m.gravity / r) if power == 1 else (lambda r: math.sqrt(m.gravity / r))
        return [
            Check(
                "terminal velocity", "terminal_velocity", "m/s",
                fn(rate.value), fn(rate.ci_high), fn(rate.ci_low),
            )
        ]  # fmt: skip
    return _pendulum_checks(m)
