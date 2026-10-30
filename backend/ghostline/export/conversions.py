"""Unit conversions from what Ghostline measures to what PyBullet takes (build plan Part 19).

Ghostline measures rates and dimensionless coefficients (a decay rate lambda, a rate per unit of
inertia gamma, a rolling-resistance coefficient c_rr). PyBullet wants its own units, and a few of
them are not what they look like. Each function here is one row of the Part 19 table.
"""

from __future__ import annotations

import math

# PyBullet gives every rigid body this much linear and angular damping unless told otherwise.
BULLET_DEFAULT_DAMPING = 0.04
# Bullet's contact solver ignores restitution below this closing speed (m/s).
BULLET_RESTITUTION_THRESHOLD = 0.2


def bullet_linear_damping(rate_per_s: float) -> float:
    """`linearDamping` that decays speed at the measured rate: 1 - e^(-lambda).

    Bullet multiplies the velocity by (1 - damping)^dt every step, so after one second the speed
    has been scaled by (1 - damping). A measured decay v ~ e^(-lambda t) needs 1 - damping =
    e^(-lambda).
    """
    if rate_per_s < 0:
        raise ValueError("a decay rate cannot be negative")
    return -math.expm1(-rate_per_s)


def decay_rate_from_bullet_damping(damping: float) -> float:
    """The inverse of `bullet_linear_damping`: the speed decay rate Bullet applies."""
    if not 0 <= damping < 1:
        raise ValueError("Bullet damping is a fraction in [0, 1)")
    return -math.log1p(-damping)


def bullet_rolling_friction(c_rr: float, radius_m: float) -> float:
    """`rollingFriction` for a ball: c_rr x radius.

    Bullet's rolling friction opposes rolling with a torque of rollingFriction x normal force,
    so it has units of length. The dimensionless coefficient c_rr is that torque divided by
    (normal force x radius).
    """
    return c_rr * radius_m


def pendulum_inertia(mass_kg: float, length_m: float, bob_radius_m: float) -> float:
    """Moment of inertia about the pivot for a solid spherical bob: m (L^2 + (2/5) r^2)."""
    return mass_kg * (length_m**2 + 0.4 * bob_radius_m**2)


def joint_damping(gamma_per_s: float, inertia_kg_m2: float) -> float:
    """`jointDamping` (N m s/rad): the measured rate per unit of inertia x I.

    PyBullet applies a torque of -jointDamping x (joint velocity).
    """
    return gamma_per_s * inertia_kg_m2


def coulomb_motor_force(c_rad_s2: float, inertia_kg_m2: float) -> float:
    """Maximum force (torque, N m) of the velocity motor that stands in for dry joint friction.

    A velocity motor with target 0 pushes against any motion with up to this torque and holds
    the joint still when the load is below it, which is how dry friction behaves. The measured
    value is a deceleration (rad/s^2), so the torque is that times the inertia.
    """
    return c_rad_s2 * inertia_kg_m2


def quadratic_drag_coefficient(k_per_unit: float, mass_or_inertia: float) -> float:
    """Drag force (or torque) = coefficient x |v| x v, for a measured rate k per unit of mass or
    inertia. PyBullet has no field for it, so the snippet applies the force every step."""
    return k_per_unit * mass_or_inertia


def restitution_at_speed(e0: float, e1: float, speed_m_s: float) -> float:
    """B1's e = e0 - e1 |v| at one impact speed, clipped to a physical value."""
    return min(max(e0 - e1 * speed_m_s, 0.0), 1.0)


def restitution_interval_at_speed(
    e0: tuple[float, float, float], e1: tuple[float, float, float], speed_m_s: float
) -> tuple[float, float, float]:
    """(value, low, high) of e0 - e1 |v| from the (value, low, high) of e0 and e1.

    The two coefficients are correlated, but only their marginal intervals are reported. Adding
    them in quadrature, as if independent, gives a wider interval than the true one, which is
    the safe side for a check that asks whether a simulated value falls inside it.
    """
    z = 1.959964
    sigma0 = (e0[2] - e0[1]) / (2 * z)
    sigma1 = (e1[2] - e1[1]) / (2 * z)
    value = restitution_at_speed(e0[0], e1[0], speed_m_s)
    half = z * math.hypot(sigma0, speed_m_s * sigma1)
    return value, max(value - half, 0.0), min(value + half, 1.0)


def bounce_height_ratio(e: float) -> float:
    """Successive bounce heights shrink by e squared: speed scales by e, height by e^2."""
    return e * e
