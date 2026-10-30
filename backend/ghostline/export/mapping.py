"""Which library parameters are exported, under what Ghostline JSON field (build plan Part 19).

A model is exportable when its fitted parameters have a counterpart in a simulator. Free-fall
without drag, the frictionless ramp and the small-angle pendulum measure nothing a simulator
needs, and R2 (rolling without slipping) fits a shape factor, which is geometry, not a material
property.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_OF_PREFIX = {
    "FF": "free_fall",
    "P": "pendulum",
    "R": "ramp",
    "B": "bounce",
    "S": "spin_down",
    "O": "oscillator",
}


@dataclass(frozen=True)
class Mapping:
    source: str  # the library parameter name
    field: str  # the Ghostline JSON field (the exported parameter's name)
    unit: str
    definition: str


_MU_K = Mapping(
    "mu_k",
    "friction.kinetic",
    "",
    "Coefficient of sliding (kinetic) friction of the object on the surface (mu_k).",
)
_C_RR = Mapping(
    "c_rr",
    "friction.rolling",
    "",
    "Rolling-resistance coefficient: the resisting force divided by the normal force (c_rr).",
)
_E = Mapping(
    "e",
    "restitution",
    "",
    "Coefficient of restitution: the fraction of the impact speed kept after each bounce (e).",
)
_E0 = Mapping(
    "e0", "restitution.e0", "", "Restitution at zero impact speed in e = e0 - e1 |v| (e0)."
)
_E1 = Mapping(
    "e1", "restitution.e1", "s/m", "Fall of restitution per m/s of impact speed in e = e0 - e1 |v| (e1)."
)
_B = Mapping(
    "b",
    "damping.linear_rate",
    "1/s",
    "Viscous drag rate b/m: the speed decays as e^(-rate t) (lambda).",
)
_K_LINEAR = Mapping(
    "k",
    "drag.quadratic",
    "1/m",
    "Quadratic air drag per unit mass: deceleration = k |v| v (k/m).",
)
_GAMMA = Mapping(
    "gamma",
    "joint.damping",
    "1/s",
    "Viscous joint damping per unit of inertia: torque = -gamma I omega (gamma).",
)
_KAPPA = Mapping(
    "kappa",
    "drag.quadratic",
    "1/rad",
    "Quadratic air drag on the swing per unit of inertia: torque = -kappa I |omega| omega (kappa).",
)
_MU_P = Mapping(
    "mu_p",
    "joint.coulomb_torque",
    "rad/s^2",
    "Dry friction at the pivot as a deceleration, torque per unit of inertia (mu_p).",
)
_SPIN_B = Mapping(
    "b",
    "joint.damping",
    "1/s",
    "Viscous friction per unit of inertia: torque = -b I omega (b).",
)
_SPIN_C = Mapping(
    "c",
    "joint.coulomb_torque",
    "rad/s^2",
    "Dry friction as a deceleration, torque per unit of inertia (c).",
)
_SPIN_K = Mapping(
    "k",
    "drag.quadratic",
    "1/rad",
    "Quadratic air drag per unit of inertia: torque = -k I omega^2 (k).",
)

# Winning library model -> the parameters it exports, in display order.
EXPORTS: dict[str, tuple[Mapping, ...]] = {
    "FF1": (_B,),
    "FF2": (_K_LINEAR,),
    "R1": (_MU_K,),
    "R3": (_C_RR,),
    "B0": (_E,),
    "B1": (_E0, _E1),
    "B2": (_E, _K_LINEAR),
    "P2": (_GAMMA,),
    "P3": (_KAPPA,),
    "P4": (_MU_P,),
    "S0": (_SPIN_B,),
    "S1": (_SPIN_C,),
    "S2": (_SPIN_B, _SPIN_C),
    "S3": (_SPIN_C, _SPIN_K),
}

# Models whose export can be replayed in PyBullet, and what the replay rebuilds.
REPLAY_SCENARIOS: dict[str, str] = {
    "FF1": "free_fall",
    "FF2": "free_fall",
    "R1": "ramp",
    "B0": "bounce",
    "B1": "bounce",
    "B2": "bounce",
    "P2": "pendulum",
    "P3": "pendulum",
    "P4": "pendulum",
}
# Systems that need the object's moment of inertia (mass, and for a pendulum the geometry).
NEEDS_INERTIA = {"pendulum", "spin_down"}


def system_of(model_id: str) -> str:
    return SYSTEM_OF_PREFIX[model_id.rstrip("0123456789")]


def is_exportable(model_id: str | None) -> bool:
    return model_id in EXPORTS


def export_formats(model_id: str | None) -> list[str]:
    """The formats `GET /export` can produce for the winner: none if it measures nothing a
    simulator takes, JSON only for a spinner (its inertia can't be seen on video), else both."""
    if not is_exportable(model_id):
        return []
    return ["json"] if system_of(model_id or "") == "spin_down" else ["json", "pybullet"]
