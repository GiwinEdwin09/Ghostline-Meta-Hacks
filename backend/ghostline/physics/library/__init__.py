"""Competing models for each system (build plan Part 10)."""

from __future__ import annotations

from . import bounce, freefall, oscillator, pendulum, ramp, spin
from .base import ModelDef, ParamSpec, integrate, ode_tolerance

__all__ = [
    "ModelDef",
    "ParamSpec",
    "catalog",
    "get_model",
    "integrate",
    "model_ids",
    "models_for",
    "ode_tolerance",
]

_FACTORIES = {
    "FF": lambda variant, planar: freefall.FreeFall(variant, planar=planar),
    "P": lambda variant, planar: pendulum.Pendulum(variant),
    "R": lambda variant, planar: ramp.Ramp(variant),
    "B": lambda variant, planar: bounce.Bounce(variant),
    "S": lambda variant, planar: spin.Spin(variant),
    "O": lambda variant, planar: oscillator.Oscillator(variant),
}
_SYSTEM_IDS = {
    "free_fall": ["FF0", "FF1", "FF2"],
    "projectile": ["FF0", "FF1", "FF2"],
    "pendulum": ["P0", "P1", "P2", "P3", "P4"],
    "ramp": ["R0", "R1", "R2", "R3"],
    "bounce": ["B0", "B1", "B2"],
    "spin_down": ["S0", "S1", "S2", "S3"],
    "oscillator": ["O0", "O1"],
}
_ALL_IDS = {model_id for ids in _SYSTEM_IDS.values() for model_id in ids}


def get_model(model_id: str, system: str | None = None) -> ModelDef:
    """A fresh library model. `system="projectile"` selects the two-axis free-fall variants."""
    if model_id not in _ALL_IDS:
        raise KeyError(f"unknown library model {model_id!r}")
    prefix = model_id.rstrip("0123456789")
    return _FACTORIES[prefix](int(model_id[len(prefix) :]), system == "projectile")


def model_ids(system: str) -> list[str]:
    return list(_SYSTEM_IDS.get(system, []))


def models_for(system: str, object_kind: str | None = None) -> list[ModelDef]:
    """Library models for a system, optionally only those that suit the object."""
    models = [get_model(i, system) for i in model_ids(system)]
    if object_kind is None:
        return models
    return [m for m in models if not m.object_kinds or object_kind in m.object_kinds]


def catalog(system: str | None = None) -> list[dict]:
    """Catalog entries for the agent's `list_library_models` tool."""
    systems = [system] if system else list(_SYSTEM_IDS)
    return [m.catalog_entry() for s in systems for m in models_for(s)]
