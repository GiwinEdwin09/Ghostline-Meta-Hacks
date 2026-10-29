"""Ramp models R0-R3 (build plan Part 10). The observable is s, distance down the incline."""

from __future__ import annotations

import math

import numpy as np

from .base import ModelDef, ParamSpec, clip_spec, poly_start

STANDARD_GRAVITY = 9.80665

_NAMES = {
    0: (
        "Frictionless slide",
        "Nothing resists the motion: only the slope's share of gravity acts.",
    ),
    1: ("Kinetic friction", "A sliding object loses part of the slope's pull to dry friction."),
    2: (
        "Rolling without slipping",
        "A rolling object spends part of the slope's pull on spinning up, so it accelerates less.",
    ),
    3: (
        "Rolling plus rolling resistance",
        "Rolling without slipping, with a small extra drag from the surface giving way.",
    ),
}
_LATEX = {
    0: r"\ddot s = g\sin\alpha",
    1: r"\ddot s = g(\sin\alpha - \mu_k\cos\alpha)",
    2: r"\ddot s = \dfrac{g\sin\alpha}{1+\beta}",
    3: r"\ddot s = \dfrac{g(\sin\alpha - c_{rr}\cos\alpha)}{1+\beta}",
}
_SIGNATURE = {
    0: "Acceleration equals g sin(angle) exactly.",
    1: "Constant acceleration below g sin(angle); the shortfall grows with cos(angle).",
    2: "Constant acceleration g sin(angle)/(1+beta); beta = 2/5 for a solid sphere.",
    3: "Same shape as R2: one run cannot separate c_rr from beta; two angles can.",
}
_OBJECTS = {
    0: ("block", "ball"),
    1: ("block",),
    2: ("sphere", "cylinder", "ball"),
    3: ("sphere", "cylinder", "ball"),
}


class Ramp(ModelDef):
    def __init__(self, variant: int):
        self.variant = variant
        self.id = f"R{variant}"
        self.name, self.mechanism = _NAMES[variant]
        self.equations_latex = _LATEX[variant]
        self.signature = _SIGNATURE[variant]
        self.system = "ramp"
        self.observables = ("s",)
        self.observable_units = {"s": "m"}
        self.state = ("s", "v")
        self.constants = ("alpha", "g")
        self.object_kinds = _OBJECTS[variant]
        self.params = self._default_params()

    def _default_params(self) -> dict[str, ParamSpec]:
        ic = "initial_condition"
        params = {
            "s0": ParamSpec(init=0.0, lower=-5.0, upper=5.0, unit="m", kind=ic),
            "v0": ParamSpec(init=0.0, lower=-5.0, upper=5.0, unit="m/s", kind=ic),
        }
        if self.variant == 1:
            params["mu_k"] = ParamSpec(init=0.2, lower=0.0, upper=1.5, unit="")
        if self.variant in (2, 3):
            params["beta"] = ParamSpec(init=0.4, lower=0.0, upper=1.0, unit="")
        if self.variant == 3:
            params["c_rr"] = ParamSpec(init=0.01, lower=0.0, upper=0.5, unit="")
        return params

    def prepare(self, tau, obs, constants):
        out = dict(constants)
        out.setdefault("g", STANDARD_GRAVITY)
        return out

    def acceleration(self, p) -> float:
        g, alpha = p["g"], p["alpha"]
        sin, cos = math.sin(alpha), math.cos(alpha)
        if self.variant == 0:
            return g * sin
        if self.variant == 1:
            return g * (sin - p["mu_k"] * cos)
        if self.variant == 2:
            return g * sin / (1 + p["beta"])
        return g * (sin - p["c_rr"] * cos) / (1 + p["beta"])

    def initial_state(self, p):
        return np.array([p["s0"], p["v0"]])

    def rhs(self, t, s, p):
        return [s[1], self.acceleration(p)]

    def closed_form(self, tau, p):
        a = self.acceleration(p)
        v0 = p["v0"]
        if a < 0 and v0 > 0:
            tau = np.minimum(tau, v0 / -a)
        return (p["s0"] + v0 * tau + 0.5 * a * tau**2)[None, :]

    def derived(self, p):
        return {"acceleration": self.acceleration(p)}

    def derived_units(self):
        return {"acceleration": "m/s^2"}

    def fit_setup(self, tau, obs, constants, segment=None):
        specs = self._default_params()
        s = obs["s"]
        g, alpha = constants.get("g", STANDARD_GRAVITY), constants["alpha"]
        reach = max(float(np.ptp(s)), 0.05)
        vmax = max(1.0, 6.0 * reach / max(float(tau[-1] - tau[0]), 1e-3))
        c0, c1, c2 = poly_start(tau, s, 2)
        accel = 2 * c2
        specs["s0"] = clip_spec(c0, s.min() - reach, s.max() + reach, "m", "initial_condition")
        specs["v0"] = clip_spec(c1, -vmax, vmax, "m/s", "initial_condition")
        sin, cos = math.sin(alpha), math.cos(alpha)
        if self.variant == 1:
            specs["mu_k"] = clip_spec((sin - accel / g) / cos, 0.0, 1.5, "", "physical")
        if self.variant in (2, 3):
            beta = g * sin / accel - 1 if accel > 1e-6 else 0.4
            specs["beta"] = clip_spec(beta if self.variant == 2 else 0.4, 0.0, 1.0, "", "physical")
        return specs
