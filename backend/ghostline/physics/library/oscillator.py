"""Spring-and-mass models O0-O1, the stretch system in build plan Part 10."""

from __future__ import annotations

import math

import numpy as np

from .base import ModelDef, ParamSpec, clip_spec, poly_start
from .pendulum import estimate_frequency

_NAMES = {
    0: (
        "Undamped spring",
        "A mass on a spring with no losses: it oscillates forever at one frequency.",
    ),
    1: (
        "Damped spring",
        "A mass on a spring that also loses energy to drag proportional to speed.",
    ),
}
_LATEX = {
    0: r"\ddot y = -\tfrac{k}{m}(y - y_{eq})",
    1: r"\ddot y = -\tfrac{k}{m}(y - y_{eq}) - \gamma\dot y",
}
_SIGNATURE = {
    0: "Constant amplitude and a fixed period.",
    1: "Amplitude shrinks by the same fraction every cycle.",
}


class Oscillator(ModelDef):
    def __init__(self, variant: int):
        self.variant = variant
        self.id = f"O{variant}"
        self.name, self.mechanism = _NAMES[variant]
        self.equations_latex = _LATEX[variant]
        self.signature = _SIGNATURE[variant]
        self.system = "oscillator"
        self.observables = ("y",)
        self.observable_units = {"y": "m"}
        self.state = ("y", "vy")
        self.object_kinds = ("spring", "mass")
        self.params = self._default_params()

    def _default_params(self) -> dict[str, ParamSpec]:
        ic = "initial_condition"
        params = {
            "y0": ParamSpec(init=0.0, lower=-5.0, upper=5.0, unit="m", kind=ic),
            "v0": ParamSpec(init=0.0, lower=-20.0, upper=20.0, unit="m/s", kind=ic),
            "y_eq": ParamSpec(init=0.0, lower=-5.0, upper=5.0, unit="m", kind="nuisance"),
            "k_over_m": ParamSpec(init=100.0, lower=0.01, upper=1e5, unit="1/s^2"),
        }
        if self.variant == 1:
            params["gamma"] = ParamSpec(init=0.2, lower=0.0, upper=50.0, unit="1/s")
        return params

    def initial_state(self, p):
        return np.array([p["y0"], p["v0"]])

    def rhs(self, t, s, p):
        damping = p["gamma"] * s[1] if self.variant == 1 else 0.0
        return [s[1], -p["k_over_m"] * (s[0] - p["y_eq"]) - damping]

    def closed_form(self, tau, p):
        w2 = p["k_over_m"]
        gamma = p["gamma"] if self.variant == 1 else 0.0
        wd2 = w2 - gamma**2 / 4
        if wd2 <= 1e-12:
            return None
        wd = math.sqrt(wd2)
        x0 = p["y0"] - p["y_eq"]
        decay = np.exp(-0.5 * gamma * tau)
        b = (p["v0"] + 0.5 * gamma * x0) / wd
        return (p["y_eq"] + decay * (x0 * np.cos(wd * tau) + b * np.sin(wd * tau)))[None, :]

    def derived(self, p):
        return {"period": 2 * math.pi / math.sqrt(p["k_over_m"])}

    def derived_units(self):
        return {"period": "s"}

    def fit_setup(self, tau, obs, constants, segment=None):
        specs = self._default_params()
        y = obs["y"]
        reach = max(float(np.ptp(y)), 0.02)
        centre = float(np.mean(y))
        omega = estimate_frequency(tau, y - centre)
        c0, c1, _ = poly_start(tau, y, 2, head=max(8, min(15, tau.size // 8)))
        specs["y0"] = clip_spec(c0, y.min() - reach, y.max() + reach, "m", "initial_condition")
        specs["v0"] = clip_spec(c1, -20.0, 20.0, "m/s", "initial_condition")
        specs["y_eq"] = clip_spec(centre, y.min() - reach, y.max() + reach, "m", "nuisance")
        specs["k_over_m"] = clip_spec(omega**2, 0.01, 1e5, "1/s^2")
        return specs
