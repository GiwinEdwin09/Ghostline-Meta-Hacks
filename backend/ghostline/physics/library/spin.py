"""Spin-down models S0-S3 (build plan Part 10). The observable is the unwrapped angle phi."""

from __future__ import annotations

import math

import numpy as np

from .base import ModelDef, ParamSpec, clip_spec, poly_start
from .numerics import phi1, phi2

_NAMES = {
    0: (
        "Viscous friction",
        "Friction proportional to spin rate, like a paddle in honey: exponential decay.",
    ),
    1: (
        "Coulomb friction",
        "A constant friction torque, like a brake pad: the spin falls linearly to a stop.",
    ),
    2: ("Viscous plus Coulomb", "Both a constant and a speed-proportional friction torque."),
    3: (
        "Coulomb plus air drag",
        "A constant friction torque plus air drag proportional to spin rate squared.",
    ),
}
_LATEX = {
    0: r"\dot\omega = -b\,\omega",
    1: r"\dot\omega = -c",
    2: r"\dot\omega = -c - b\,\omega",
    3: r"\dot\omega = -c - k\,\omega^2",
}
_SIGNATURE = {
    0: "Spin rate halves in equal times and never quite stops.",
    1: "Spin rate falls on a straight line and stops at a definite time.",
    2: "Exponential decay at high speed, linear fall and a definite stop at low speed.",
    3: "Fast initial drop from air drag, then a linear fall to a definite stop.",
}


def _stop_time_viscous_coulomb(omega0: float, b: float, c: float) -> float:
    if c <= 1e-12:
        return math.inf
    x = b * omega0 / c
    ratio = 1 - x / 2 + x**2 / 3 if x < 1e-8 else math.log1p(x) / x
    return omega0 / c * ratio


def _magnitude(model: Spin, tau: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Angle swept since the start of the window, for a spin that starts positive."""
    w0 = max(p["omega0"], 0.0)
    v = model.variant
    b = p.get("b", 0.0)
    c = p.get("c", 0.0)
    k = p.get("k", 0.0)
    if v == 0:
        return w0 * tau * phi1(b * tau)
    if v == 1 or (v == 3 and k < 1e-12):
        stop = w0 / c if c > 1e-12 else math.inf
        te = np.minimum(tau, stop)
        return w0 * te - 0.5 * c * te**2
    if v == 2:
        te = np.minimum(tau, _stop_time_viscous_coulomb(w0, b, c))
        return w0 * te * phi1(b * te) - c * te**2 * phi2(b * te)
    c = max(c, 1e-12)
    root = math.sqrt(k * c)
    theta0 = math.atan(w0 * math.sqrt(k / c))
    te = np.minimum(tau, theta0 / root)
    delta = root * te
    tan0 = w0 * math.sqrt(k / c)
    ratio_minus_one = -2 * np.sin(delta / 2) ** 2 + tan0 * np.sin(delta)
    return np.log1p(ratio_minus_one) / k


class Spin(ModelDef):
    def __init__(self, variant: int):
        self.variant = variant
        self.id = f"S{variant}"
        self.name, self.mechanism = _NAMES[variant]
        self.equations_latex = _LATEX[variant]
        self.signature = _SIGNATURE[variant]
        self.system = "spin_down"
        self.observables = ("phi",)
        self.observable_units = {"phi": "rad"}
        self.state = ("phi", "omega")
        self.object_kinds = ("spinner", "wheel")
        self.params = self._default_params()

    def _default_params(self) -> dict[str, ParamSpec]:
        ic = "initial_condition"
        params = {
            "phi0": ParamSpec(init=0.0, lower=-1e4, upper=1e4, unit="rad", kind=ic),
            "omega0": ParamSpec(init=10.0, lower=0.0, upper=400.0, unit="rad/s", kind=ic),
        }
        if self.variant in (0, 2):
            params["b"] = ParamSpec(init=0.2, lower=0.0, upper=20.0, unit="1/s")
        if self.variant in (1, 2, 3):
            params["c"] = ParamSpec(init=1.0, lower=0.0, upper=400.0, unit="rad/s^2")
        if self.variant == 3:
            params["k"] = ParamSpec(init=0.002, lower=0.0, upper=1.0, unit="1/rad")
        return params

    def prepare(self, tau, obs, constants):
        out = dict(constants)
        if "direction" not in out:
            out["direction"] = -1.0 if obs["phi"][-1] < obs["phi"][0] else 1.0
        return out

    def initial_state(self, p):
        return np.array([p["phi0"], max(p["omega0"], 0.0)])

    def rhs(self, t, s, p):
        w = max(s[1], 0.0)
        d = p.get("direction", 1.0)
        v = self.variant
        decel = 0.0
        if v in (0, 2):
            decel += p["b"] * w
        if v in (1, 2, 3):
            decel += p["c"] if w > 0 else 0.0
        if v == 3:
            decel += p["k"] * w * w
        return [d * w, -decel]

    def closed_form(self, tau, p):
        d = p.get("direction", 1.0)
        return (p["phi0"] + d * _magnitude(self, tau, p))[None, :]

    def omega_curve(self, tau, p) -> np.ndarray:
        """Signed angular velocity, for the omega(t) plot."""
        eps = 1e-4 * max(float(tau[-1] - tau[0]), 1e-3)
        d = p.get("direction", 1.0)
        return d * (_magnitude(self, tau + eps, p) - _magnitude(self, tau - eps, p)) / (2 * eps)

    def derived(self, p):
        w0, b, c = max(p["omega0"], 0.0), p.get("b", 0.0), p.get("c", 0.0)
        if self.variant == 1 and c > 1e-12:
            return {"time_to_stop": w0 / c}
        if self.variant == 2:
            stop = _stop_time_viscous_coulomb(w0, b, c)
            return {"time_to_stop": stop} if math.isfinite(stop) else {}
        if self.variant == 0 and b > 1e-12:
            return {"half_life": math.log(2) / b}
        return {}

    def derived_units(self):
        return {"time_to_stop": "s", "half_life": "s"}

    def fit_setup(self, tau, obs, constants, segment=None):
        specs = self._default_params()
        phi = obs["phi"]
        d = constants.get("direction", 1.0)
        span = max(float(tau[-1] - tau[0]), 1e-3)
        head = max(8, min(20, tau.size // 6))
        c0, c1, _ = poly_start(tau, d * phi, 2, head=head)
        w0 = max(c1, 0.1)
        tail = d * (phi[-1] - phi[-head]) / max(float(tau[-1] - tau[-head]), 1e-6)
        w_end = float(np.clip(tail, 0.02 * w0, w0))
        specs["phi0"] = clip_spec(phi[0], -1e4, 1e4, "rad", "initial_condition")
        specs["omega0"] = clip_spec(w0, 0.0, 400.0, "rad/s", "initial_condition")
        decay = math.log(w0 / w_end) / span
        linear = (w0 - w_end) / span
        if "b" in specs:
            specs["b"] = clip_spec(decay if self.variant == 0 else 0.5 * decay, 0.0, 20.0, "1/s")
        if "c" in specs:
            c = linear if self.variant == 1 else 0.5 * linear
            specs["c"] = clip_spec(c, 0.0, 400.0, "rad/s^2")
        if "k" in specs:
            specs["k"] = clip_spec(0.5 * linear / max(w0**2, 1e-6), 0.0, 1.0, "1/rad")
        return specs
