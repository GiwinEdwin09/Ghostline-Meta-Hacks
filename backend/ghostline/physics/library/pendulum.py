"""Pendulum models P0-P4 (build plan Part 10). The observable is the angle from vertical."""

from __future__ import annotations

import math

import numpy as np

from .base import ModelDef, ParamSpec, clip_spec, poly_start

COULOMB_EPS = 0.05  # rad/s, width of the tanh that smooths the pivot-friction sign flip

_NAMES = {
    0: (
        "Small-angle SHM",
        "The bob swings like a spring: the restoring force is proportional to the angle.",
    ),
    1: ("Nonlinear pendulum", "The restoring force is g sin(angle), so big swings take longer."),
    2: ("Viscous damping", "The nonlinear pendulum plus air or pivot drag proportional to speed."),
    3: (
        "Quadratic air drag",
        "The nonlinear pendulum plus air drag proportional to speed squared.",
    ),
    4: (
        "Pivot friction",
        "The nonlinear pendulum plus a constant dry-friction torque at the pivot.",
    ),
}
_LATEX = {
    0: r"\ddot\theta = -\tfrac{g}{L}\,\theta",
    1: r"\ddot\theta = -\tfrac{g}{L}\sin\theta",
    2: r"\ddot\theta = -\tfrac{g}{L}\sin\theta - \gamma\dot\theta",
    3: r"\ddot\theta = -\tfrac{g}{L}\sin\theta - \kappa|\dot\theta|\dot\theta",
    4: r"\ddot\theta = -\tfrac{g}{L}\sin\theta - \mu_p\tanh(\dot\theta/\varepsilon)",
}
_SIGNATURE = {
    0: "Period independent of amplitude; swing stays the same size.",
    1: "Period lengthens with amplitude; swing stays the same size.",
    2: "Amplitude decays exponentially, with the same fractional loss every swing.",
    3: "Amplitude decays fast when large, then slowly: loss per swing shrinks with amplitude.",
    4: "Amplitude decays linearly: the same absolute loss every swing, then it stops.",
}


def estimate_frequency(tau: np.ndarray, y: np.ndarray) -> float:
    """Angular frequency of the dominant oscillation, from zero crossings or the FFT."""
    s = np.sign(y)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if idx.size >= 3:
        tc = tau[idx] - y[idx] * (tau[idx + 1] - tau[idx]) / (y[idx + 1] - y[idx])
        half_period = np.polyfit(np.arange(idx.size), tc, 1)[0]
        if half_period > 0:
            return math.pi / half_period
    grid = np.linspace(tau[0], tau[-1], max(256, tau.size))
    yg = np.interp(grid, tau, y - np.mean(y))
    spectrum = np.abs(np.fft.rfft(yg * np.hanning(yg.size)))
    spectrum[0] = 0.0
    k = int(np.argmax(spectrum))
    if 0 < k < spectrum.size - 1:
        a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-300)
        k += 0.5 * (a - c) / (a - 2 * b + c)
    span = grid[-1] - grid[0]
    return 2 * math.pi * k / span if span > 0 else 1.0


class Pendulum(ModelDef):
    def __init__(self, variant: int):
        self.variant = variant
        self.id = f"P{variant}"
        self.name, self.mechanism = _NAMES[variant]
        self.equations_latex = _LATEX[variant]
        self.signature = _SIGNATURE[variant]
        self.system = "pendulum"
        self.observables = ("theta",)
        self.observable_units = {"theta": "rad"}
        self.state = ("theta", "omega")
        self.constants = ("L",)
        self.length_constants = ("L", "L_tape")
        self.object_kinds = ("bob", "pendulum")
        self.tie_rank = 1 if variant == 0 else 0
        self.params = self._default_params()

    def _default_params(self) -> dict[str, ParamSpec]:
        ic = "initial_condition"
        params = {
            "theta0": ParamSpec(init=0.0, lower=-math.pi, upper=math.pi, unit="rad", kind=ic),
            "omega0": ParamSpec(init=0.0, lower=-20.0, upper=20.0, unit="rad/s", kind=ic),
            "g": ParamSpec(init=9.81, lower=1.0, upper=30.0, unit="m/s^2"),
        }
        if self.variant == 2:
            params["gamma"] = ParamSpec(init=0.05, lower=0.0, upper=5.0, unit="1/s")
        if self.variant == 3:
            params["kappa"] = ParamSpec(init=0.05, lower=0.0, upper=5.0, unit="1/rad")
        if self.variant == 4:
            params["mu_p"] = ParamSpec(init=0.05, lower=0.0, upper=2.0, unit="rad/s^2")
        return params

    def initial_state(self, p):
        return np.array([p["theta0"], p["omega0"]])

    def rhs(self, t, s, p):
        theta, omega = s[0], s[1]
        if self.variant == 0:
            acc = -p["g"] / p["L"] * theta
        else:
            acc = -p["g"] / p["L"] * math.sin(theta)
        if self.variant == 2:
            acc -= p["gamma"] * omega
        elif self.variant == 3:
            acc -= p["kappa"] * abs(omega) * omega
        elif self.variant == 4:
            acc -= p["mu_p"] * math.tanh(omega / p.get("eps", COULOMB_EPS))
        return [omega, acc]

    def closed_form(self, tau, p):
        if self.variant != 0:
            return None
        w = math.sqrt(p["g"] / p["L"])
        return (p["theta0"] * np.cos(w * tau) + p["omega0"] / w * np.sin(w * tau))[None, :]

    def derived(self, p):
        out = {"period_small_angle": 2 * math.pi * math.sqrt(p["L"] / p["g"])}
        if "L_tape" in p and p["L_tape"] > 0:
            out["g_tape"] = p["g"] * p["L_tape"] / p["L"]
        return out

    def derived_units(self):
        return {"period_small_angle": "s", "g_tape": "m/s^2"}

    def fit_setup(self, tau, obs, constants, segment=None):
        specs = self._default_params()
        theta = obs["theta"]
        length = constants["L"]
        omega_obs = estimate_frequency(tau, theta)
        # RMS amplitude: with damping the swing shrinks, so the peak overstates the average.
        amplitude = min(float(np.sqrt(2 * np.mean(theta**2))), 2.5)
        omega_small = omega_obs / max(1 - amplitude**2 / 16, 0.3) if self.variant else omega_obs
        c0, c1, _ = poly_start(tau, theta, 2, head=max(8, min(15, tau.size // 8)))
        specs["theta0"] = clip_spec(c0, -math.pi, math.pi, "rad", "initial_condition")
        specs["omega0"] = clip_spec(c1, -20.0, 20.0, "rad/s", "initial_condition")
        # The swing frequency fixes g to a few percent; a narrow box keeps the optimizer out of
        # the neighbouring phase-slipped minima.
        g0 = omega_small**2 * length
        specs["g"] = clip_spec(g0, max(1.0, 0.5 * g0), min(30.0, 2.0 * g0), "m/s^2", "physical")
        return specs
