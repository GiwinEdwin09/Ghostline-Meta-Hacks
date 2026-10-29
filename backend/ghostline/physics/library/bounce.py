"""Bounce models B0-B2 (build plan Part 10).

A bounce is a chain of flights. Each flight is a parabola (B0, B1) or a quadratic-drag arc (B2),
all in closed form, and each impact turns the speed v into e(v) * v.
"""

from __future__ import annotations

import math

import numpy as np

from .base import ModelDef, ParamSpec, clip_spec
from .numerics import log_cosh

_NAMES = {
    0: ("Constant restitution", "Every impact returns the same fraction e of the impact speed."),
    1: (
        "Speed-dependent restitution",
        "The ball returns less of its speed at harder impacts: e falls linearly with impact speed.",
    ),
    2: (
        "Constant restitution plus air drag",
        "Every impact returns the same fraction e, and air drag slows each flight.",
    ),
}
_LATEX = {
    0: r"v_{out} = e\,v_{in}",
    1: r"v_{out} = (e_0 - e_1 v_{in})\,v_{in}",
    2: r"v_{out} = e\,v_{in},\quad \dot v = -g - \tfrac{k}{m}|v|v",
}
_SIGNATURE = {
    0: "Successive bounce heights shrink by the same factor e squared each time.",
    1: "The height ratio between bounces is smaller for the first, harder impacts.",
    2: "Flights are slightly asymmetric: the way up is faster than the way down.",
}
MAX_IMPACTS = 60
REST_SPEED = 1e-3


def flight_labels(tau: np.ndarray, y: np.ndarray, segment: np.ndarray | None) -> np.ndarray:
    """Flight index per sample: the given segment ids, or cuts at the minima of y."""
    if segment is not None and np.unique(segment).size > 1:
        _, labels = np.unique(segment, return_inverse=True)
        return labels
    near_floor = y < y.min() + 0.1 * max(float(np.ptp(y)), 1e-6)
    labels = np.zeros(y.size, dtype=int)
    for i in range(1, y.size - 1):
        if near_floor[i] and y[i] <= y[i - 1] and y[i] < y[i + 1]:
            labels[i + 1 :] += 1
    return labels


def flight_parabolas(tau: np.ndarray, y: np.ndarray, labels: np.ndarray):
    """Fit y = c0 + c1 t + c2 t^2 to each flight: a list of (t_start, t_end, coefficients)."""
    out = []
    for j in np.unique(labels):
        mask = labels == j
        if mask.sum() < 4:
            continue
        t0 = float(tau[mask][0])
        coef = np.polynomial.polynomial.polyfit(tau[mask] - t0, y[mask], 2)
        out.append((t0, float(tau[mask][-1]), coef))
    return out


class Bounce(ModelDef):
    by_flight = True

    def __init__(self, variant: int):
        self.variant = variant
        self.id = f"B{variant}"
        self.name, self.mechanism = _NAMES[variant]
        self.equations_latex = _LATEX[variant]
        self.signature = _SIGNATURE[variant]
        self.system = "bounce"
        self.observables = ("y",)
        self.observable_units = {"y": "m"}
        self.state = ("y", "vy")
        self.object_kinds = ("ball",)
        self.params = self._default_params()

    def _default_params(self) -> dict[str, ParamSpec]:
        ic = "initial_condition"
        params = {
            "y0": ParamSpec(init=1.0, lower=-5.0, upper=10.0, unit="m", kind=ic),
            "v0": ParamSpec(init=0.0, lower=-15.0, upper=15.0, unit="m/s", kind=ic),
            "y_floor": ParamSpec(init=0.0, lower=-5.0, upper=10.0, unit="m", kind="nuisance"),
            "g": ParamSpec(init=9.81, lower=1.0, upper=30.0, unit="m/s^2"),
        }
        if self.variant == 1:
            params["e0"] = ParamSpec(init=0.8, lower=0.0, upper=1.0, unit="")
            params["e1"] = ParamSpec(init=0.01, lower=0.0, upper=0.3, unit="s/m")
        else:
            params["e"] = ParamSpec(init=0.8, lower=0.0, upper=1.0, unit="")
        if self.variant == 2:
            params["k"] = ParamSpec(init=0.02, lower=0.0, upper=2.0, unit="1/m")
        return params

    def restitution(self, v_in: float, p) -> float:
        if self.variant == 1:
            return float(np.clip(p["e0"] - p["e1"] * v_in, 0.0, 0.999))
        return min(p["e"], 0.999)

    def _flight(self, y_start: float, v_start: float, p):
        """One flight to the floor: (duration, impact speed, position(local time))."""
        g, floor = p["g"], p["y_floor"]
        k = p.get("k", 0.0) if self.variant == 2 else 0.0
        height = max(y_start - floor, 0.0)
        if k < 1e-7:
            disc = math.sqrt(v_start * v_start + 2 * g * height)
            duration = (v_start + disc) / g
            return duration, disc, lambda tl: y_start + v_start * tl - 0.5 * g * tl**2
        a = math.sqrt(g * k)
        terminal = math.sqrt(g / k)
        if v_start > 0:
            theta0 = math.atan(v_start / terminal)
            rise = theta0 / a
            y_apex = y_start - math.log(math.cos(theta0)) / k
            c = 0.0
        else:
            rise, y_apex = 0.0, y_start
            c = math.atanh(min(-v_start / terminal, 0.999))
        h_apex = max(y_apex - floor, 0.0)
        big_e = math.exp(log_cosh(c) + k * h_apex)
        fall = (math.acosh(max(big_e, 1.0)) - c) / a
        v_in = terminal * math.sqrt(max(1.0 - 1.0 / big_e**2, 0.0))

        def position(tl):
            tl = np.asarray(tl, dtype=float)
            fall_t = np.maximum(tl - rise, 0.0)
            down = y_apex - (log_cosh(a * fall_t + c) - log_cosh(c)) / k
            if not rise:
                return down
            swing = np.log(np.cos(theta0 - a * np.minimum(tl, rise))) - math.log(math.cos(theta0))
            return np.where(tl < rise, y_start + swing / k, down)

        return rise + fall, v_in, position

    def closed_form(self, tau, p):
        tau = np.asarray(tau, dtype=float)
        y_out = np.full(tau.shape, p["y_floor"], dtype=float)
        y = max(p["y0"], p["y_floor"])
        v = p["v0"]
        t_start = 0.0
        t_end = float(tau[-1]) if tau.size else 0.0
        for _ in range(MAX_IMPACTS):
            duration, v_in, position = self._flight(y, v, p)
            inside = (tau >= t_start) & (tau < t_start + duration)
            if inside.any():
                y_out[inside] = position(tau[inside] - t_start)
            t_start += duration
            v = self.restitution(v_in, p) * v_in
            y = p["y_floor"]
            if t_start > t_end or v < REST_SPEED:
                break
        return y_out[None, :]

    def initial_state(self, p):
        return np.array([p["y0"], p["v0"]])

    def rhs(self, t, s, p):
        drag = p["k"] * abs(s[1]) * s[1] if self.variant == 2 else 0.0
        return [s[1], -p["g"] - drag]

    def derived(self, p):
        if self.variant == 1:
            return {"e_at_2ms": float(np.clip(p["e0"] - 2.0 * p["e1"], 0.0, 1.0))}
        if self.variant == 2 and p["k"] > 1e-9:
            return {"terminal_velocity": math.sqrt(p["g"] / p["k"])}
        return {}

    def derived_units(self):
        return {"e_at_2ms": "", "terminal_velocity": "m/s"}

    def fit_setup(self, tau, obs, constants, segment=None):
        specs = self._default_params()
        y = obs["y"]
        labels = flight_labels(tau, y, segment)
        flights = flight_parabolas(tau, y, labels)
        reach = max(float(np.ptp(y)), 0.05)
        floor = float(y.min())
        g_est, speed_in, speed_out = 9.81, [], []
        if len(flights) >= 2:
            g_est = float(np.median([-2 * f[2][2] for f in flights]))
            for (t0a, t1a, ca), (t0b, _, cb) in zip(flights[:-1], flights[1:], strict=False):
                t_imp = 0.5 * (t1a + t0b)
                v_in = -(ca[1] + 2 * ca[2] * (t_imp - t0a))
                v_out = cb[1] + 2 * cb[2] * (t_imp - t0b)
                if v_in > 0.2 and v_out > 0:
                    speed_in.append(v_in)
                    speed_out.append(v_out)
            floors = [
                np.polynomial.polynomial.polyval(0.5 * (a[1] + b[0]) - a[0], a[2])
                for a, b in zip(flights[:-1], flights[1:], strict=False)
            ]
            floor = float(np.median(floors))
        v_start = float(flights[0][2][1]) if flights else 0.0
        e_est = (
            float(np.clip(np.median(np.array(speed_out) / np.array(speed_in)), 0.05, 0.98))
            if speed_in
            else 0.7
        )
        specs["y0"] = clip_spec(y[0], y.min() - reach, y.max() + reach, "m", "initial_condition")
        specs["v0"] = clip_spec(v_start, -15.0, 15.0, "m/s", "initial_condition")
        specs["y_floor"] = clip_spec(floor, y.min() - reach, y.min() + 0.5 * reach, "m", "nuisance")
        specs["g"] = clip_spec(g_est, 1.0, 30.0, "m/s^2")
        if "e" in specs:
            specs["e"] = clip_spec(e_est, 0.0, 1.0, "")
        if "e0" in specs:
            specs["e0"] = clip_spec(e_est + 0.02, 0.0, 1.0, "")
            slope = 0.01
            if len(speed_in) >= 3:
                slope = -float(np.polyfit(speed_in, np.array(speed_out) / np.array(speed_in), 1)[0])
            specs["e1"] = clip_spec(slope, 0.0, 0.3, "s/m")
        return specs
