"""Free fall and projectile models FF0-FF2 (build plan Part 10)."""

from __future__ import annotations

import math

import numpy as np

from .base import ModelDef, ParamSpec, clip_spec, poly_start
from .numerics import linear_drag_axis, log_cosh

_NAMES = {
    0: ("Constant gravity", "Gravity is the only force, so the acceleration is constant."),
    1: (
        "Linear drag",
        "Gravity plus air drag proportional to speed, so the ball approaches a terminal speed.",
    ),
    2: (
        "Quadratic drag",
        "Gravity plus air drag proportional to speed squared, the usual drag for a ball in air.",
    ),
}
_LATEX = {
    0: r"\ddot{\vec r} = \vec g",
    1: r"\dot{\vec v} = \vec g - \tfrac{b}{m}\vec v",
    2: r"\dot{\vec v} = \vec g - \tfrac{k}{m}\,|\vec v|\,\vec v",
}
_SIGNATURE = {
    0: "Acceleration stays constant, so the residuals of a parabola are pure noise.",
    1: "Acceleration shrinks steadily as speed grows; speed saturates at g/b.",
    2: "Acceleration shrinks faster at high speed; speed saturates at sqrt(g/k).",
}


class FreeFall(ModelDef):
    """FF0 constant g, FF1 linear drag, FF2 quadratic drag. `planar` adds the horizontal axis."""

    def __init__(self, variant: int, planar: bool = False):
        self.variant = variant
        self.planar = planar
        self.id = f"FF{variant}"
        self.name, self.mechanism = _NAMES[variant]
        self.equations_latex = _LATEX[variant]
        self.signature = _SIGNATURE[variant]
        self.system = "projectile" if planar else "free_fall"
        self.observables = ("x", "y") if planar else ("y",)
        self.observable_units = {o: "m" for o in self.observables}
        self.state = ("x", "y", "vx", "vy") if planar else ("y", "vy")
        self.object_kinds = ("ball", "projectile")
        self.params = self._default_params()

    def _default_params(self) -> dict[str, ParamSpec]:
        ic = "initial_condition"
        params: dict[str, ParamSpec] = {}
        if self.planar:
            params["x0"] = ParamSpec(init=0.0, lower=-10, upper=10, unit="m", kind=ic)
        params["y0"] = ParamSpec(init=0.0, lower=-10, upper=10, unit="m", kind=ic)
        if self.planar:
            params["vx0"] = ParamSpec(init=0.0, lower=-30, upper=30, unit="m/s", kind=ic)
        params["vy0"] = ParamSpec(init=0.0, lower=-30, upper=30, unit="m/s", kind=ic)
        params["g"] = ParamSpec(init=9.81, lower=1.0, upper=30.0, unit="m/s^2")
        if self.planar:
            params["tilt"] = ParamSpec(init=0.0, lower=-0.2, upper=0.2, unit="rad", kind="nuisance")
        if self.variant == 1:
            params["b"] = ParamSpec(init=0.1, lower=0.0, upper=20.0, unit="1/s")
        if self.variant == 2:
            params["k"] = ParamSpec(init=0.05, lower=0.0, upper=10.0, unit="1/m")
        return params

    def initial_state(self, p):
        if self.planar:
            return np.array([p["x0"], p["y0"], p["vx0"], p["vy0"]])
        return np.array([p["y0"], p["vy0"]])

    def _gravity(self, p) -> tuple[float, float]:
        if not self.planar:
            return 0.0, -p["g"]
        return p["g"] * math.sin(p["tilt"]), -p["g"] * math.cos(p["tilt"])

    def rhs(self, t, s, p):
        gx, gy = self._gravity(p)
        if self.planar:
            vx, vy = s[2], s[3]
        else:
            vx, vy = 0.0, s[1]
        if self.variant == 0:
            ax, ay = gx, gy
        elif self.variant == 1:
            ax, ay = gx - p["b"] * vx, gy - p["b"] * vy
        else:
            speed = math.hypot(vx, vy)
            ax, ay = gx - p["k"] * speed * vx, gy - p["k"] * speed * vy
        return [vx, vy, ax, ay] if self.planar else [vy, ay]

    def closed_form(self, tau, p):
        gx, gy = self._gravity(p)
        if self.variant == 2:
            return self._quadratic_drag_closed_form(tau, p)
        b = p["b"] if self.variant == 1 else 0.0
        y = linear_drag_axis(p["y0"], p["vy0"], gy, b, tau)
        if not self.planar:
            return y[None, :]
        x = linear_drag_axis(p["x0"], p["vx0"], gx, b, tau)
        return np.vstack([x, y])

    def _quadratic_drag_closed_form(self, tau, p):
        """Downward fall from below terminal speed; anything else goes through the ODE."""
        k, g = p["k"], p["g"]
        if self.planar or k < 1e-7:
            return None
        u0 = -p["vy0"]
        terminal = math.sqrt(g / k)
        if not 0.0 <= u0 < 0.999 * terminal:
            return None
        a = math.sqrt(g * k)
        c = math.atanh(u0 / terminal)
        y = p["y0"] - (log_cosh(a * tau + c) - log_cosh(c)) / k
        return y[None, :]

    def derived(self, p):
        if self.variant == 1 and p["b"] > 1e-9:
            return {"terminal_velocity": p["g"] / p["b"]}
        if self.variant == 2 and p["k"] > 1e-9:
            return {"terminal_velocity": math.sqrt(p["g"] / p["k"])}
        return {}

    def derived_units(self):
        return {"terminal_velocity": "m/s"} if self.variant else {}

    def fit_setup(self, tau, obs, constants, segment=None):
        specs = self._default_params()
        y = obs["y"]
        span = max(float(tau[-1] - tau[0]), 1e-3)
        reach = max(float(np.ptp(y)), 0.05)
        vmax = max(5.0, 6.0 * reach / span)
        # With drag the whole-window curvature underestimates g; the first quarter is nearer to it.
        head = max(8, tau.size // 4) if self.variant else None
        c0, c1, c2 = poly_start(tau, y, 2, head=head)
        specs["y0"] = clip_spec(c0, y.min() - reach, y.max() + reach, "m", "initial_condition")
        specs["vy0"] = clip_spec(c1, -vmax, vmax, "m/s", "initial_condition")
        g_est = float(np.clip(-2 * c2, 1.0, 30.0))
        specs["g"] = clip_spec(g_est, 1.0, 30.0, "m/s^2", "physical")
        if self.variant:
            # Drag coefficient from how much the late acceleration falls short of g.
            m = max(8, tau.size // 4)
            _, d1, d2 = poly_start(tau[-m:] - tau[-m], y[-m:], 2)
            speed = abs(d1 + d2 * (tau[-1] - tau[-m]))
            shortfall = g_est + 2 * d2
            if shortfall > 0.05 * g_est and speed > 0.1:
                name, power = ("b", 1) if self.variant == 1 else ("k", 2)
                spec = specs[name]
                specs[name] = clip_spec(shortfall / speed**power, spec.lower, spec.upper, spec.unit)
        if self.planar:
            x = obs["x"]
            xreach = max(float(np.ptp(x)), 0.05)
            x0, vx0 = poly_start(tau, x, 1)
            specs["x0"] = clip_spec(
                x0, x.min() - xreach, x.max() + xreach, "m", "initial_condition"
            )
            specs["vx0"] = clip_spec(vx0, -vmax, vmax, "m/s", "initial_condition")
        return specs


FREE_FALL_IDS = ("FF0", "FF1", "FF2")


def build(model_id: str, planar: bool = False) -> FreeFall:
    return FreeFall(int(model_id[2:]), planar=planar)
