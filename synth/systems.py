"""True motion of the five systems, in world coordinates (see camera.py).

Each system returns the geometry to draw at a time `t` and the ground-truth tracks, so the
renderer and the tests share one source of truth. Positions are metres, angles radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

G0 = 9.81
ORANGE = (0, 128, 255)  # BGR
STEEL = (200, 170, 120)
RED = (30, 30, 220)
DARK = (60, 60, 60)
WOOD = (90, 140, 190)
GRAY = (110, 110, 110)


@dataclass
class Prim:
    """A shape in world coordinates: a camera-facing disc or a polygon."""

    kind: str  # "disc" or "poly"
    tag: str  # object | marker | hub | body | string | board | floor | pivot
    color: tuple[int, int, int]
    center: np.ndarray | None = None
    radius: float = 0.0
    pts: np.ndarray | None = None
    shaded: bool = False


def _disc(tag, center, radius, color, shaded=True) -> Prim:
    return Prim("disc", tag, color, center=np.asarray(center, float), radius=radius, shaded=shaded)


def _poly(tag, pts, color) -> Prim:
    return Prim("poly", tag, color, pts=np.asarray(pts, float))


def _segment(tag, p0, p1, width, color, z) -> Prim:
    """A thin rectangle from p0 to p1 (2D points) in the plane at depth z."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-12) * width / 2
    quad = np.array([p0 + n, p1 + n, p1 - n, p0 - n])
    return _poly(tag, np.column_stack([quad, np.full(4, z)]), color)


class _Ode:
    """Dense ODE solution that holds its first state before t0 and its last state after."""

    def __init__(self, rhs, y0, t0, events=None, t_end=120.0):
        self.t0 = t0
        self.sol = solve_ivp(
            rhs, (t0, t_end), y0, method="DOP853", rtol=1e-10, atol=1e-12,
            dense_output=True, events=events,
        )  # fmt: skip
        self.t1 = float(self.sol.t[-1])

    def __call__(self, t) -> np.ndarray:
        return self.sol.sol(np.clip(np.atleast_1d(t), self.t0, self.t1))


class System:
    name = ""
    primary = "object"
    sam_prompt = "orange ball"

    def __init__(self, plane_offset: float, dz_rate: float, t_release: float):
        self.d = plane_offset
        self.dz_rate = dz_rate
        self.t_release = t_release

    def z(self, t) -> np.ndarray:
        return self.d + self.dz_rate * np.maximum(np.atleast_1d(t) - self.t_release, 0.0)

    def static(self) -> list[Prim]:
        return []

    def dynamic(self, t: float) -> list[Prim]:
        raise NotImplementedError

    def truth(self, t: np.ndarray) -> dict:
        """{'tracks': {key: (N, 3) xyz}, 'series': {name: (N,)}}"""
        raise NotImplementedError

    def events(self) -> dict:
        return {"release_t_s": self.t_release}

    def params(self) -> dict:
        raise NotImplementedError


class FreeFall(System):
    name = "free_fall"

    def __init__(
        self, plane_offset=0.02, dz_rate=0.0, t_release=0.4, g=G0, x0=0.22, y0=0.50,
        vx0=0.0, vy0=0.0, drag=0.0, radius=0.02, y_floor=-0.55, color=ORANGE,
    ):  # fmt: skip
        super().__init__(plane_offset, dz_rate, t_release)
        self.g, self.x0, self.y0, self.vx0, self.vy0 = g, x0, y0, vx0, vy0
        self.drag, self.r, self.y_floor, self.color = drag, radius, y_floor, color

        def rhs(_, s):
            v = s[2:]
            a = -self.drag * np.linalg.norm(v) * v - np.array([0.0, self.g])
            return [s[2], s[3], a[0], a[1]]

        def floor(_, s):
            return s[1] - (self.y_floor + self.r)

        floor.terminal, floor.direction = True, -1
        self.ode = _Ode(rhs, [x0, y0, vx0, vy0], t_release, events=[floor] if y_floor else None)

    def _xy(self, t):
        return self.ode(t)[:2].T

    def dynamic(self, t):
        xy = self._xy(t)[0]
        return [_disc("object", [*xy, self.z(t)[0]], self.r, self.color)]

    def truth(self, t):
        xy = self._xy(t)
        return {"tracks": {"object": np.column_stack([xy, self.z(t)])}, "series": {}}

    def events(self):
        out = super().events()
        if self.ode.sol.t_events and len(self.ode.sol.t_events[0]):
            out["stop_t_s"] = float(self.ode.sol.t_events[0][0])
        return out

    def params(self):
        return {
            "g": self.g, "drag_quadratic": self.drag, "x0": self.x0, "y0": self.y0,
            "vx0": self.vx0, "vy0": self.vy0, "radius_m": self.r,
        }  # fmt: skip


class Pendulum(System):
    name = "pendulum"
    sam_prompt = "steel ball"

    def __init__(
        self, plane_offset=0.02, dz_rate=0.0, t_release=0.4, g=G0, length=0.60, theta0=0.35,
        omega0=0.0, gamma=0.0, pivot=(0.38, 0.30), radius=0.02, color=STEEL,
    ):  # fmt: skip
        super().__init__(plane_offset, dz_rate, t_release)
        self.g, self.L, self.theta0, self.omega0 = g, length, theta0, omega0
        self.gamma, self.pivot, self.r, self.color = gamma, np.asarray(pivot, float), radius, color
        self.L_eff = length + 0.4 * radius**2 / length

        def rhs(_, s):
            return [s[1], -(self.g / self.L_eff) * np.sin(s[0]) - self.gamma * s[1]]

        self.ode = _Ode(rhs, [theta0, omega0], t_release)

    def _xy(self, t):
        th = self.ode(t)[0]
        return self.pivot + self.L * np.column_stack([np.sin(th), -np.cos(th)])

    def static(self):
        return [_disc("pivot", [*self.pivot, self.d], 0.008, DARK, shaded=False)]

    def dynamic(self, t):
        xy = self._xy(t)[0]
        z = self.z(t)[0]
        return [
            _segment("string", self.pivot, xy, 0.0015, (235, 235, 235), self.d),
            _disc("object", [*xy, z], self.r, self.color),
        ]

    def truth(self, t):
        xy = self._xy(t)
        z = self.z(t)
        pivot = np.tile([*self.pivot, self.d], (len(xy), 1))
        return {
            "tracks": {"object": np.column_stack([xy, z]), "pivot": pivot},
            "series": {"theta": self.ode(t)[0]},
        }

    def params(self):
        return {
            "g": self.g, "L_m": self.L, "L_eff_m": self.L_eff, "theta0_rad": self.theta0,
            "omega0": self.omega0, "gamma": self.gamma, "pivot": list(self.pivot),
            "radius_m": self.r,
        }  # fmt: skip


class Ramp(System):
    name = "ramp"

    def __init__(
        self, plane_offset=0.02, dz_rate=0.0, t_release=0.4, g=G0, alpha_deg=20.0, mu=0.20,
        mode="block", beta=0.4, top=(0.22, 0.12), board_length=0.80, s0=0.05, radius=0.025,
        block_size=(0.07, 0.04),
    ):  # fmt: skip
        super().__init__(plane_offset, dz_rate, t_release)
        self.g, self.alpha, self.mu, self.mode, self.beta = g, np.radians(alpha_deg), mu, mode, beta
        self.top, self.length, self.s0, self.r = np.asarray(top, float), board_length, s0, radius
        self.block = block_size
        self.sam_prompt = "wooden block" if mode == "block" else "orange ball"
        sa, ca = np.sin(self.alpha), np.cos(self.alpha)
        self.acc = g * (sa - mu * ca) if mode == "block" else g * sa / (1 + beta)
        self.acc = max(self.acc, 0.0)
        self.dir = np.array([ca, -sa])
        self.normal = np.array([sa, ca])
        self.s_max = board_length - (block_size[0] / 2 if mode == "block" else radius)

    def _s(self, t):
        tau = np.maximum(np.atleast_1d(t) - self.t_release, 0.0)
        return np.minimum(self.s0 + 0.5 * self.acc * tau**2, self.s_max)

    def _xy(self, t):
        h = self.block[1] / 2 if self.mode == "block" else self.r
        return self.top + self._s(t)[:, None] * self.dir + h * self.normal

    def static(self):
        thick = 0.02
        a, b = self.top, self.top + self.length * self.dir
        quad = np.array([a, b, b - thick * self.normal, a - thick * self.normal])
        return [_poly("board", np.column_stack([quad, np.full(4, self.d)]), WOOD)]

    def dynamic(self, t):
        c = self._xy(t)[0]
        z = self.z(t)[0]
        if self.mode == "ball":
            return [_disc("object", [*c, z], self.r, ORANGE)]
        w, h = self.block[0] / 2, self.block[1] / 2
        corners = [c + a * self.dir * w + b * self.normal * h for a, b in
                   ((-1, -1), (1, -1), (1, 1), (-1, 1))]  # fmt: skip
        return [_poly("object", np.column_stack([corners, np.full(4, z)]), (60, 100, 230))]

    def truth(self, t):
        xy = self._xy(t)
        return {
            "tracks": {"object": np.column_stack([xy, self.z(t)])},
            "series": {"s": self._s(t) - self.s0},
        }

    def events(self):
        out = super().events()
        if self.acc > 0:
            out["stop_t_s"] = self.t_release + float(np.sqrt(2 * (self.s_max - self.s0) / self.acc))
        return out

    def params(self):
        return {
            "g": self.g, "alpha_deg": float(np.degrees(self.alpha)), "mu_k": self.mu,
            "mode": self.mode, "beta": self.beta, "acceleration": self.acc, "s0": self.s0,
            "top": list(self.top), "board_length": self.length,
        }  # fmt: skip


class Bounce(System):
    name = "bounce"

    def __init__(
        self, plane_offset=0.02, dz_rate=0.0, t_release=0.3, g=G0, y0=0.45, x0=0.22, vx=0.0,
        e=0.75, radius=0.02, y_floor=-0.50, color=ORANGE,
    ):  # fmt: skip
        super().__init__(plane_offset, dz_rate, t_release)
        self.g, self.y0, self.x0, self.vx, self.e = g, y0, x0, vx, e
        self.r, self.y_floor, self.color = radius, y_floor, color
        floor = y_floor + radius
        t = t_release + np.sqrt(2 * (y0 - floor) / g)
        v = g * np.sqrt(2 * (y0 - floor) / g)
        self.impacts = [float(t)]
        self.speeds = [float(v)]  # speed just before each impact
        while True:
            v *= e
            flight = 2 * v / g
            if flight < 0.02:
                break
            t += flight
            self.impacts.append(float(t))
            self.speeds.append(float(v))
        self.t_rest = self.impacts[-1]

    def _y(self, t):
        t = np.atleast_1d(t).astype(float)
        floor = self.y_floor + self.r
        y = np.full(t.shape, self.y0)
        fall = (t >= self.t_release) & (t < self.impacts[0])
        y[fall] = self.y0 - 0.5 * self.g * (t[fall] - self.t_release) ** 2
        for k, t_hit in enumerate(self.impacts):
            t_next = self.impacts[k + 1] if k + 1 < len(self.impacts) else np.inf
            m = (t >= t_hit) & (t < t_next)
            v_up = self.e * self.speeds[k]
            tau = t[m] - t_hit
            y[m] = floor + np.where(np.isfinite(t_next), v_up * tau - 0.5 * self.g * tau**2, 0.0)
        return y

    def _xy(self, t):
        t = np.atleast_1d(t)
        x = self.x0 + self.vx * np.clip(t - self.t_release, 0, self.t_rest - self.t_release)
        return np.column_stack([x, self._y(t)])

    def static(self):
        floor = self.y_floor
        quad = [[-1.5, floor], [1.5, floor], [1.5, floor - 0.1], [-1.5, floor - 0.1]]
        return [_poly("floor", np.column_stack([quad, np.full(4, self.d)]), GRAY)]

    def dynamic(self, t):
        xy = self._xy(t)[0]
        return [_disc("object", [*xy, self.z(t)[0]], self.r, self.color)]

    def truth(self, t):
        return {"tracks": {"object": np.column_stack([self._xy(t), self.z(t)])}, "series": {}}

    def events(self):
        out = super().events()
        out["impacts_t_s"] = self.impacts
        out["stop_t_s"] = self.t_rest
        return out

    def params(self):
        return {
            "g": self.g, "e": self.e, "y0": self.y0, "x0": self.x0, "vx": self.vx,
            "radius_m": self.r, "y_floor": self.y_floor, "impact_speeds": self.speeds,
        }  # fmt: skip


class SpinDown(System):
    name = "spin_down"
    primary = "marker"
    sam_prompt = "red dot sticker"

    def __init__(
        self, plane_offset=0.008, dz_rate=0.0, t_release=0.3, hub=(0.28, 0.0), rho=0.035,
        omega0=40.0, coulomb=2.0, viscous=0.25, phi0=0.5, arm_radius=0.015,
    ):  # fmt: skip
        super().__init__(plane_offset, dz_rate, t_release)
        self.hub, self.rho, self.omega0 = np.asarray(hub, float), rho, omega0
        self.c, self.b, self.phi0, self.arm_r = coulomb, viscous, phi0, arm_radius
        # omega' = -c - b*omega, clamped at 0
        self.t_stop_rel = (
            np.log1p(omega0 * viscous / coulomb) / viscous if viscous > 0 else omega0 / coulomb
        )

    def _phi(self, t):
        tau = np.clip(np.atleast_1d(t) - self.t_release, 0.0, self.t_stop_rel)
        if self.b > 0:
            k = self.omega0 + self.c / self.b
            ph = k * (1 - np.exp(-self.b * tau)) / self.b - self.c / self.b * tau
        else:
            ph = self.omega0 * tau - 0.5 * self.c * tau**2
        return self.phi0 + ph

    def _marker(self, t):
        ph = self._phi(t)
        return self.hub + self.rho * np.column_stack([np.cos(ph), np.sin(ph)])

    def dynamic(self, t):
        ph = float(self._phi(t)[0])
        z = self.z(t)[0]
        prims = [_disc("body", [*self.hub, self.d], 0.012, DARK)]
        for k in range(3):
            a = ph + k * 2 * np.pi / 3
            tip = self.hub + self.rho * np.array([np.cos(a), np.sin(a)])
            prims.append(_segment("body", self.hub, tip, 0.012, DARK, self.d))
            prims.append(_disc("body", [*tip, self.d], self.arm_r, (90, 90, 90)))
            if k == 0:
                prims.append(_disc("marker", [*tip, z + 0.001], 0.006, RED, shaded=False))
        prims.append(_disc("hub", [*self.hub, z + 0.001], 0.007, (200, 200, 200), shaded=False))
        return prims

    def truth(self, t):
        t = np.atleast_1d(t)
        z = self.z(t)
        hub = np.tile([*self.hub, self.d], (len(t), 1))
        return {
            "tracks": {"marker": np.column_stack([self._marker(t), z]), "hub": hub},
            "series": {"phi": self._phi(t)},
        }

    def events(self):
        out = super().events()
        out["stop_t_s"] = self.t_release + float(self.t_stop_rel)
        return out

    def params(self):
        return {
            "omega0": self.omega0, "coulomb": self.c, "viscous": self.b, "phi0": self.phi0,
            "rho_m": self.rho, "hub": list(self.hub),
        }  # fmt: skip


SYSTEMS: dict[str, type[System]] = {
    "free_fall": FreeFall,
    "pendulum": Pendulum,
    "ramp": Ramp,
    "bounce": Bounce,
    "spin_down": SpinDown,
}


def make_system(name: str, plane_offset: float | None = None, **params) -> System:
    cls = SYSTEMS[name]
    if plane_offset is not None:
        params["plane_offset"] = plane_offset
    return cls(**params)
