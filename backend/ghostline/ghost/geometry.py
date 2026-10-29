"""From a model's observables to a position in the world frame (build plan Part 16).

A model predicts an angle, a distance along a ramp or a height. The ghost needs to know where on
the sheet that puts the object, in metres, so the error can be measured in millimetres and drawn
through H^-1. The geometry (pivot, hub, ramp axis) comes from the trial being drawn, the same
numbers the trajectory stage turned its pixels into angles with.
"""

from __future__ import annotations

import math

import numpy as np

from .model import GhostError
from .series import TrialSeries


class Geometry:
    """Maps predicted observables (n_obs, n) to world positions (n, 2)."""

    kind = "direct"
    anchor: np.ndarray | None = None  # the pivot, for the ghost string

    def position(self, obs: np.ndarray, t: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def angle(self, obs: np.ndarray) -> np.ndarray | None:
        """Rotation of the object itself, in radians counter-clockwise. Spinners only."""
        return None


class PendulumGeometry(Geometry):
    kind = "pendulum"

    def __init__(self, pivot: np.ndarray, length: float):
        self.anchor = np.asarray(pivot, float)
        self.length = float(length)

    def position(self, obs, t):
        theta = obs[0]
        return self.anchor + self.length * np.column_stack([np.sin(theta), -np.cos(theta)])


class SpinnerGeometry(Geometry):
    kind = "spinner"

    def __init__(self, hub: np.ndarray, radius: float):
        self.hub = np.asarray(hub, float)
        self.radius = float(radius)

    def position(self, obs, t):
        phi = obs[0]
        return self.hub + self.radius * np.column_stack([np.cos(phi), np.sin(phi)])

    def angle(self, obs):
        return np.asarray(obs[0], float)


class RampGeometry(Geometry):
    kind = "ramp"

    def __init__(self, origin: np.ndarray, axis: np.ndarray):
        self.origin = np.asarray(origin, float)
        self.axis = np.asarray(axis, float)

    def position(self, obs, t):
        return self.origin + np.outer(obs[0], self.axis)


class DirectGeometry(Geometry):
    """The model predicts a coordinate (y, or x and y). A missing x follows a straight line
    through the launch window, which is what no horizontal force predicts."""

    def __init__(self, names: tuple[str, ...], x_line: tuple[float, float] | None):
        self.names = names
        self.x_line = x_line  # (x at t = 0, slope), when x is not predicted

    def position(self, obs, t):
        by_name = dict(zip(self.names, obs, strict=True))
        y = by_name["y"]
        if "x" in by_name:
            x = by_name["x"]
        else:
            x0, slope = self.x_line or (0.0, 0.0)
            x = x0 + slope * np.asarray(t, float)
        return np.column_stack([x, y])


def make_geometry(observables: tuple[str, ...], series: TrialSeries) -> Geometry:
    """The geometry for a model with these observables, measured on `series` (trial 2)."""
    s = series.summary
    if "theta" in observables:
        info = s.get("pendulum")
        if not info:
            raise GhostError("The pendulum's pivot and length were not measured in this clip.")
        return PendulumGeometry(np.asarray(info["pivot_xy_m"]), info["length_m"])
    if "phi" in observables:
        info = s.get("spinner")
        if not info:
            raise GhostError("The spinner's hub was not measured in this clip.")
        return SpinnerGeometry(np.asarray(info["hub_xy_m"]), info["marker_radius_m"])
    if "s" in observables:
        info = s.get("ramp")
        if not info or "s" not in series.df:
            raise GhostError("The ramp's direction was not measured in this clip.")
        a = math.radians(info["path_angle_deg"])
        axis = np.array([math.cos(a), math.sin(a)])
        rows = series.motion
        origin = np.median(rows[["x", "y"]].to_numpy(float) - np.outer(rows["s"], axis), axis=0)
        return RampGeometry(origin, axis)
    if "y" in observables:
        x_line = None
        if "x" not in observables:
            rows = series.launch_rows()
            t = rows["t"].to_numpy(float)
            slope, intercept = (
                np.polyfit(t, rows["x"].to_numpy(float), 1)
                if len(t) >= 3
                else (
                    0.0,
                    float(rows["x"].median()),
                )
            )
            x_line = (float(intercept), float(slope))
        return DirectGeometry(tuple(observables), x_line)
    raise GhostError(f"Don't know how to place a model that predicts {list(observables)}.")
