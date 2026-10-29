"""Model interface shared by library models and Wonder-written models (build plan Part 10).

Conventions every model follows:

- `p` is a plain dict of floats holding the fitted parameters and the job constants.
- Time is relative to the first sample of the motion window (`tau`), so initial conditions are the
  state at the start of the window.
- `predict` returns an array of shape (len(observables), len(tau)).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, model_validator
from scipy.integrate import solve_ivp

ParamKind = Literal["physical", "initial_condition", "nuisance"]


class ParamSpec(BaseModel):
    init: float
    lower: float
    upper: float
    unit: str
    kind: ParamKind = "physical"

    @model_validator(mode="after")
    def _ordered(self) -> ParamSpec:
        values = (self.init, self.lower, self.upper)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("init, lower and upper must be finite")
        if not self.lower <= self.init <= self.upper:
            raise ValueError(f"need lower <= init <= upper, got {values}")
        return self


_ODE_TOL = {"rtol": 1e-9, "atol": 1e-11}


@contextlib.contextmanager
def ode_tolerance(rtol: float, atol: float) -> Iterator[None]:
    """Temporarily loosen the ODE tolerances, for bootstrap refits."""
    previous = dict(_ODE_TOL)
    _ODE_TOL.update(rtol=rtol, atol=atol)
    try:
        yield
    finally:
        _ODE_TOL.update(previous)


def integrate(
    fun: Callable[[float, np.ndarray], Any], s0: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Integrate from tau[0] and sample at tau. DOP853, falling back to LSODA. NaN if both fail."""
    tau = np.asarray(tau, dtype=float)
    s0 = np.asarray(s0, dtype=float)
    if tau.size < 2 or tau[-1] <= tau[0]:
        return np.tile(s0[:, None], (1, tau.size))
    for method in ("DOP853", "LSODA"):
        try:
            sol = solve_ivp(fun, (tau[0], tau[-1]), s0, method=method, t_eval=tau, **_ODE_TOL)
        except (ValueError, FloatingPointError, OverflowError):
            continue
        if sol.success and sol.y.shape[1] == tau.size and np.all(np.isfinite(sol.y)):
            return sol.y
    return np.full((s0.size, tau.size), np.nan)


class ModelDef:
    """One competing explanation of a motion. Subclasses fill in the class attributes."""

    id: str = ""
    name: str = ""
    system: str = ""
    mechanism: str = ""
    equations_latex: str = ""
    signature: str = ""
    observables: tuple[str, ...] = ()
    observable_units: dict[str, str] = {}
    state: tuple[str, ...] = ()
    params: dict[str, ParamSpec] = {}
    constants: tuple[str, ...] = ()
    length_constants: tuple[str, ...] = ()
    object_kinds: tuple[str, ...] = ()
    tie_rank: int = 0
    by_flight: bool = False

    def initial_state(self, p: dict[str, float]) -> np.ndarray:
        raise NotImplementedError

    def rhs(self, t: float, s: np.ndarray, p: dict[str, float]) -> Any:
        raise NotImplementedError

    def closed_form(self, tau: np.ndarray, p: dict[str, float]) -> np.ndarray | None:
        return None

    def observe(self, s: np.ndarray) -> np.ndarray:
        return np.asarray(s, dtype=float)[: len(self.observables)]

    def simulate(self, tau: np.ndarray, p: dict[str, float]) -> np.ndarray:
        s0 = np.asarray(self.initial_state(p), dtype=float)
        states = integrate(lambda t, s: self.rhs(t, s, p), s0, tau)
        return np.asarray(self.observe(states), dtype=float)

    def predict(self, tau: np.ndarray, p: dict[str, float]) -> np.ndarray:
        out = self.closed_form(tau, p)
        if out is None:
            return self.simulate(tau, p)
        return np.asarray(out, dtype=float)

    def prepare(
        self, tau: np.ndarray, obs: dict[str, np.ndarray], constants: dict[str, float]
    ) -> dict[str, float]:
        """Return the constants with any data-derived defaults filled in."""
        return dict(constants)

    def fit_setup(
        self,
        tau: np.ndarray,
        obs: dict[str, np.ndarray],
        constants: dict[str, float],
        segment: np.ndarray | None = None,
    ) -> dict[str, ParamSpec]:
        """Parameter specs for this data: heuristic initial guesses and data-aware bounds."""
        return {name: spec.model_copy() for name, spec in self.params.items()}

    def derived(self, p: dict[str, float]) -> dict[str, float]:
        """Quantities computed from the parameters, reported with their own intervals."""
        return {}

    def derived_units(self) -> dict[str, str]:
        return {}

    def missing_constants(self, constants: dict[str, float]) -> list[str]:
        return [name for name in self.constants if name not in constants]

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "system": self.system,
            "mechanism": self.mechanism,
            "equations_latex": self.equations_latex,
            "signature": self.signature,
            "observables": list(self.observables),
            "state": list(self.state),
            "params": {k: v.model_dump() for k, v in self.params.items()},
            "constants": list(self.constants),
            "object_kinds": list(self.object_kinds),
        }


def clip_spec(
    init: float, lower: float, upper: float, unit: str, kind: ParamKind = "physical"
) -> ParamSpec:
    """A ParamSpec whose init is clipped into [lower, upper]."""
    init = float(np.clip(init, lower, upper))
    return ParamSpec(init=init, lower=float(lower), upper=float(upper), unit=unit, kind=kind)


def poly_start(tau: np.ndarray, y: np.ndarray, degree: int, head: int | None = None):
    """Polynomial coefficients c0..cdegree of y(tau); `head` limits the fit to the first points."""
    n = tau.size if head is None else min(tau.size, max(head, degree + 2))
    deg = min(degree, max(n - 1, 0))
    coef = np.polynomial.polynomial.polyfit(tau[:n], y[:n], deg)
    return np.concatenate([coef, np.zeros(degree + 1 - coef.size)])
