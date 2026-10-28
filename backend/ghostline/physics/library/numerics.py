"""Cancellation-free helpers for the closed-form solutions."""

from __future__ import annotations

import numpy as np

_SMALL = 1e-2


def phi1(x):
    """(1 - exp(-x)) / x, accurate near zero."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < _SMALL
    xs = np.where(small, 1.0, x)
    series = 1 - x / 2 + x**2 / 6 - x**3 / 24 + x**4 / 120 - x**5 / 720
    return np.where(small, series, -np.expm1(-xs) / xs)


def phi2(x):
    """(x - 1 + exp(-x)) / x^2, accurate near zero."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < _SMALL
    xs = np.where(small, 1.0, x)
    series = 0.5 - x / 6 + x**2 / 24 - x**3 / 120 + x**4 / 720 - x**5 / 5040
    return np.where(small, series, (xs + np.expm1(-xs)) / xs**2)


def log_cosh(x):
    """ln(cosh(x)) without overflow."""
    x = np.asarray(x, dtype=float)
    return np.logaddexp(x, -x) - np.log(2.0)


def linear_drag_axis(x0: float, v0: float, a: float, b: float, tau: np.ndarray) -> np.ndarray:
    """Position under constant acceleration a and linear drag b: v' = a - b v."""
    bt = b * tau
    return x0 + v0 * tau * phi1(bt) + a * tau**2 * phi2(bt)
