"""Residual diagnostics (build plan Part 11)."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import legendre

from .schema import Diagnostics

SYSTEMATIC_R2 = 0.30
MIN_POINTS = 12


def lag1_autocorrelation(r: np.ndarray) -> float | None:
    if r.size < 3 or np.allclose(r, 0):
        return None
    r = r - r.mean()
    denom = float(np.dot(r, r))
    return float(np.dot(r[:-1], r[1:]) / denom) if denom > 0 else None


def durbin_watson(r: np.ndarray) -> float | None:
    denom = float(np.dot(r, r))
    return float(np.sum(np.diff(r) ** 2) / denom) if r.size > 2 and denom > 0 else None


def runs_test_p(r: np.ndarray) -> float | None:
    """Wald-Wolfowitz runs test on the signs of the residuals; two-sided normal p-value."""
    signs = r[r != 0] > 0
    n_pos, n_neg = int(signs.sum()), int((~signs).sum())
    if n_pos < 2 or n_neg < 2:
        return None
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    n = n_pos + n_neg
    mean = 2 * n_pos * n_neg / n + 1
    var = (mean - 1) * (mean - 2) / (n - 1)
    if var <= 0:
        return None
    z = (runs - mean) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2)))


def structure_r2(tau: np.ndarray, r: np.ndarray) -> float:
    """Largest share of the residual variance explained by a cubic polynomial or one sinusoid."""
    if r.size < MIN_POINTS:
        return 0.0
    r = r - r.mean()
    total = float(np.dot(r, r))
    if total <= 1e-300:
        return 0.0
    span = float(tau[-1] - tau[0])
    if span <= 0:
        return 0.0
    u = 2 * (tau - tau[0]) / span - 1
    poly = legendre.legvander(u, 3)[:, 1:]
    best = 0.0
    coef, *_ = np.linalg.lstsq(poly, r, rcond=None)
    best = max(best, 1 - float(np.sum((r - poly @ coef) ** 2)) / total)
    for freq in np.geomspace(1.0 / span, max(r.size / 4, 2.0) / span, 60):
        arg = 2 * math.pi * freq * (tau - tau[0])
        design = np.column_stack([np.sin(arg), np.cos(arg)])
        coef, *_ = np.linalg.lstsq(design, r, rcond=None)
        best = max(best, 1 - float(np.sum((r - design @ coef) ** 2)) / total)
    return best


def diagnose(
    tau: np.ndarray,
    normalized: np.ndarray,
    raw_rmse: float,
    n_params: int,
    condition: float | None = None,
) -> Diagnostics:
    """Diagnostics from normalized residuals of shape (n_observables, n_points)."""
    n = normalized.shape[1]
    total = normalized.size
    chi2 = float(np.sum(normalized**2))
    r_first = normalized[0]
    r2 = max((structure_r2(tau, row) for row in normalized), default=0.0)
    return Diagnostics(
        n=n,
        n_params=n_params,
        rmse=raw_rmse,
        chi2=chi2,
        chi2_red=chi2 / max(total - n_params, 1),
        lag1=lag1_autocorrelation(r_first),
        durbin_watson=durbin_watson(r_first),
        runs_p=runs_test_p(r_first),
        systematic=r2 > SYSTEMATIC_R2,
        systematic_r2=float(r2),
        condition=condition,
    )
