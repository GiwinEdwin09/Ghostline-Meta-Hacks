"""Local polynomial fits: noise estimate, derivatives and their uncertainty.

A Savitzky-Golay filter assumes evenly spaced samples. Frames get dropped and phone timestamps
jitter, so the cubic is fitted in real time on each point's own neighbours instead (the same
smoother, without the uniform-grid shortcut).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WINDOW = 15  # frames
ORDER = 3
MIN_POINTS = 5


@dataclass
class LocalFit:
    value: np.ndarray  # (n, k) fitted value at each point
    slope: np.ndarray  # (n, k) fitted derivative
    resid: np.ndarray  # (n, k) y - value, divided by sqrt(1 - leverage) so its std is sigma
    slope_factor: np.ndarray  # (n,) std of the slope in units of sigma
    valid: np.ndarray  # (n,) bool, False for points in segments that are too short


def robust_std(x: np.ndarray) -> float:
    """1.4826 x the median absolute deviation, ignoring NaNs."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _fit_window(t: np.ndarray, y: np.ndarray, idx: np.ndarray, order: int):
    """Batched polynomial fits; idx is (n, w), each fitted around its own point's time."""
    tau = t[idx] - t[:, None]  # (n, w), zero at the point each window belongs to
    span = np.maximum(np.abs(tau).max(axis=1), 1e-9)
    tn = tau / span[:, None]
    vand = tn[:, :, None] ** np.arange(order + 1)  # (n, w, p)
    a = np.einsum("nwp,nwq->npq", vand, vand)
    b = np.einsum("nwp,nwk->npk", vand, y[idx])
    ainv = np.linalg.inv(a)
    coef = ainv @ b  # (n, p, k)
    value = coef[:, 0, :]
    slope = coef[:, 1, :] / span[:, None]
    leverage = ainv[:, 0, 0]
    resid = (y - value) / np.sqrt(np.maximum(1 - leverage, 1e-6))[:, None]
    slope_factor = np.sqrt(ainv[:, 1, 1]) / span
    return value, slope, resid, slope_factor


def local_fit(
    t: np.ndarray,
    y: np.ndarray,
    segment: np.ndarray | None = None,
    window: int = WINDOW,
    order: int = ORDER,
) -> LocalFit:
    """Cubic fit in a window of `window` neighbours (within the same segment) around each point.

    y has shape (n,) or (n, k). Near segment edges the window is one-sided, so the fit is
    evaluated at the point itself, not at the window centre. Segments with fewer than 5 points
    are left invalid.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]
    n, k = y.shape
    segment = np.zeros(n, int) if segment is None else np.asarray(segment)
    value = np.full((n, k), np.nan)
    slope = np.full((n, k), np.nan)
    resid = np.full((n, k), np.nan)
    factor = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    for seg in np.unique(segment):
        members = np.nonzero(segment == seg)[0]
        m = len(members)
        if m < MIN_POINTS:
            continue
        w = min(window, m)
        o = min(order, w - 2)
        start = np.clip(np.arange(m) - w // 2, 0, m - w)
        local = start[:, None] + np.arange(w)  # indices into the segment
        ts, ys = t[members], y[members]
        v, s, r, f = _fit_window(ts, ys, local, o)
        value[members], slope[members], resid[members], factor[members] = v, s, r, f
        valid[members] = True
    if squeeze:
        return LocalFit(value[:, 0], slope[:, 0], resid[:, 0], factor, valid)
    return LocalFit(value, slope, resid, factor, valid)


def noise_sigma(fit: LocalFit) -> np.ndarray:
    """Robust std of the (leverage-corrected) residuals, per column."""
    r = fit.resid[fit.valid]
    if r.ndim == 1:
        r = r[:, None]
    return np.array([robust_std(r[:, j]) for j in range(r.shape[1])])
