"""Events (release, stop, impacts) and quality checks (out-of-plane motion, re-timed clips)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from .noise import LocalFit

NOISE_MEAN_FACTOR = 1.25  # mean of a 2D Rayleigh speed with unit-sigma components
SPEED_K = 3.0  # "3 sigma above noise"
RELEASE_RUN = 5  # frames in a row
STOP_DWELL_S = 0.5
STOP_TAIL_S = 0.2  # a slow run that reaches the end of the clip counts after this long
IMPACT_DROP_FRAMES = 2
IMPACT_MIN_SPACING = 4  # frames between impacts
IMPACT_SLOPE_K = 3.0
G0 = 9.81
RETIME_FACTORS = (2, 4, 8)
RETIME_TOL = 0.2


@dataclass
class Impact:
    index: int  # sample index of the lowest point
    frame: int
    t: float  # refined impact time


@dataclass
class Events:
    release_index: int | None = None
    release_t: float | None = None
    stop_index: int | None = None
    stop_t: float | None = None
    impacts: list[Impact] = field(default_factory=list)
    last_moving_t: float | None = None
    speed: np.ndarray | None = None
    speed_threshold: np.ndarray | None = None

    @property
    def window_end_t(self) -> float | None:
        """End of the motion window: the stop, else the last time the speed was above noise."""
        return self.stop_t if self.stop_t is not None else self.last_moving_t


def speed_and_threshold(fit: LocalFit, sigma_pos: float) -> tuple[np.ndarray, np.ndarray]:
    """Speed from the local fits and the level above which it counts as motion.

    The threshold is the mean noise speed plus 3 sigma, with sigma from the fit's slope variance.
    """
    speed = np.hypot(fit.slope[:, 0], fit.slope[:, 1])
    sigma_v = sigma_pos * fit.slope_factor
    return speed, (NOISE_MEAN_FACTOR + SPEED_K) * sigma_v


def _first_run(mask: np.ndarray, length: int) -> int | None:
    count = 0
    for i, v in enumerate(mask):
        count = count + 1 if v else 0
        if count >= length:
            return i - length + 1
    return None


def detect_release_stop(t: np.ndarray, xy: np.ndarray, fit: LocalFit, sigma_pos: float) -> Events:
    """Release: speed above noise for five frames in a row, then backed up to where the object
    first leaves its resting position. Stop: speed under the threshold for more than 0.5 s (or
    for 0.2 s up to the end of the clip), backed up to where it first settles."""
    ev = Events()
    speed, thr = speed_and_threshold(fit, sigma_pos)
    ev.speed, ev.speed_threshold = speed, thr
    n = len(t)
    moving = (speed > thr) & fit.valid
    onset = _first_run(moving, RELEASE_RUN)
    if onset is None:
        return ev
    release = onset
    rest = np.arange(max(onset - 6, 0))  # the fit smears the onset by up to half its window
    if len(rest) >= 3:
        rest_pos = np.median(xy[rest], axis=0)
        away = np.linalg.norm(xy - rest_pos, axis=1) > 4 * sigma_pos
        lo = max(onset - 15, 0)
        first = _first_run(away[lo:], 3)
        if first is not None:
            release = lo + first
    ev.release_index, ev.release_t = release, float(t[release])
    ev.last_moving_t = float(t[np.nonzero(moving)[0][-1]])
    slow = (speed < thr) | ~fit.valid
    i = min(release + RELEASE_RUN, n - 1)
    while i < n:
        if not slow[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and slow[j + 1]:
            j += 1
        dwell = t[j] - t[i]
        if dwell >= STOP_DWELL_S or (j == n - 1 and dwell >= STOP_TAIL_S):
            tail = np.arange(min(i + 7, j), j + 1)  # skip the smeared start of the slow run
            rest_pos = np.median(xy[tail], axis=0)
            near = np.linalg.norm(xy - rest_pos, axis=1) < 4 * sigma_pos
            stop = i
            for k in range(max(i - 15, release + 1), i + 1):
                if near[k : k + 3].all():
                    stop = k
                    break
            ev.stop_index, ev.stop_t = stop, float(t[stop])
            break
        i = j + 1
    return ev


def _line(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(t, y, 1)
    return float(slope), float(intercept)


def detect_impacts(
    t: np.ndarray, y: np.ndarray, frames: np.ndarray, sigma_pos: float
) -> list[Impact]:
    """Minima of y where the vertical velocity flips from down to up."""
    if len(y) < 12:
        return []
    prominence = max(5 * sigma_pos, 0.003 * float(np.ptp(y)))
    peaks, _ = find_peaks(-y, prominence=prominence, distance=IMPACT_MIN_SPACING)
    # a real impact reverses the velocity abruptly, so both sides must be clearly above the noise of
    # a 4-point slope; a smooth minimum (a pendulum's lowest point) is not
    slope_noise = sigma_pos * np.sqrt(12 / (4 * 15)) / float(np.median(np.diff(t)))
    min_slope = IMPACT_SLOPE_K * slope_noise
    out = []
    for j in peaks:
        if j < 4 or j > len(y) - 5:
            continue
        before, after = slice(j - 4, j), slice(j + 1, j + 5)
        sb, ib = _line(t[before], y[before])
        sa, ia = _line(t[after], y[after])
        if not (sb < -min_slope and sa > min_slope):
            continue
        t_hit = (ib - ia) / (sa - sb) if sa != sb else t[j]
        if not t[j - 1] <= t_hit <= t[j + 1]:
            t_hit = t[j]
        out.append(Impact(int(j), int(frames[j]), float(t_hit)))
    return out


def impact_drop_mask(frames: np.ndarray, impacts: list[Impact]) -> np.ndarray:
    """Samples within two frames of an impact (the frame of the impact included)."""
    drop = np.zeros(len(frames), bool)
    for imp in impacts:
        drop |= np.abs(frames - imp.frame) <= IMPACT_DROP_FRAMES
    return drop


def flight_segments(t: np.ndarray, impacts: list[Impact]) -> np.ndarray:
    """segment_id: 0 before the first impact, 1 between the first and second, and so on."""
    return (
        np.searchsorted(np.array([i.t for i in impacts]), t) if impacts else np.zeros(len(t), int)
    )


def retime_check(t: np.ndarray, y: np.ndarray, segment: np.ndarray, window: np.ndarray) -> dict:
    """Fit g to each free flight; flag if one matches 9.81 / k^2 for k = 2, 4 or 8.

    A clip that was slowed by k looks like gravity weaker by k^2. `window` marks samples inside
    the motion window. Flights shorter than 8 points or 0.15 s are skipped.
    """
    flights = []
    for seg in np.unique(segment):
        m = (segment == seg) & window
        if m.sum() < 8 or np.ptp(t[m]) < 0.15:
            continue
        c2 = np.polyfit(t[m] - t[m].mean(), y[m], 2)[0]
        g = float(-2 * c2)
        flights.append({"segment_id": int(seg), "g_fit": g, "n": int(m.sum())})
    flagged = None
    for f in flights:
        for k in RETIME_FACTORS:
            target = G0 / k**2
            if f["g_fit"] > 0 and abs(f["g_fit"] - target) / target < RETIME_TOL:
                if abs(f["g_fit"] - G0) / G0 > RETIME_TOL:
                    flagged = k
    return {"checked": bool(flights), "flights": flights, "suspected_factor": flagged}


def out_of_plane(
    area: np.ndarray, columns: dict[str, np.ndarray], speed: np.ndarray | None
) -> dict:
    """Area changes that follow the position suggest motion toward or away from the camera.

    Mask area scales as 1/Z^2, so dZ/Z is about dA / (2A). Motion blur also changes the mask area,
    roughly linearly with speed, so a linear speed term is removed first. What is left is flagged
    when it varies by more than 5% and correlates with a position or with time (|r| over 0.5).
    Motion toward the camera that rises in step with speed, such as a drop, can't be told from
    blur this way.
    """
    ok = np.isfinite(area)
    if ok.sum() < 8:
        return {"checked": False, "flag": False}
    a = area[ok]
    med = float(np.median(a))

    def spread(v: np.ndarray) -> float:
        return float((np.percentile(v, 95) - np.percentile(v, 5)) / med)

    corrected = a
    r_speed = 0.0
    if speed is not None and np.std(speed[ok]) > 0 and np.std(a) > 0:
        sp = speed[ok]
        coef = np.polyfit(sp, a, 1)
        corrected = a - np.polyval(coef, sp) + med
        r_speed = abs(float(np.corrcoef(a, sp)[0, 1]))
    variation = spread(corrected)
    r_pos = 0.0
    for v in columns.values():
        v = v[ok]
        if np.std(v) > 0 and np.std(corrected) > 0:
            r_pos = max(r_pos, abs(float(np.corrcoef(corrected, v)[0, 1])))
    flag = variation > 0.05 and r_pos > 0.5
    return {
        "checked": True,
        "flag": bool(flag),
        "area_rel_variation": spread(a),
        "blur_corrected_variation": variation,
        "r_position": r_pos,
        "r_speed": r_speed,
        "dz_over_z": variation / 2,
    }
