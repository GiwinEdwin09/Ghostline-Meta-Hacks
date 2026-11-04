"""Ground truth for the benchmark (build plan Part 25): how each true value is obtained, and how a
measurement is compared with it.

    g (free fall, pendulum)   local gravity from latitude and altitude
    mu_k (ramp, block)        tan(theta) of the tilt at which the block slides at constant speed,
                              mean +- SD over repeats
    beta (ramp, ball)         theory: 2/5 for a solid sphere, so a = (5/7) g sin(alpha)
    e (bounce)                audio clicks (bench/audio_bounce.py)

Everything here is plain arithmetic on numbers; the manifest loader at the bottom ties it to
bench/manifest.csv.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, model_validator
from scipy.stats import norm

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BENCH_DIR / "manifest.csv"

# --- local gravity -------------------------------------------------------------------------
G_EQUATOR = 9.780327  # m/s^2, the International Gravity Formula 1980
FREE_AIR_GRADIENT = 3.086e-6  # m/s^2 lost per metre of altitude
# What the formula can't know: local anomalies (a few tens of mGal, 1 mGal = 1e-5 m/s^2, and up
# to about 200 mGal over mountain ranges). 0.002 m/s^2 covers the realistic cases.
GRAVITY_ANOMALY_SIGMA = 0.002


def local_gravity(latitude_deg: float, altitude_m: float = 0.0) -> float:
    """g = 9.780327 (1 + 0.0053024 sin^2(phi) - 0.0000058 sin^2(2 phi)) - 3.086e-6 h, in m/s^2.

    The altitude term is the free-air correction: it ignores the rock under the observer, which
    adds back roughly a third of it. For a benchmark filmed under 500 m, that is under 5e-4."""
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError(f"latitude must be between -90 and 90 degrees, got {latitude_deg}")
    phi = math.radians(latitude_deg)
    sea_level = G_EQUATOR * (
        1 + 0.0053024 * math.sin(phi) ** 2 - 0.0000058 * math.sin(2 * phi) ** 2
    )
    return sea_level - FREE_AIR_GRADIENT * altitude_m


def local_gravity_sigma(altitude_sigma_m: float = 10.0) -> float:
    """The 1-sigma uncertainty of local_gravity: anomalies plus an uncertain altitude."""
    return math.hypot(GRAVITY_ANOMALY_SIGMA, FREE_AIR_GRADIENT * altitude_sigma_m)


# --- a truth value ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Truth:
    """A true value with its 1-sigma uncertainty and how it was obtained."""

    value: float
    sigma: float
    method: str


def gravity_truth(
    latitude_deg: float, altitude_m: float = 0.0, altitude_sigma_m: float = 10.0
) -> Truth:
    return Truth(
        value=local_gravity(latitude_deg, altitude_m),
        sigma=local_gravity_sigma(altitude_sigma_m),
        # No coordinates here: this string is published with the results, and the repo is public.
        method="local gravity from latitude and altitude (IGF 1980, free-air)",
    )


# --- friction ------------------------------------------------------------------------------
def mu_k_from_tilts(angles_deg: Sequence[float]) -> Truth:
    """mu_k = tan(theta), where theta is the board tilt at which the block slides at constant
    speed. Each repeat gives one value; the truth is their mean and the sigma their sample SD
    (not the standard error): it is the spread of the method, which is what a single clip is
    compared against."""
    if len(angles_deg) < 2:
        raise ValueError("need at least two tilt repeats to estimate a spread")
    for angle in angles_deg:
        if not 0.0 < angle < 90.0:
            raise ValueError(f"a tilt must be between 0 and 90 degrees, got {angle}")
    values = [math.tan(math.radians(a)) for a in angles_deg]
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return Truth(
        value=mean,
        sigma=math.sqrt(var),
        method=f"tan(tilt) over {len(values)} repeats, mean +- SD",
    )


# --- rolling ---------------------------------------------------------------------------------
# beta = I / (m r^2) for a body rolling without slipping: a = g sin(alpha) / (1 + beta).
ROLLING_BETA = {"solid_sphere": 2 / 5, "hollow_sphere": 2 / 3, "solid_cylinder": 1 / 2}


def rolling_acceleration(g: float, alpha_deg: float, beta: float = ROLLING_BETA["solid_sphere"]):
    """a = g sin(alpha) / (1 + beta). For a solid sphere this is (5/7) g sin(alpha)."""
    return g * math.sin(math.radians(alpha_deg)) / (1 + beta)


def beta_from_acceleration(accel: float, g: float, alpha_deg: float) -> float:
    """The inverse: the beta that gives this acceleration on this slope."""
    if accel <= 0:
        raise ValueError("the acceleration must be positive")
    return g * math.sin(math.radians(alpha_deg)) / accel - 1


def rolling_truth(shape: str = "solid_sphere") -> Truth:
    """Theory has no measurement error, so sigma is 0. How well the real ball matches the shape
    (a hollow ball, slip, rolling resistance) shows up as a bias in the measurement instead."""
    return Truth(
        value=ROLLING_BETA[shape], sigma=0.0, method=f"theory: {shape} rolls with I/(mr^2)"
    )


# --- pendulum sub-checks ---------------------------------------------------------------------
def pendulum_period_correction(amplitude_deg: float) -> float:
    """T / T0 for a finite swing: 1 + theta0^2/16 + 11 theta0^4/3072 (theta0 in radians)."""
    theta = math.radians(amplitude_deg)
    return 1 + theta**2 / 16 + 11 * theta**4 / 3072


def g_from_pendulum(
    length_m: float,
    total_time_s: float,
    n_periods: int = 20,
    *,
    sigma_length_m: float = 0.001,
    sigma_time_s: float = 0.2,
    amplitude_deg: float | None = None,
) -> Truth:
    """The by-hand cross-check: g = 4 pi^2 L / T0^2, with L from a tape (+-1 mm) and T from a
    stopwatch over 20 periods (+-0.2 s of reaction time on the total). Pass the amplitude to
    remove the finite-swing lengthening of the period."""
    if length_m <= 0 or total_time_s <= 0 or n_periods <= 0:
        raise ValueError("length, time and the number of periods must be positive")
    period = total_time_s / n_periods
    if amplitude_deg is not None:
        period /= pendulum_period_correction(amplitude_deg)
    g = 4 * math.pi**2 * length_m / period**2
    rel = math.hypot(sigma_length_m / length_m, 2 * sigma_time_s / total_time_s)
    return Truth(value=g, sigma=g * rel, method=f"tape and stopwatch over {n_periods} periods")


# --- comparing a measurement with the truth ------------------------------------------------
def combine_sigma(*sigmas: float | None) -> float:
    """Quadrature sum of independent 1-sigma uncertainties. None means 'none given' (0)."""
    total = 0.0
    for sigma in sigmas:
        if sigma is None:
            continue
        if sigma < 0 or math.isnan(sigma):
            raise ValueError(f"a sigma can't be {sigma}")
        total += sigma**2
    return math.sqrt(total)


def sigma_from_interval(ci_low: float, ci_high: float, level: float = 0.95) -> float:
    """The 1-sigma that gives a normal interval of this half-width at this level."""
    if ci_high < ci_low:
        raise ValueError("the interval's upper end is below its lower end")
    return (ci_high - ci_low) / 2 / float(norm.ppf((1 + level) / 2))


def z_score(
    measured: float,
    truth: float,
    sigma_measured: float | None,
    sigma_truth: float | None = 0.0,
) -> float | None:
    """(measured - truth) / sqrt(sigma_measured^2 + sigma_truth^2). None when both sigmas are 0,
    where z is meaningless."""
    combined = combine_sigma(sigma_measured, sigma_truth)
    if combined == 0:
        return None
    return (measured - truth) / combined


def interval_contains(ci_low: float, ci_high: float, truth: float) -> bool:
    """The benchmark's coverage test: the 95% interval contains the true value."""
    return ci_low <= truth <= ci_high


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for a proportion such as coverage, which is honest at small n."""
    if trials <= 0:
        return (0.0, 1.0)
    z = float(norm.ppf((1 + level) / 2))
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --- the manifest ------------------------------------------------------------------------------
_TEXT_FIELDS = {
    "kind",
    "object",
    "setup_notes",
    "unit",
    "method",
    "tilt_deg",
    "audio_files",
    "expected_model",
    "failure_expected",
    "file",
}


class ManifestRow(BaseModel):
    """One planned clip. A blank truth means 'still to film or measure'."""

    clip_id: str
    kind: str = "scored"  # scored | story | failure
    system: str
    object: str = ""
    setup_notes: str = ""
    quantity: str | None = None
    unit: str = ""
    truth: float | None = None
    truth_sigma: float | None = None
    method: str = ""  # local_gravity | tilt | theory_rolling | audio_clicks | expect_low_confidence
    latitude: float | None = None
    altitude_m: float | None = None
    tilt_deg: str = ""  # semicolon-separated repeats, for method=tilt
    audio_files: str = ""  # semicolon-separated WAVs, for method=audio_clicks
    pendulum_length_cm: float | None = None
    plane_offset_cm: float | None = None
    capture_fps: float | None = None  # blank or under 120: normal video; otherwise slow-mo
    expected_model: str = ""  # library ids, '|' between acceptable ones
    failure_expected: str = ""  # the quality-flag code a bad clip should be marked low for
    file: str = ""  # video path, relative to bench/clips/
    trim_start_s: float | None = None  # analysis window, set through the trim API
    trim_end_s: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _blanks(cls, data):
        """CSV cells are strings: a blank text cell is "" and a blank number is None."""
        if not isinstance(data, dict):
            return data
        out = {}
        for key, value in data.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                out[key] = ("scored" if key == "kind" else "") if key in _TEXT_FIELDS else None
            else:
                out[key] = value.strip() if isinstance(value, str) else value
        return out

    @property
    def expected_models(self) -> list[str]:
        return [m for m in self.expected_model.split("|") if m]

    @property
    def tilts(self) -> list[float]:
        return [float(v) for v in self.tilt_deg.split(";") if v.strip()]

    @property
    def audio_paths(self) -> list[str]:
        return [v.strip() for v in self.audio_files.split(";") if v.strip()]

    @property
    def declared_capture(self) -> dict:
        if self.capture_fps and self.capture_fps >= 120:
            return {"mode": "slow_mo", "fps": self.capture_fps}
        return {"mode": "normal"}


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> list[ManifestRow]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [ManifestRow.model_validate(r) for r in csv.DictReader(handle)]
    ids = [r.clip_id for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"clip ids must be unique, repeated: {dupes}")
    return rows


def resolve_truth(
    row: ManifestRow,
    *,
    latitude: float | None = None,
    altitude_m: float | None = None,
    clips_dir: Path | str | None = None,
) -> Truth | None:
    """The row's true value and sigma, or None while it still needs filming or measuring.

    A truth written into the manifest wins. Otherwise it is computed from the method's inputs:
    latitude and altitude (the row's own, else the defaults passed in, which the runner takes from
    BENCH_LATITUDE and BENCH_ALTITUDE_M), tilt repeats, or the audio recordings."""
    if row.truth is not None:
        return Truth(row.truth, row.truth_sigma or 0.0, row.method or "manifest")
    if row.method == "local_gravity":
        lat = row.latitude if row.latitude is not None else latitude
        if lat is None:
            return None
        alt = row.altitude_m if row.altitude_m is not None else (altitude_m or 0.0)
        return gravity_truth(lat, alt)
    if row.method == "tilt":
        return mu_k_from_tilts(row.tilts) if len(row.tilts) >= 2 else None
    if row.method == "theory_rolling":
        return rolling_truth()
    if row.method == "audio_clicks" and row.audio_paths:
        from bench.audio_bounce import truth_from_recordings

        base = Path(clips_dir) if clips_dir else BENCH_DIR / "clips"
        return truth_from_recordings([base / p for p in row.audio_paths])
    return None
