"""Bounce restitution from audio (build plan Part 25): the ground truth for e.

A ball dropped on a hard floor clicks at every impact. Between impacts n-1 and n it flies for
dt_n, and for a constant restitution e each flight is e times the one before, so

    e = dt_{n+1} / dt_n,   averaged over the bounces and over the drops.

Audio timing is the truth because a microphone samples at 48 kHz and needs no video frame
timing model. The method:

1. band-pass 1 to 12 kHz with a zero-phase filter, which removes mains hum and its harmonics,
   rumble, and handling thuds, and adds no delay;
2. energy envelope: the squared signal, smoothed over about a millisecond;
3. onsets: envelope peaks above a noise floor estimated from the recording itself; each onset is
   put where the envelope first rises through a fraction of its peak, so every click is timed at
   the same point of its own shape and a constant detection delay cancels in dt_n;
4. the dt series, the ratios, and their mean after dropping impossible ratios and outliers.

Record from the moment the ball is released or just after (the click of letting go would be read
as an impact), several drops per recording are fine: drops are split where the silence is long.

    python -m bench.audio_bounce drop1.wav drop2.wav
    python -m bench.audio_bounce /data/experiments/<exp>        # finds every audio.wav under it
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from bench.truth import Truth

DEFAULT_BAND_HZ = (1000.0, 12000.0)
SMOOTH_MS = 1.0
MIN_GAP_S = 0.015  # clicks closer than this are one click
MIN_INTERVAL_S = 0.03  # past this the ball is rattling to a stop and clicks merge
DROP_GAP_S = 2.0  # silence that separates two drops (a first flight lasts about 1 s)
NOISE_K = 10.0  # threshold: noise floor plus this many robust sigmas of the envelope
ONSET_FRACTION = 0.15  # an onset is where the envelope rises through this share of its peak
SEARCH_MS = 8.0  # how far back from a peak the onset can be
RATIO_RANGE = (0.05, 1.1)  # a restitution outside this is a mis-detection, not physics
OUTLIER_MADS = 4.5
SIGMA_FLOOR = 0.005  # air drag and the choice of onset point bias e by about this much


class AudioBounceError(ValueError):
    """The recording doesn't hold enough bounces to measure e."""


# --- reading audio -------------------------------------------------------------------------
def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Mono float samples in [-1, 1] and the sample rate. Stereo is averaged."""
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as handle:
            rate, width, channels = (
                handle.getframerate(),
                handle.getsampwidth(),
                handle.getnchannels(),
            )
            raw = handle.readframes(handle.getnframes())
    except wave.Error:
        # Float WAVs (format 3) are not PCM, which the wave module refuses.
        from scipy.io import wavfile

        rate, data = wavfile.read(str(path))
        samples = np.asarray(data, dtype=np.float64)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return samples, int(rate)
    if width == 3:  # 24-bit little-endian, which numpy has no dtype for
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        samples = ints.astype(np.float64) / 2**23
    elif width in (1, 2, 4):
        dtype = {1: np.uint8, 2: "<i2", 4: "<i4"}[width]
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        samples = (samples - 128) / 128 if width == 1 else samples / 2 ** (8 * width - 1)
    else:
        raise AudioBounceError(f"{path.name}: {8 * width}-bit samples are not supported")
    if channels > 1:
        samples = samples[: len(samples) // channels * channels].reshape(-1, channels).mean(axis=1)
    return samples, int(rate)


def find_audio(path: Path | str) -> list[Path]:
    """A WAV file as given, or every audio.wav under a trial, experiment or backup folder."""
    path = Path(path)
    if path.is_dir():
        found = sorted(path.rglob("audio.wav"))
        if not found:
            raise FileNotFoundError(f"no audio.wav under {path}")
        return found
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


# --- onset detection ---------------------------------------------------------------------------
def bandpass(x: np.ndarray, rate: int, band: tuple[float, float] = DEFAULT_BAND_HZ) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass (the upper edge is kept under Nyquist)."""
    low, high = band[0], min(band[1], 0.45 * rate)
    if not 0 < low < high:
        raise AudioBounceError(f"a {rate} Hz recording can't hold the {band} Hz band")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    return signal.sosfiltfilt(sos, x - np.mean(x))


def energy_envelope(y: np.ndarray, rate: int, smooth_ms: float = SMOOTH_MS) -> np.ndarray:
    window = max(1, round(smooth_ms * rate / 1000))
    return uniform_filter1d(y * y, size=window, mode="nearest")


def detect_onsets(
    x: np.ndarray,
    rate: int,
    *,
    band: tuple[float, float] = DEFAULT_BAND_HZ,
    smooth_ms: float = SMOOTH_MS,
    min_gap_s: float = MIN_GAP_S,
    noise_k: float = NOISE_K,
    onset_fraction: float = ONSET_FRACTION,
    search_ms: float = SEARCH_MS,
) -> np.ndarray:
    """Click onset times, in seconds from the start of x."""
    env = energy_envelope(bandpass(x, rate, band), rate, smooth_ms)
    floor = float(np.median(env))
    spread = 1.4826 * float(np.median(np.abs(env - floor)))
    # A noiseless track has no spread; the relative term then stops numerical ripples counting.
    threshold = max(floor + noise_k * spread, 1e-4 * float(env.max()))
    if env.max() <= threshold:
        return np.empty(0)
    peaks, _ = signal.find_peaks(env, height=threshold, distance=max(1, round(min_gap_s * rate)))
    back = max(1, round(search_ms * rate / 1000))
    quiet = floor + 3 * spread  # an onset level must clear the noise
    onsets = []
    for peak in peaks:
        level = max(onset_fraction * env[peak], quiet)
        lo = max(0, peak - back)
        segment = env[lo : peak + 1]
        below = np.nonzero(segment < level)[0]
        if len(below) == 0:  # already above the level at the edge of the search window
            onsets.append(lo / rate)
            continue
        i = lo + int(below[-1])  # the last sample under the level before the peak
        rise = env[i + 1] - env[i]
        frac = (level - env[i]) / rise if rise > 0 else 0.0
        onsets.append((i + frac) / rate)
    return np.asarray(onsets)


def split_drops(onsets: np.ndarray, gap_s: float = DROP_GAP_S) -> list[np.ndarray]:
    """Group onsets into drops: a silence longer than gap_s starts a new one."""
    if len(onsets) == 0:
        return []
    cuts = np.nonzero(np.diff(onsets) > gap_s)[0] + 1
    return [chunk for chunk in np.split(onsets, cuts) if len(chunk)]


# --- restitution -------------------------------------------------------------------------------
@dataclass
class BounceResult:
    """One drop. `e` is the mean of the accepted ratios dt_{n+1} / dt_n; `e_fit` is a
    cross-check, the slope of log dt_n against n weighted by dt_n."""

    onsets_s: list[float]
    intervals_s: list[float]
    ratios: list[float]
    rejected_ratios: list[float] = field(default_factory=list)
    e: float = math.nan
    e_sd: float | None = None
    e_sem: float | None = None
    e_fit: float | None = None

    @property
    def n_clicks(self) -> int:
        return len(self.onsets_s)


def restitution_from_onsets(onsets: np.ndarray, *, min_interval_s: float = MIN_INTERVAL_S):
    """e for one drop from its impact times."""
    onsets = np.asarray(onsets, dtype=float)
    intervals = np.diff(onsets)
    short = np.nonzero(intervals < min_interval_s)[0]
    if len(short):  # everything from the first merged click on is a rattle, not a flight
        intervals = intervals[: short[0]]
    if len(intervals) < 2:
        raise AudioBounceError(
            f"found {len(onsets)} clicks ({len(intervals)} usable flights); need at least 3 "
            "clicks to compare two flights"
        )
    ratios = intervals[1:] / intervals[:-1]
    keep = (ratios > RATIO_RANGE[0]) & (ratios <= RATIO_RANGE[1])
    if keep.sum() >= 5:
        centre = np.median(ratios[keep])
        mad = max(1.4826 * np.median(np.abs(ratios[keep] - centre)), 0.01)
        keep &= np.abs(ratios - centre) <= OUTLIER_MADS * mad
    if not keep.any():
        raise AudioBounceError(
            f"none of the {len(ratios)} flight ratios were physical (between "
            f"{RATIO_RANGE[0]} and {RATIO_RANGE[1]}); is the release click in the recording?"
        )
    accepted = ratios[keep]
    n = len(accepted)
    sd = float(np.std(accepted, ddof=1)) if n > 1 else None
    fit = None
    if len(intervals) >= 3:
        index = np.arange(len(intervals))
        slope = np.polyfit(index, np.log(intervals), 1, w=intervals)[0]
        fit = float(math.exp(slope))
    return BounceResult(
        onsets_s=[float(t) for t in onsets[: len(intervals) + 1]],
        intervals_s=[float(v) for v in intervals],
        ratios=[float(v) for v in accepted],
        rejected_ratios=[float(v) for v in ratios[~keep]],
        e=float(accepted.mean()),
        e_sd=sd,
        e_sem=None if sd is None else sd / math.sqrt(n),
        e_fit=fit,
    )


def analyze_bounces(
    x: np.ndarray,
    rate: int,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    skip_first: int = 0,
    drop_gap_s: float = DROP_GAP_S,
    **detect: float,
) -> list[BounceResult]:
    """Every drop in a recording. A drop too short to measure is skipped, but if none can be
    measured the error says why."""
    lo = round(start_s * rate)
    hi = len(x) if end_s is None else round(end_s * rate)
    onsets = detect_onsets(x[lo:hi], rate, **detect) + lo / rate
    drops = split_drops(onsets, drop_gap_s)
    results, errors = [], []
    for drop in drops:
        try:
            results.append(restitution_from_onsets(drop[skip_first:]))
        except AudioBounceError as exc:
            errors.append(str(exc))
    if not results:
        detail = errors[0] if errors else "no clicks above the noise floor"
        raise AudioBounceError(f"could not measure e: {detail}")
    return results


def analyze_file(path: Path | str, **kwargs) -> list[BounceResult]:
    x, rate = read_wav(path)
    return analyze_bounces(x, rate, **kwargs)


# --- the truth for a bounce clip -----------------------------------------------------------
def truth_from_recordings(paths: list[Path | str], *, sigma_floor: float = SIGMA_FLOOR) -> Truth:
    """e averaged over every drop in these recordings (five, per the plan).

    sigma is the spread between drops, like the SD over tilt repeats for mu_k: one video shows one
    drop, so how much e varies from drop to drop is part of what it is compared against. With a
    single drop it is the spread of that drop's ratios. A floor covers air drag and onset bias."""
    drops = [d for path in paths for d in analyze_file(path)]
    values = np.array([d.e for d in drops])
    if len(values) >= 2:
        spread = float(np.std(values, ddof=1))
    else:
        spread = drops[0].e_sd or 0.0
    return Truth(
        value=float(values.mean()),
        sigma=math.hypot(spread, sigma_floor),
        method=f"audio clicks, e = dt(n+1)/dt(n), mean of {len(drops)} drop(s)",
    )


# --- command line --------------------------------------------------------------------------------
def _describe(label: str, drops: list[BounceResult]) -> list[str]:
    lines = [label]
    for i, d in enumerate(drops, 1):
        sd = "" if d.e_sd is None else f" +- {d.e_sd:.4f}"
        fit = "" if d.e_fit is None else f"   slope fit {d.e_fit:.4f}"
        lines.append(f"  drop {i}: {d.n_clicks} clicks, e = {d.e:.4f}{sd}{fit}")
        lines.append("          flights (s): " + ", ".join(f"{v:.3f}" for v in d.intervals_s))
        if d.rejected_ratios:
            lines.append(f"          rejected ratios: {[round(v, 3) for v in d.rejected_ratios]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.audio_bounce",
        description="Restitution e from the impact clicks in a recording of a bouncing ball.",
    )
    parser.add_argument("paths", nargs="+", help="WAV files, or folders holding audio.wav files")
    parser.add_argument("--start", type=float, default=0.0, help="ignore audio before this (s)")
    parser.add_argument("--end", type=float, default=None, help="ignore audio after this (s)")
    parser.add_argument("--skip-first", type=int, default=0, help="clicks to skip in each drop")
    parser.add_argument(
        "--band", type=float, nargs=2, default=DEFAULT_BAND_HZ, metavar=("LO", "HI")
    )
    parser.add_argument("--min-gap-ms", type=float, default=MIN_GAP_S * 1000)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args(argv)

    files = [f for p in args.paths for f in find_audio(p)]
    kwargs = {
        "start_s": args.start,
        "end_s": args.end,
        "skip_first": args.skip_first,
        "band": tuple(args.band),
        "min_gap_s": args.min_gap_ms / 1000,
    }
    per_file: dict[str, list[BounceResult]] = {}
    failures: dict[str, str] = {}
    for file in files:
        try:
            per_file[str(file)] = analyze_file(file, **kwargs)
        except (AudioBounceError, OSError) as exc:
            failures[str(file)] = str(exc)
    drops = [d for ds in per_file.values() for d in ds]
    summary = None
    if drops:
        values = np.array([d.e for d in drops])
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else (drops[0].e_sd or 0.0)
        summary = {
            "e": float(values.mean()),
            "sigma": math.hypot(spread, SIGMA_FLOOR),
            "spread": spread,
            "n_drops": len(drops),
        }
    if args.json:
        payload = {
            "files": {k: [asdict(d) for d in v] for k, v in per_file.items()},
            "failures": failures,
            "summary": summary,
        }
        print(json.dumps(payload, indent=2))
    else:
        for name, result in per_file.items():
            print("\n".join(_describe(name, result)))
        for name, message in failures.items():
            print(f"{name}\n  {message}")
        if summary:
            print(
                f"\ne = {summary['e']:.4f} +- {summary['sigma']:.4f} "
                f"({summary['n_drops']} drop(s); spread {summary['spread']:.4f}, "
                f"floor {SIGMA_FLOOR})"
            )
    return 0 if summary and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
