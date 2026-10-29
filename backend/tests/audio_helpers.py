"""Synthetic bouncing-ball recordings with a known restitution, for the audio tests."""

import wave
from pathlib import Path

import numpy as np


def impact_times(e: float, n_clicks: int = 9, first_flight: float = 0.62, start: float = 0.4):
    """Impact n+1 follows impact n by first_flight * e^n seconds (so dt_{n+1} / dt_n = e)."""
    flights = first_flight * e ** np.arange(n_clicks - 1)
    return start + np.concatenate([[0.0], np.cumsum(flights)])


def click_track(
    times,
    *,
    amplitudes=None,
    e: float = 0.8,
    rate: int = 48_000,
    duration: float | None = None,
    noise: float = 0.02,
    hum: float = 0.5,
    hum_hz: float = 60.0,
    rumble: float = 0.3,
    click_hz: float = 3200.0,
    tau_ms: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Float samples: a damped-sine click at each impact time (louder for faster impacts, in
    proportion to e^n), white noise, mains hum with harmonics, and a low rumble.

    Click times are real numbers, evaluated between samples, so a track has no grid-aligned onsets
    for a detector to get right by luck."""
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    duration = duration or float(times[-1] + 0.5)
    t = np.arange(round(duration * rate)) / rate
    x = np.zeros_like(t)
    if amplitudes is None:
        amplitudes = e ** np.arange(len(times))
    for amplitude, t0 in zip(amplitudes, times, strict=True):
        dt = t - t0
        live = dt >= 0
        phase = rng.uniform(0, 2 * np.pi)
        burst = np.exp(-dt[live] / (tau_ms / 1000))
        wave_ = np.sin(2 * np.pi * click_hz * dt[live] + phase)
        wave_ += 0.5 * np.sin(2 * np.pi * 1.7 * click_hz * dt[live] + 2 * phase)
        x[live] += amplitude * burst * wave_
    for harmonic in range(1, 6):
        x += hum / harmonic * np.sin(2 * np.pi * hum_hz * harmonic * t + rng.uniform(0, 6.28))
    x += rumble * np.sin(2 * np.pi * 14 * t + rng.uniform(0, 6.28))
    x += rng.normal(0.0, noise, size=t.shape)
    return x


def write_wav(path: Path, x: np.ndarray, rate: int = 48_000, *, width: int = 2, channels: int = 1):
    """Write PCM, scaled so the loudest sample is 0.9 of full scale."""
    x = np.asarray(x) / (np.max(np.abs(x)) + 1e-12) * 0.9
    if width == 2:
        raw = (x * 32767).astype("<i2")
        frames = np.repeat(raw[:, None], channels, axis=1).tobytes()
    elif width == 3:
        ints = (x * (2**23 - 1)).astype(np.int32)
        packed = np.stack([(ints >> s) & 0xFF for s in (0, 8, 16)], axis=1).astype(np.uint8)
        frames = np.repeat(packed[:, None, :], channels, axis=1).tobytes()
    else:
        raise ValueError(width)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(frames)
    return path


def drops_recording(es, *, gap: float = 3.0, rate: int = 48_000, **kw) -> np.ndarray:
    """Several drops one after another in a single recording, `gap` seconds of quiet between."""
    times, amplitudes, offset = [], [], 0.0
    for i, e in enumerate(es):
        drop = impact_times(e, start=offset + (0.4 if i == 0 else gap))
        times.append(drop)
        amplitudes.append(e ** np.arange(len(drop)))
        offset = float(drop[-1])
    return click_track(
        np.concatenate(times), amplitudes=np.concatenate(amplitudes), rate=rate, **kw
    )
