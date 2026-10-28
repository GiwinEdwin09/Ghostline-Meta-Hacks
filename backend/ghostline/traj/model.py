"""Inputs and outputs of the trajectory stage.

trajectory.parquet columns (the physics code depends on these names):
    t          real seconds since the first frame of the clip (timestamps.npy)
    x, y       metres in the physics frame (origin at the sheet centre, y up)
    theta      pendulum angle from the downward vertical, radians (pendulum only)
    phi        spinner angle, unwrapped, radians counter-clockwise (spin-down only)
    s          distance along the incline from the first kept point, metres (ramp only)
    weight     tracking confidence in [0, 1]
    flags      bit field, see the FLAG_* constants
    segment_id 0 everywhere except bounces, where it counts the flights
plus display-only extras: frame (index in the master), vx, vy, omega (Savitzky-Golay-style
derivatives, for plots and diagnostics, never for fitting).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# flags column
FLAG_AT_REST = 1  # before the release or after the stop, outside the motion window
FLAG_NEAR_IMPACT = 2  # the kept frame closest to an impact on either side
FLAG_ALIAS_RISK = 4  # spinner angle step above 120 degrees per frame near this row
FLAG_BORDER = 1  # bit 0 of the `flags` column of tracks.parquet: mask touches the frame border

# why a frame was dropped (TrajectoryResult.frames["drop"])
DROP_REASONS = ("low_conf", "border", "area_outlier", "jump", "impact")


@dataclass
class TrackSeries:
    """One tracked object, one entry per frame in which it was found."""

    frame: np.ndarray  # frame index in the master
    t: np.ndarray  # real seconds
    cx: np.ndarray  # centroid in pixels (sub-pixel)
    cy: np.ndarray
    area: np.ndarray  # mask area in pixels
    conf: np.ndarray  # SAM confidence 0-1
    border: np.ndarray  # mask touches the frame border

    def __post_init__(self):
        n = len(self.frame)
        for name in ("t", "cx", "cy", "area", "conf", "border"):
            arr = np.asarray(getattr(self, name))
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} entries, expected {n}")
            setattr(self, name, arr.astype(bool if name == "border" else float))
        self.frame = np.asarray(self.frame).astype(int)
        order = np.argsort(self.frame, kind="stable")
        if np.any(order != np.arange(n)):
            for name in ("frame", "t", "cx", "cy", "area", "conf", "border"):
                setattr(self, name, getattr(self, name)[order])

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def px(self) -> np.ndarray:
        return np.column_stack([self.cx, self.cy])

    @classmethod
    def from_arrays(
        cls, frame, cx, cy, area=None, conf=None, border=None, timestamps=None, t=None
    ) -> TrackSeries:
        """Build from per-frame arrays; times come from `t` or `timestamps[frame]`."""
        frame = np.asarray(frame).astype(int)
        n = len(frame)
        if t is None:
            if timestamps is None:
                raise ValueError("give either t or timestamps")
            t = np.asarray(timestamps, float)[frame]
        return cls(
            frame, t, cx, cy,
            np.full(n, np.nan) if area is None else area,
            np.ones(n) if conf is None else conf,
            np.zeros(n, bool) if border is None else border,
        )  # fmt: skip

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, object_key: str | None = None, timestamps=None
    ) -> TrackSeries:
        """Read rows of tracks.parquet (object_key, frame, t, cx, cy, area, conf, flags)."""
        if object_key is not None:
            df = df[df["object_key"] == object_key]
        if "border" in df:
            border = df["border"].to_numpy(bool)
        elif "flags" in df:
            border = (df["flags"].to_numpy(int) & FLAG_BORDER) != 0
        else:
            border = None
        return cls.from_arrays(
            df["frame"].to_numpy(), df["cx"].to_numpy(), df["cy"].to_numpy(),
            df["area"].to_numpy() if "area" in df else None,
            df["conf"].to_numpy() if "conf" in df else None,
            border, timestamps, df["t"].to_numpy() if "t" in df else None,
        )  # fmt: skip


@dataclass
class TrajectoryResult:
    trajectory: pd.DataFrame  # kept rows, the contents of trajectory.parquet
    frames: pd.DataFrame  # every input frame with its `drop` reason ("" if kept)
    summary: dict
    notices: list = field(default_factory=list)

    def write(self, out_dir: str | Path) -> None:
        """Write trajectory.parquet and traj_summary.json."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.trajectory.to_parquet(out / "trajectory.parquet", index=False)
        (out / "traj_summary.json").write_text(json.dumps(self.summary, indent=2, default=_jsonable))


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} is not JSON serializable")
