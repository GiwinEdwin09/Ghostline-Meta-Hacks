"""One trial's measured data, as the ghost code needs it (build plan Parts 9 and 16).

`TrialSeries` bundles trajectory.parquet, its summary, the calibration, the frame timestamps and
the pixel track of the moving body. The pixel track is what the video shows; the trajectory is
what the physics saw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ghostline.calib import Calibration
from ghostline.physics.analysis import build_series
from ghostline.physics.harness.schema import SeriesData
from ghostline.traj import FLAG_AT_REST

from .model import LAUNCH_FRACTION, LAUNCH_MIN_POINTS, LAUNCH_MIN_S, GhostError

POSITION_COLUMNS = ("x", "y", "s")
ANGLE_COLUMNS = ("theta", "phi")


@dataclass
class TrialSeries:
    system: str
    df: pd.DataFrame  # the rows of trajectory.parquet
    summary: dict  # traj_summary.json
    calib: Calibration
    timestamps: np.ndarray  # real seconds of every frame of the master
    track: pd.DataFrame | None = None  # frame, cx, cy, area, conf of the moving body

    # --- construction -----------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        *,
        system: str,
        trajectory_parquet: Path,
        summary: dict,
        calibration: dict,
        timestamps_npy: Path | None,
        tracks_parquet: Path | None,
        object_key: str | None,
    ) -> TrialSeries:
        df = pd.read_parquet(trajectory_parquet)
        calib = Calibration.model_validate(calibration)
        stamps = _timestamps(timestamps_npy, df)
        track = None
        if tracks_parquet is not None and tracks_parquet.exists():
            table = pd.read_parquet(tracks_parquet)
            if object_key is not None and "object_key" in table:
                table = table[table["object_key"] == object_key]
            if len(table):
                track = table.sort_values("frame").reset_index(drop=True)
        return cls(system, df, summary, calib, stamps, track)

    # --- the motion window ------------------------------------------------------------------
    @property
    def in_window(self) -> np.ndarray:
        """Rows that are inside the motion: not before the release or after the stop."""
        flags = (
            self.df["flags"].to_numpy(int) if "flags" in self.df else np.zeros(len(self.df), int)
        )
        window = (flags & FLAG_AT_REST) == 0
        return window if window.sum() >= 2 else np.ones(len(self.df), bool)

    @property
    def motion(self) -> pd.DataFrame:
        return self.df[self.in_window]

    @property
    def t_start(self) -> float:
        return float(self.motion["t"].iloc[0])

    @property
    def t_end(self) -> float:
        return float(self.motion["t"].iloc[-1])

    @property
    def frame_range(self) -> tuple[int, int]:
        m = self.motion
        return int(m["frame"].iloc[0]), int(m["frame"].iloc[-1])

    def launch_window_s(self) -> float:
        """The first 10% of the motion or 0.25 s, whichever is longer. A slow frame rate gets
        a longer window so the fit has enough points."""
        t = self.motion["t"].to_numpy()
        duration = self.t_end - self.t_start
        length = max(LAUNCH_FRACTION * duration, LAUNCH_MIN_S)
        needed = float(t[min(LAUNCH_MIN_POINTS, len(t)) - 1] - t[0])
        return max(min(length, 0.5 * duration), needed)

    def launch_rows(self) -> pd.DataFrame:
        m = self.motion
        return m[m["t"] <= self.t_start + self.launch_window_s() + 1e-9]

    # --- noise, geometry --------------------------------------------------------------------
    def sigma(self, observable: str) -> float:
        noise = self.summary.get("noise", {})
        key = "sigma_angle_rad" if observable in ANGLE_COLUMNS else "sigma_pos_m"
        value = noise.get(key)
        if value is None or not math.isfinite(value) or value <= 0:
            value = 1e-3
        return float(value)

    @property
    def down(self) -> np.ndarray:
        """The 'down' direction in the sheet frame that this trial's world frame was built with
        (the trajectory stage may refine the camera-level default from the data)."""
        angle = self.summary.get("gravity", {}).get("angle_deg")
        if angle is None:
            return self.calib.gravity_down
        a = math.radians(angle)
        return np.array([math.sin(a), math.cos(a)])

    def object_radius_mm(self) -> float:
        """Radius of the moving body, from the median mask area."""
        area = None
        if self.track is not None and "area" in self.track:
            area = float(np.nanmedian(self.track["area"].to_numpy()))
        if area is None or not math.isfinite(area) or area <= 0:
            return 0.0
        return math.sqrt(area / math.pi) * self.calib.m_per_px * 1000.0

    def real_fps(self) -> float:
        dt = np.diff(self.timestamps)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else 0.0

    # --- model inputs -----------------------------------------------------------------------
    def series_data(
        self, observables: tuple[str, ...] | list[str], rows: pd.DataFrame | None = None
    ) -> SeriesData:
        """The harness input for these observables on `rows` (default: the whole motion)."""
        rows = self.motion if rows is None else rows
        missing = [o for o in observables if o not in rows.columns]
        if missing:
            raise GhostError(f"The measured trajectory has no {missing} column for this model.")
        observed = {o: rows[o].to_numpy(float) for o in observables}
        weights = rows["weight"].to_numpy(float) if "weight" in rows else None
        segment = rows["segment_id"].to_numpy(int) if "segment_id" in rows else None
        sigma = {o: self.sigma(o) for o in observables}
        return build_series(rows["t"].to_numpy(float), observed, sigma, weights, segment)

    # --- pixels -----------------------------------------------------------------------------
    def actual_px(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel centroid of the moving body at each frame, and whether it was measured there.

        Frames with no measurement are filled by linear interpolation for drawing only.
        """
        frames = np.asarray(frames, int)
        if self.track is not None and len(self.track) >= 2:
            have = self.track["frame"].to_numpy(int)
            xy = self.track[["cx", "cy"]].to_numpy(float)
        else:
            have = self.df["frame"].to_numpy(int)
            xy = self.calib.world_to_px(self.df[["x", "y"]].to_numpy(float), self.down)
        out = np.column_stack([np.interp(frames, have, xy[:, k]) for k in range(2)])
        return out, np.isin(frames, have)


def _timestamps(path: Path | None, df: pd.DataFrame) -> np.ndarray:
    if path is not None and path.exists():
        return np.load(path).astype(float)
    # No timestamps file: extend the trajectory's own (frame, t) pairs to every frame.
    frame = df["frame"].to_numpy(float)
    t = df["t"].to_numpy(float)
    n = int(frame.max()) + 1
    return np.interp(np.arange(n), frame, t, left=None, right=None)
