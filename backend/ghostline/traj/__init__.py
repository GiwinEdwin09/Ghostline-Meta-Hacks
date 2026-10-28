"""World transform, cleaning, events, noise and the out-of-plane check (build plan Part 9).

Main entry point: `build_trajectory(tracks, calib, system)`, which returns a `TrajectoryResult`;
`.write(dir)` produces trajectory.parquet and traj_summary.json.
"""

from .model import (
    FLAG_ALIAS_RISK,
    FLAG_AT_REST,
    FLAG_NEAR_IMPACT,
    TrackSeries,
    TrajectoryResult,
)
from .pipeline import TrajectoryError, build_trajectory

__all__ = [
    "FLAG_ALIAS_RISK",
    "FLAG_AT_REST",
    "FLAG_NEAR_IMPACT",
    "TrackSeries",
    "TrajectoryError",
    "TrajectoryResult",
    "build_trajectory",
]
