"""Per-object, per-frame tracks from SAM output: features, stitching and selection (Part 7).

A `Track` is column arrays plus references to each frame's mask, which is the easiest shape to
hand to calibration and trajectory code:

    track.frame                 (n,) int, ascending master frame indices
    track.cx, track.cy          (n,) sub-pixel centroid in master pixels (centre of the top-left
                                pixel is (0, 0))
    track.area, track.theta, track.eccentricity, track.conf   (n,) conf is NaN when absent
    track.box                   (n, 4) SAM's box: left, top, right, bottom (half-open)
    track.touches_border        (n,) bool: exclude the centroid, the object is cut off
    track.stitched              (n,) bool: frames joined in from another SAM ID
    track.mask(frame)           full-frame uint8 mask, decoded on demand
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from meta_sam_parser import SegmentationMaskRecord, SegmentationResult

from .masks import decode_mask, mask_features, place_mask

FLAG_BORDER = 1  # the mask touches the frame border
FLAG_STITCHED = 2  # the frame came from a SAM ID joined to this track by stitching

TRACK_COLUMNS = [
    "object_key",
    "frame",
    "t",
    "cx",
    "cy",
    "area",
    "theta",
    "eccentricity",
    "conf",
    "box_left",
    "box_top",
    "box_right",
    "box_bottom",
    "flags",
]


@dataclass
class Track:
    key: str  # the scene's object key once chosen, else the SAM object ID
    frame_size: tuple[int, int]  # (width, height) of the master
    frame: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area: np.ndarray
    theta: np.ndarray
    eccentricity: np.ndarray
    conf: np.ndarray
    box: np.ndarray
    touches_border: np.ndarray
    stitched: np.ndarray
    sam_id: np.ndarray  # (n,) the SAM object ID each frame came from
    masks: dict[int, SegmentationMaskRecord] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def sam_ids(self) -> list[str]:
        return list(dict.fromkeys(self.sam_id.tolist()))

    @property
    def radius(self) -> float:
        """Median equivalent-disc radius in pixels."""
        return float(np.sqrt(np.median(self.area) / math.pi))

    @property
    def extent(self) -> float:
        """The diagonal of the centroids' bounding box: how far the object moved. Unlike path
        length, jitter on a static object doesn't inflate it."""
        return float(np.hypot(np.ptp(self.cx), np.ptp(self.cy)))

    @property
    def mean_confidence(self) -> float:
        valid = self.conf[~np.isnan(self.conf)]
        return float(valid.mean()) if len(valid) else 0.0

    @property
    def flags(self) -> np.ndarray:
        return self.touches_border * FLAG_BORDER + self.stitched * FLAG_STITCHED

    def mask(self, frame: int) -> np.ndarray:
        """The full-frame 0/1 mask for a master frame index."""
        record = self.masks[frame]
        return place_mask(
            decode_mask(record.mask), record.bounds.left, record.bounds.top, self.frame_size
        )

    def jitter_rms(self) -> float:
        """RMS distance of the centroid from its mean, in pixels. For a static object this is
        the tracking noise."""
        return float(
            np.sqrt(np.mean((self.cx - self.cx.mean()) ** 2 + (self.cy - self.cy.mean()) ** 2))
        )

    def to_frame(self, timestamps: Sequence[float] | np.ndarray | None = None) -> pd.DataFrame:
        """One row per frame with the `TRACK_COLUMNS`. `t` is the real time of each frame from
        the timing model (NaN when `timestamps` isn't given)."""
        t = np.full(len(self), np.nan) if timestamps is None else np.asarray(timestamps)[self.frame]
        return pd.DataFrame(
            {
                "object_key": self.key,
                "frame": self.frame.astype(np.int64),
                "t": t,
                "cx": self.cx,
                "cy": self.cy,
                "area": self.area,
                "theta": self.theta,
                "eccentricity": self.eccentricity,
                "conf": self.conf,
                "box_left": self.box[:, 0],
                "box_top": self.box[:, 1],
                "box_right": self.box[:, 2],
                "box_bottom": self.box[:, 3],
                "flags": self.flags.astype(np.int64),
            },
            columns=TRACK_COLUMNS,
        )

    def renamed(self, key: str) -> Track:
        return Track(**{**self.__dict__, "key": key})


def _build(
    key: str, rows: list[tuple[int, str, SegmentationMaskRecord]], frame_size: tuple[int, int]
) -> Track | None:
    """Features for (frame, sam_id, record) rows, one row per frame."""
    kept = []
    for frame, sam_id, record in sorted(rows, key=lambda r: r[0]):
        raster = decode_mask(record.mask)
        features = mask_features(raster, record.bounds.left, record.bounds.top, frame_size)
        if features is not None:
            kept.append((frame, sam_id, record, features))
    if not kept:
        return None
    return Track(
        key=key,
        frame_size=frame_size,
        frame=np.array([k[0] for k in kept], dtype=np.int64),
        cx=np.array([k[3].cx for k in kept]),
        cy=np.array([k[3].cy for k in kept]),
        area=np.array([k[3].area for k in kept]),
        theta=np.array([k[3].theta for k in kept]),
        eccentricity=np.array([k[3].eccentricity for k in kept]),
        conf=np.array([np.nan if k[2].confidence is None else k[2].confidence for k in kept]),
        box=np.array(
            [
                [k[2].bounds.left, k[2].bounds.top, k[2].bounds.right, k[2].bounds.bottom]
                for k in kept
            ]
        ),
        touches_border=np.array([k[3].touches_border for k in kept], dtype=bool),
        stitched=np.zeros(len(kept), dtype=bool),
        sam_id=np.array([k[1] for k in kept], dtype="<U32"),
        masks={k[0]: k[2] for k in kept},
    )


def tracks_from_result(result: SegmentationResult, frame_size: tuple[int, int]) -> list[Track]:
    """One track per SAM object ID, with features computed on every decoded mask.

    Frames are keyed by the `<Nf>` marker and the object ID, never by line position. If a frame
    carries the same object twice, the latest revision wins.
    """
    rows: dict[str, dict[int, SegmentationMaskRecord]] = defaultdict(dict)
    for record in result.records:
        if record.kind == "mask" and record.frame is not None:
            previous = rows[record.object_id].get(record.frame.frame_index)
            if previous is None or record.revision >= previous.revision:
                rows[record.object_id][record.frame.frame_index] = record
    tracks = []
    for sam_id, by_frame in rows.items():
        track = _build(sam_id, [(f, sam_id, r) for f, r in by_frame.items()], frame_size)
        if track is not None:
            tracks.append(track)
    return tracks


def _heading(track: Track, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """The last position of a track and its velocity (pixels per frame) from the last k frames."""
    n = min(k, len(track))
    frames = track.frame[-n:].astype(float)
    xy = np.stack([track.cx[-n:], track.cy[-n:]], axis=1)
    if n < 2 or frames[-1] == frames[0]:
        return xy[-1], np.zeros(2)
    slope = np.polyfit(frames, xy, 1)[0]
    return xy[-1], slope


def stitch_tracks(
    tracks: Sequence[Track], radius_mult: float = 3.0, max_gap_frames: int = 30
) -> list[Track]:
    """Join tracks when an object reappears under a new SAM ID (IDs aren't reliable across exits,
    entries and full occlusion).

    A track that starts after another ended joins it if its first centroid lies within
    `radius_mult` object radii of where the earlier track was heading (a linear extrapolation of
    its last frames), and the gap is at most `max_gap_frames`. Tracks that overlap in time are
    different objects and are never joined.
    """
    chains: list[Track] = []
    for track in sorted(tracks, key=lambda t: t.frame[0]):
        best, best_distance = None, math.inf
        for i, chain in enumerate(chains):
            gap = int(track.frame[0] - chain.frame[-1])
            if gap < 1 or gap > max_gap_frames:
                continue
            position, velocity = _heading(chain)
            predicted = position + velocity * gap
            distance = float(np.hypot(track.cx[0] - predicted[0], track.cy[0] - predicted[1]))
            if distance <= radius_mult * chain.radius and distance < best_distance:
                best, best_distance = i, distance
        if best is None:
            chains.append(track)
        else:
            chains[best] = _join(chains[best], track)
    return chains


def _join(first: Track, second: Track) -> Track:
    return Track(
        key=first.key,
        frame_size=first.frame_size,
        frame=np.concatenate([first.frame, second.frame]),
        cx=np.concatenate([first.cx, second.cx]),
        cy=np.concatenate([first.cy, second.cy]),
        area=np.concatenate([first.area, second.area]),
        theta=np.concatenate([first.theta, second.theta]),
        eccentricity=np.concatenate([first.eccentricity, second.eccentricity]),
        conf=np.concatenate([first.conf, second.conf]),
        box=np.concatenate([first.box, second.box]),
        touches_border=np.concatenate([first.touches_border, second.touches_border]),
        stitched=np.concatenate([first.stitched, np.ones(len(second), dtype=bool)]),
        sam_id=np.concatenate([first.sam_id, second.sam_id]),
        masks={**first.masks, **second.masks},
    )


def select_primary(tracks: Sequence[Track], min_frames: int = 3) -> Track | None:
    """Choose among several matches (for example two balls in the frame): the track whose
    centroid moved the most (`extent`), ties broken by mean confidence. Tracks shorter than
    `min_frames` only count when nothing longer exists."""
    pool = [t for t in tracks if len(t) >= min_frames] or list(tracks)
    if not pool:
        return None
    return max(pool, key=lambda t: (round(t.extent, 3), t.mean_confidence))


def write_tracks_parquet(
    path: str | Path,
    tracks: Sequence[Track],
    timestamps: Sequence[float] | np.ndarray | None = None,
) -> Path:
    """Write `tracks.parquet`: `TRACK_COLUMNS`, one row per object and frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [t.to_frame(timestamps) for t in tracks]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRACK_COLUMNS)
    table.to_parquet(path, index=False)
    return path
