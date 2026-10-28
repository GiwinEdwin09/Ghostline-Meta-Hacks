"""Tracking with SAM 3.1: moving objects, the retry ladder, and the calibration sheet (Part 7).

track = track_object(meta, file_id=..., media_sha256=..., key="ball", prompt="orange foam ball",
                     frame_size=(1920, 1080), n_frames=600, events_dir=sam_dir)
sheet = segment_sheet(meta, master_path, prompt="white sheet of paper")
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt

from ghostline.llm import MetaClient, SamResult

from .frames import (
    count_frames,
    encode_image,
    median_frame,
    read_frames,
    spread_indices,
    video_size,
)
from .masks import decode_mask, mask_features, place_mask
from .parse import SamStreamError, frame_size_of, parse_sam_output
from .tracks import Track, select_primary, stitch_tracks, tracks_from_result

log = logging.getLogger("ghostline.vision")

MIN_TRACK_FRAMES = 3
MAX_REPHRASES = 2
FILMING_TIPS = (
    "Use a plain, contrasting background, keep the object fully in frame and in focus, "
    "and film in steady light on a tripod (see the setup checklist)."
)


class TrackingFailed(Exception):
    """No rung of the retry ladder found the object. The message is safe to show the user."""

    def __init__(self, key: str, attempts: list[str]):
        super().__init__(
            f"Ghostline couldn't find the '{key}' in the video (tried: {'; '.join(attempts)}). "
            f"{FILMING_TIPS}"
        )
        self.key = key
        self.attempts = attempts


@dataclass(frozen=True)
class BoxHint:
    """A box around the object on one frame, in the master's pixel coordinates."""

    frame: int
    left: int
    top: int
    right: int  # half-open
    bottom: int  # half-open


@dataclass
class RescueHooks:
    """How the ladder asks for help (both are Wonder calls; see `ghostline.scene.rescue`).

    - `rephrase(previous_prompt, attempt)` returns a new concept prompt, or None to give up.
    - `locate_box()` returns a `BoxHint` for the object, or None.
    """

    rephrase: Callable[[str, int], str | None] | None = None
    locate_box: Callable[[], BoxHint | None] | None = None


@dataclass
class TrackResult:
    key: str
    track: Track
    candidates: list[Track]  # every stitched track the winning request returned
    prompt: str  # the concept prompt, or the box prompt text, that worked
    attempts: list[str]  # the ladder rungs tried, in order
    sam: SamResult
    events_path: Path | None
    cost_usd: float  # every SAM request, including failed rungs


def _tracks_from(
    sam: SamResult, frame_size: tuple[int, int] | None, min_frames: int
) -> tuple[list[Track], tuple[int, int]]:
    size = frame_size or frame_size_of(sam.output_text)
    if size is None:
        return [], (0, 0)
    parsed = parse_sam_output(sam.output_text, "completed", "video")
    errors = [d for d in parsed.diagnostics if d.severity == "error"]
    if errors:
        log.warning("SAM output had %d malformed lines, first: %s", len(errors), errors[0].message)
    tracks = stitch_tracks(tracks_from_result(parsed, size))
    return [t for t in tracks if len(t) >= min_frames], size


def track_object(
    meta: MetaClient,
    *,
    file_id: str,
    key: str,
    prompt: str,
    frame_size: tuple[int, int] | None = None,
    n_frames: int | None = None,
    media_sha256: str | None = None,
    rescue: RescueHooks | None = None,
    accept: Callable[[Track], bool] | None = None,
    events_dir: str | Path | None = None,
    experiment_id: str | None = None,
    min_frames: int = MIN_TRACK_FRAMES,
) -> TrackResult:
    """Track one object through the master video, climbing the retry ladder if needed.

    1. The concept `prompt`.
    2. Up to two Wonder rephrasings (`rescue.rephrase`).
    3. A Wonder bounding box on one frame, sent as a box prompt with `propagation_direction:
       both` (`rescue.locate_box`).
    4. `TrackingFailed`, with filming tips.

    A rung succeeds when a stitched track of at least `min_frames` frames comes back (and
    `accept(track)` is true, when given). Of several candidate tracks the one with the largest
    centroid displacement wins. The raw event stream of the winning request is saved to
    `events_dir/<key>.events.jsonl`.
    """
    rescue = rescue or RescueHooks()
    attempts: list[str] = []
    cost = 0.0

    def run(prompt_text: str, label: str, metadata: dict[str, str] | None = None):
        nonlocal cost
        sam = meta.sam_video(
            file_id,
            prompt_text,
            metadata=metadata,
            media_sha256=media_sha256,
            frames_hint=n_frames,
            experiment_id=experiment_id,
            phase="track",
        )
        cost += 0 if sam.cache_hit else sam.cost_usd
        candidates, size = _tracks_from(sam, frame_size, min_frames)
        chosen = select_primary(candidates, min_frames)
        ok = chosen is not None and (accept is None or accept(chosen))
        attempts.append(f"{label}: {'found' if ok else 'nothing usable'}")
        return sam, candidates, chosen, ok

    def finish(sam, candidates, chosen, used_prompt) -> TrackResult:
        path = None
        if events_dir is not None:
            path = sam.write_events_jsonl(Path(events_dir) / f"{key}.events.jsonl")
        return TrackResult(
            key, chosen.renamed(key), candidates, used_prompt, attempts, sam, path, cost
        )

    sam, candidates, chosen, ok = run(prompt, f"prompt '{prompt}'")
    if ok:
        return finish(sam, candidates, chosen, prompt)

    current = prompt
    for attempt in range(1, MAX_REPHRASES + 1):
        if rescue.rephrase is None:
            break
        new_prompt = rescue.rephrase(current, attempt)
        if not new_prompt or new_prompt == current:
            break
        current = new_prompt
        sam, candidates, chosen, ok = run(current, f"rephrased '{current}'")
        if ok:
            return finish(sam, candidates, chosen, current)

    if rescue.locate_box is not None:
        hint = rescue.locate_box()
        size = frame_size or frame_size_of(sam.output_text)
        if hint is not None and size is None:
            attempts.append("box: the frame size is unknown, pass frame_size")
        elif hint is not None:
            text = box_prompt(hint, size)
            sam, candidates, chosen, ok = run(
                text, f"box on frame {hint.frame}", {"propagation_direction": "both"}
            )
            if ok:
                return finish(sam, candidates, chosen, text)
    raise TrackingFailed(key, attempts)


def box_prompt(hint: BoxHint, frame_size: tuple[int, int]) -> str:
    """The SAM prompt text for one box on one frame (right and bottom are half-open)."""
    return build_object_prompt(
        size=frame_size,
        objects=[
            PromptObject(
                id=1,
                prompts=[
                    FramePrompt(
                        frame=hint.frame,
                        box=PromptBox(hint.left, hint.top, hint.right, hint.bottom),
                    )
                ],
            )
        ],
    )


@dataclass
class SheetMask:
    """One segmentation of the sheet."""

    frame: int | None  # master frame index, or None for the median image
    mask: np.ndarray  # full-frame uint8, 1 inside the sheet
    box: tuple[int, int, int, int]  # SAM's box: left, top, right, bottom (half-open)
    confidence: float | None
    area: float
    centroid: tuple[float, float]


@dataclass
class SheetResult:
    """The sheet from the median image, and from the first, middle and last frames (to detect
    camera motion, Part 8)."""

    median: SheetMask
    median_bgr: np.ndarray
    probes: list[SheetMask]  # first, middle and last frame, in that order
    frame_size: tuple[int, int]
    cost_usd: float
    sam: list[SamResult] = field(default_factory=list, repr=False)

    def write(self, folder: str | Path) -> Path:
        """Write `sheet_mask.png` (255 inside the sheet), `median.png` and `sheet.json` (boxes,
        confidences and the probe frames) next to `tracks.parquet`."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sheet_mask.png").write_bytes(encode_image(self.median.mask * 255))
        (folder / "median.png").write_bytes(encode_image(self.median_bgr))
        summary = {
            "frame_size": list(self.frame_size),
            "median": _mask_summary(self.median),
            "probes": [_mask_summary(m) for m in self.probes],
        }
        (folder / "sheet.json").write_text(json.dumps(summary, indent=1))
        return folder / "sheet.json"


def _mask_summary(mask: SheetMask) -> dict:
    return {
        "frame": mask.frame,
        "box": list(mask.box),
        "confidence": mask.confidence,
        "area": mask.area,
        "centroid": list(mask.centroid),
    }


def _largest_sheet(
    sam: SamResult, frame_size: tuple[int, int], frame: int | None
) -> SheetMask | None:
    parsed = parse_sam_output(sam.output_text, "completed", "image")
    best = None
    for record in parsed.records:
        if record.kind != "mask":
            continue
        raster = decode_mask(record.mask)
        features = mask_features(raster, record.bounds.left, record.bounds.top, frame_size)
        if features and (best is None or features.area > best[1].area):
            best = (record, features, raster)
    if best is None:
        return None
    record, features, raster = best
    return SheetMask(
        frame=frame,
        mask=place_mask(raster, record.bounds.left, record.bounds.top, frame_size),
        box=(
            int(record.bounds.left),
            int(record.bounds.top),
            int(record.bounds.right),
            int(record.bounds.bottom),
        ),
        confidence=record.confidence,
        area=features.area,
        centroid=(features.cx, features.cy),
    )


def segment_sheet(
    meta: MetaClient,
    video_path: str | Path,
    *,
    prompt: str = "white sheet of paper",
    n_frames: int | None = None,
    median_count: int = 15,
    experiment_id: str | None = None,
) -> SheetResult:
    """Segment the calibration sheet. The sheet doesn't move, so it comes from one still: the
    median of 15 frames spread across the clip, which removes the moving object. The first,
    middle and last frames are segmented too, to detect camera motion. Four image requests,
    under $0.01. Masks stay lossless: corner precision matters."""
    n_frames = n_frames or count_frames(video_path)
    size = video_size(video_path)
    median = median_frame(video_path, median_count, n_frames)
    indices = spread_indices(n_frames, 3)
    first, middle, last = indices[0], indices[len(indices) // 2], indices[-1]
    stills = read_frames(video_path, set(indices))

    found, sam_results = [], []
    for frame, image in [
        (None, median),
        (first, stills[first]),
        (middle, stills[middle]),
        (last, stills[last]),
    ]:
        sam = meta.sam_image(
            encode_image(image),
            prompt,
            mime="image/png",
            experiment_id=experiment_id,
            phase="sheet",
        )
        sam_results.append(sam)
        mask = _largest_sheet(sam, size, frame)
        if mask is None and frame is None:
            raise TrackingFailed("calibration sheet", [f"prompt '{prompt}': nothing found"])
        found.append(mask)
    cost = sum(0 if s.cache_hit else s.cost_usd for s in sam_results)
    return SheetResult(
        median=found[0],
        median_bgr=median,
        probes=[m for m in found[1:] if m is not None],
        frame_size=size,
        cost_usd=cost,
        sam=sam_results,
    )


__all__ = [
    "BoxHint",
    "RescueHooks",
    "SamStreamError",
    "SheetMask",
    "SheetResult",
    "TrackResult",
    "TrackingFailed",
    "box_prompt",
    "segment_sheet",
    "track_object",
]
