"""SAM 3.1 tracking (build plan Part 7): the client glue, mask decoding and features, track
stitching, and the calibration sheet.

    from ghostline.vision import track_object, segment_sheet, write_tracks_parquet
"""

from .frames import count_frames, median_frame, read_frames, video_size
from .masks import MaskFeatures, decode_mask, mask_features, place_mask
from .parse import (
    SamStreamError,
    frame_size_of,
    parse_sam_events,
    parse_sam_output,
    read_events_jsonl,
)
from .tracker import (
    BoxHint,
    RescueHooks,
    SheetMask,
    SheetResult,
    TrackingFailed,
    TrackResult,
    box_prompt,
    segment_sheet,
    track_object,
)
from .tracks import (
    FLAG_BORDER,
    FLAG_STITCHED,
    TRACK_COLUMNS,
    Track,
    select_primary,
    stitch_tracks,
    tracks_from_result,
    write_tracks_parquet,
)

__all__ = [
    "FLAG_BORDER",
    "FLAG_STITCHED",
    "TRACK_COLUMNS",
    "BoxHint",
    "MaskFeatures",
    "RescueHooks",
    "SamStreamError",
    "SheetMask",
    "SheetResult",
    "Track",
    "TrackResult",
    "TrackingFailed",
    "box_prompt",
    "count_frames",
    "decode_mask",
    "frame_size_of",
    "mask_features",
    "median_frame",
    "parse_sam_events",
    "parse_sam_output",
    "place_mask",
    "read_events_jsonl",
    "read_frames",
    "segment_sheet",
    "select_primary",
    "stitch_tracks",
    "track_object",
    "tracks_from_result",
    "video_size",
    "write_tracks_parquet",
]
