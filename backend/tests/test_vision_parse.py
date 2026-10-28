"""The SAM parser glue, on the stream recorded from a real response (Part 1, check 5)."""

import cv2
import numpy as np
import pandas as pd
import pytest
from fixtures import scenarios
from fixtures.synthetic_clip import ClipTruth

from ghostline.vision import (
    TRACK_COLUMNS,
    SamStreamError,
    decode_mask,
    frame_size_of,
    parse_sam_events,
    parse_sam_output,
    place_mask,
    read_events_jsonl,
    tracks_from_result,
    write_tracks_parquet,
)
from ghostline.vision.parse import output_text_of

SIZE = scenarios.FRAME_SIZE


@pytest.fixture
def sam(monkeypatch):
    meta, _, _ = scenarios.replay_clients(monkeypatch)
    return {
        "ball": scenarios.sam_video(meta, scenarios.BALL_PROMPT),
        "dot": scenarios.sam_video(meta, scenarios.DOT_PROMPT),
        "sheet": scenarios.sam_sheet(meta),
    }


@pytest.fixture
def truth():
    return ClipTruth.from_json(scenarios.TRUTH.read_text())


def test_events_parse_to_one_box_and_one_mask_per_frame(sam):
    result = parse_sam_events(sam["ball"].events)

    assert not result.diagnostics
    assert result.outcome.status == "completed"
    boxes = [r for r in result.records if r.kind == "box"]
    masks = [r for r in result.records if r.kind == "mask"]
    assert len(boxes) == len(masks) == 90
    assert {r.object_id for r in masks} == {"0"}
    assert [r.frame.frame_index for r in masks] == list(range(90))
    assert all(r.confidence is not None and 0 < r.confidence <= 1 for r in masks)


def test_events_and_output_text_parse_the_same(sam):
    text, status = output_text_of(sam["ball"].events)

    assert text == sam["ball"].output_text and status == "completed"
    assert frame_size_of(text) == SIZE
    assert len(parse_sam_output(text).records) == len(parse_sam_events(sam["ball"].events).records)


def test_mask_rasters_are_box_sized_and_anchored_at_the_box_top_left(sam, truth):
    """Day 0 check 5: the raster is exactly the box's size, so its origin is the box's top-left."""
    masks = [r for r in parse_sam_events(sam["ball"].events).records if r.kind == "mask"]
    errors = []
    for record in masks:
        raster = decode_mask(record.mask)
        assert raster.shape == (
            round(record.bounds.bottom - record.bounds.top),
            round(record.bounds.right - record.bounds.left),
        )
        ys, xs = np.nonzero(raster)
        centre = (xs.mean() + record.bounds.left, ys.mean() + record.bounds.top)
        errors.append(np.hypot(*(np.array(centre) - truth.centers[record.frame.frame_index])))
    assert np.median(errors) < 0.3 and max(errors) < 1.0  # against the true ball centre, pixels


def test_a_ball_is_tracked_continuously_with_subpixel_centroids(sam, truth):
    (track,) = tracks_from_result(parse_sam_events(sam["ball"].events), SIZE)

    assert len(track) == 90 and (track.frame == np.arange(90)).all()
    centers = truth.centers[:90]  # the recorded clip is the first 3 s
    error = np.hypot(track.cx - centers[:, 0], track.cy - centers[:, 1])
    assert np.median(error) < 0.3 and error.max() < 1.0
    assert not track.touches_border.any() and not track.stitched.any()
    assert track.sam_ids == ["0"]
    assert track.radius == pytest.approx(18, rel=0.1)
    assert track.extent > 100  # it moved
    assert 0 < track.mean_confidence <= 1


def test_a_static_object_jitters_under_a_third_of_a_pixel(sam):
    """Part 7 'done when' (on a synthetic clip, a proxy for a phone clip)."""
    (dot,) = tracks_from_result(parse_sam_events(sam["dot"].events), SIZE)

    assert dot.jitter_rms() < 0.3
    assert dot.extent < 1.0  # never mistaken for a mover


def test_a_track_decodes_its_mask_on_demand(sam):
    (track,) = tracks_from_result(parse_sam_events(sam["ball"].events), SIZE)

    mask = track.mask(45)

    assert mask.shape == (480, 854) and mask.sum() == track.area[45]
    ys, xs = np.nonzero(mask)
    assert (xs.mean(), ys.mean()) == pytest.approx((track.cx[45], track.cy[45]), abs=1e-6)


def test_the_sheet_mask_lines_up_with_the_true_sheet(sam, truth):
    parsed = parse_sam_output(sam["sheet"].output_text, media="image")
    (record,) = [r for r in parsed.records if r.kind == "mask"]
    placed = place_mask(decode_mask(record.mask), record.bounds.left, record.bounds.top, SIZE)
    true_sheet = np.zeros((480, 854), np.uint8)
    cv2.fillConvexPoly(true_sheet, np.round(truth.sheet_corners).astype(np.int32), 1)

    iou = (placed & true_sheet).sum() / (placed | true_sheet).sum()

    assert iou > 0.97
    assert record.confidence is not None


def test_the_event_stream_round_trips_through_jsonl(sam, tmp_path):
    path = sam["ball"].write_events_jsonl(tmp_path / "sam" / "ball.events.jsonl")

    again = read_events_jsonl(path)

    assert again == sam["ball"].events
    assert len(parse_sam_events(again).records) == 180


def test_a_failed_stream_raises():
    events = [
        {"type": "response.created"},
        {"type": "response.failed", "response": {"error": {"message": "boom"}}},
    ]

    with pytest.raises(SamStreamError, match="boom"):
        output_text_of(events)


def test_text_is_read_from_the_done_event_when_no_deltas_arrive():
    line = "<0f>0<|box;x1=0;y1=0;x2=0;y2=0;w=10;h=10|><|mask;x=0;y=0;data=1,1,~!!!!!|>\n"
    events = [
        {"type": "response.output_text.done", "text": line},
        {"type": "response.completed", "response": {}},
    ]

    assert output_text_of(events) == (line, "completed")


def test_an_incomplete_stream_is_not_reported_as_completed():
    assert output_text_of([{"type": "response.incomplete"}])[1] == "incomplete"
    assert output_text_of([])[1] == "eof"


def test_tracks_parquet_has_the_documented_columns(sam, tmp_path):
    (track,) = tracks_from_result(parse_sam_events(sam["ball"].events), SIZE)
    timestamps = np.arange(300) / 30.0

    path = write_tracks_parquet(
        tmp_path / "trial" / "tracks.parquet", [track.renamed("ball")], timestamps
    )

    table = pd.read_parquet(path)
    assert list(table.columns) == TRACK_COLUMNS
    assert len(table) == 90 and set(table.object_key) == {"ball"}
    assert table.t.iloc[10] == pytest.approx(10 / 30)
    assert table.cx.iloc[0] == pytest.approx(track.cx[0])
    assert (table["flags"] == 0).all()


def test_parquet_without_timestamps_leaves_t_empty(sam, tmp_path):
    (track,) = tracks_from_result(parse_sam_events(sam["ball"].events), SIZE)

    table = pd.read_parquet(write_tracks_parquet(tmp_path / "t.parquet", [track]))

    assert table.t.isna().all()
