import numpy as np
import pytest

from ghostline.vision import (
    FLAG_BORDER,
    FLAG_STITCHED,
    TRACK_COLUMNS,
    Track,
    select_primary,
    stitch_tracks,
)

AREA = 300.0  # radius about 9.8 px


def make_track(key, frames, cx, cy, conf=0.8, area=AREA, border=None):
    n = len(frames)
    return Track(
        key=key,
        frame_size=(854, 480),
        frame=np.array(frames, dtype=np.int64),
        cx=np.array(cx, dtype=float),
        cy=np.array(cy, dtype=float),
        area=np.full(n, area),
        theta=np.zeros(n),
        eccentricity=np.zeros(n),
        conf=np.full(n, conf),
        box=np.zeros((n, 4)),
        touches_border=np.zeros(n, bool) if border is None else np.array(border, bool),
        stitched=np.zeros(n, bool),
        sam_id=np.array([key] * n, dtype="<U32"),
    )


def mover(key, start, stop, x0, speed=5.0, y=100.0, conf=0.8):
    frames = list(range(start, stop))
    return make_track(
        key, frames, [x0 + speed * (f - start) for f in frames], [y] * len(frames), conf
    )


def test_an_object_that_reappears_where_it_was_heading_is_joined():
    first = mover("0", 0, 40, x0=100)  # last seen at x=295, heading right at 5 px/frame
    second = mover("7", 50, 90, x0=345 + 12)  # reappears 12 px from where it should be

    (joined,) = stitch_tracks([second, first])

    assert len(joined) == 80 and joined.key == "0"
    assert (np.diff(joined.frame) > 0).all()
    assert joined.sam_ids == ["0", "7"]
    assert joined.stitched[:40].sum() == 0 and joined.stitched[40:].all()
    assert (joined.flags[40:] == FLAG_STITCHED).all()


def test_an_object_that_reappears_far_from_its_heading_is_a_new_object():
    first = mover("0", 0, 40, x0=100)
    far = mover("7", 50, 90, x0=345 + 3 * 9.8 + 30)  # beyond three radii of the prediction

    assert len(stitch_tracks([first, far])) == 2


def test_the_prediction_uses_the_heading_not_the_last_position():
    first = mover("0", 0, 40, x0=100)  # last at 295, will be at 345 ten frames later
    behind = mover("7", 50, 90, x0=295)  # back where it was: 50 px behind the prediction

    assert len(stitch_tracks([first, behind])) == 2


def test_tracks_that_overlap_in_time_are_different_objects():
    first = mover("0", 0, 60, x0=100)
    second = mover("1", 30, 90, x0=130)  # right beside the first, at the same time

    assert len(stitch_tracks([first, second])) == 2


def test_a_long_gap_is_not_bridged():
    first = mover("0", 0, 40, x0=100)
    late = mover("7", 120, 160, x0=295 + 5 * 81)  # the right place, 81 frames later

    assert len(stitch_tracks([first, late])) == 2
    assert len(stitch_tracks([first, late], max_gap_frames=100)) == 1


def test_a_chain_of_reappearances_is_one_track():
    a = mover("0", 0, 20, x0=0)
    b = mover("1", 25, 45, x0=125)
    c = mover("2", 50, 70, x0=250)

    (joined,) = stitch_tracks([a, b, c])

    assert len(joined) == 60 and joined.sam_ids == ["0", "1", "2"]


def test_two_separate_objects_each_keep_their_own_track():
    ball = mover("0", 0, 40, x0=100, y=100)
    other = mover("1", 0, 40, x0=100, y=300)
    returns = mover("2", 45, 80, x0=100 + 5 * 39 + 25, y=302)  # the second object, back again

    tracks = stitch_tracks([ball, other, returns])

    assert len(tracks) == 2
    assert sorted(len(t) for t in tracks) == [40, 75]


def test_the_object_with_the_largest_displacement_is_chosen():
    static = make_track("0", range(100), [200 + 0.4 * (-1) ** i for i in range(100)], [50.0] * 100)
    moving = mover("1", 0, 100, x0=10, speed=3.0)

    assert select_primary([static, moving]).key == "1"


def test_jitter_on_a_static_object_does_not_beat_real_motion():
    rng = np.random.default_rng(3)
    jittery = make_track(
        "0", range(600), 300 + rng.normal(0, 0.5, 600), 100 + rng.normal(0, 0.5, 600)
    )
    slow = mover("1", 0, 20, x0=10, speed=2.0)  # 38 px of real motion in 20 frames

    assert select_primary([jittery, slow]).key == "1"


def test_ties_are_broken_by_mean_confidence():
    a = mover("0", 0, 30, x0=0, conf=0.6)
    b = mover("1", 0, 30, x0=0, conf=0.9, y=400)

    assert select_primary([a, b]).key == "1"


def test_very_short_tracks_only_count_when_nothing_longer_exists():
    spike = make_track("0", [0, 1], [0.0, 800.0], [0.0, 0.0])  # a two-frame false detection
    real = mover("1", 0, 50, x0=100, speed=1.0)

    assert select_primary([spike, real]).key == "1"
    assert select_primary([spike]).key == "0"
    assert select_primary([]) is None


def test_flags_and_the_table():
    track = make_track(
        "ball", [3, 4, 5], [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], border=[True, False, False]
    )

    table = track.to_frame(timestamps=np.arange(10) / 10)

    assert list(table.columns) == TRACK_COLUMNS
    assert table["flags"].tolist() == [FLAG_BORDER, 0, 0]
    assert table.t.tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert table.object_key.tolist() == ["ball"] * 3
    assert track.renamed("x").key == "x" and track.key == "ball"
