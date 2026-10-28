import numpy as np
import pytest

from ghostline.confidence import QualityFlag, assess_confidence, motion_extent_flags
from ghostline.physics.analysis import analyze
from ghostline.physics.harness.schema import Diagnostics
from ghostline.physics.select import ModelScore, Selection
from ghostline.physics.synthetic import make_case


def selection(status="decided", won=3, tied_with=(), equal_or_simpler=(), systematic=False):
    diag = Diagnostics(n=100, n_params=2, rmse=0.1, chi2=100, chi2_red=1.0, systematic=systematic)
    return Selection(
        status=status,
        selected_id="W",
        splits_won=won,
        n_splits=3,
        tied_with=list(tied_with),
        tied_equal_or_simpler=list(equal_or_simpler),
        message="The data can't distinguish W from R.",
        models=[ModelScore(id="W", status="ok", full_diagnostics=diag)],
    )


def test_high_needs_every_condition():
    out = assess_confidence(selection(), system="pendulum", review="accepted")

    assert out.level == "high" and not out.cant_tell
    assert out.codes == {"all_clear"}
    assert out.fixes == []


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(sel=selection(won=2)), "partial_win"),
        (dict(sel=selection(systematic=True)), "systematic_residual"),
        (dict(flags=[QualityFlag(code="blur", note="some motion blur")]), "blur"),
        (dict(review="accepted_with_caveats"), "review_caveats"),
        (dict(review=None), "not_reviewed"),
        (dict(review="not_reviewed"), "not_reviewed"),
        (
            dict(sel=selection("tied", 0, tied_with=["R"], equal_or_simpler=[])),
            "parsimony",
        ),
    ],
)
def test_each_medium_reason_alone_gives_medium(kwargs, expected):
    sel = kwargs.pop("sel", selection())
    kwargs.setdefault("review", "accepted")

    out = assess_confidence(sel, system="free_fall", **kwargs)

    assert out.level == "medium" and not out.cant_tell
    assert expected in out.codes


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(sel=selection("tied", 0, tied_with=["R"], equal_or_simpler=["R"])), "tied"),
        (dict(sel=selection("inconclusive", 0)), "inconclusive"),
        (dict(flags=[QualityFlag.major("out_of_plane")]), "out_of_plane"),
        (dict(flags=[QualityFlag.major("camera_motion")]), "camera_motion"),
        (dict(flags=[QualityFlag.major("time_base_uncertain")]), "time_base_uncertain"),
        (dict(flags=motion_extent_flags(0.3)), "short_duration"),
        (dict(flags=motion_extent_flags(2.0, period_s=1.5)), "few_periods"),
        (dict(review="contested"), "contested"),
    ],
)
def test_each_low_reason_alone_gives_low_with_a_fix(kwargs, expected):
    sel = kwargs.pop("sel", selection())
    kwargs.setdefault("review", "accepted")

    out = assess_confidence(sel, system="pendulum", **kwargs)

    assert out.level == "low" and out.cant_tell
    assert expected in out.codes
    assert out.fixes, "low confidence must say what to change"


def test_low_beats_medium_and_the_fixes_are_specific():
    flags = [QualityFlag.major("out_of_plane"), QualityFlag(code="blur")]
    out = assess_confidence(
        selection("tied", 0, tied_with=["R"], equal_or_simpler=["R"]),
        system="pendulum", flags=flags, review="accepted_with_caveats",
    )  # fmt: skip

    assert out.level == "low"
    assert any("amplitude" in f for f in out.fixes)
    assert any("plane of the sheet" in f for f in out.fixes)
    assert [r.level for r in out.reasons].count("low") >= 2


def test_motion_extent_flags():
    assert motion_extent_flags(3.0, period_s=1.0) == []
    codes = {f.code for f in motion_extent_flags(0.4, period_s=0.5)}
    assert codes == {"short_duration", "few_periods"}
    assert all(f.severity == "major" for f in motion_extent_flags(0.4, period_s=0.5))


def test_the_winners_systematic_residual_check_is_read_from_the_selection_by_default():
    out = assess_confidence(selection(systematic=True), review="accepted")
    assert "systematic_residual" in out.codes
    out = assess_confidence(
        selection(systematic=True), review="accepted", systematic_residual=False
    )
    assert out.level == "high"


def test_a_short_pendulum_clip_comes_out_low_because_it_holds_under_two_periods():
    rng = np.random.default_rng(7)
    t = np.arange(0, 2.2, 1 / 60)  # about 1.4 periods of a 0.6 m pendulum
    case = make_case(
        "pendulum", "P1", dict(theta0=0.7, omega0=0.0, g=9.81), {"L": 0.6}, t, 2e-3, rng
    )

    out = analyze(
        case.t, case.observed, case.sigma, system="pendulum", constants={"L": 0.6},
        library_ids=["P0", "P1"], review="accepted", n_boot=20, n_perturb=6,
    )  # fmt: skip

    assert out.confidence.level == "low"
    assert "few_periods" in out.confidence.codes
    assert any("two full periods" in f for f in out.confidence.fixes)
