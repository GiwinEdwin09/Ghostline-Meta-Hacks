"""The ghost's prediction and divergence (build plan Part 16), on analytic synthetic trials."""

import numpy as np
import pytest
from fixtures.ghost.rig import series_for

from ghostline.ghost.divergence import find_onset, measure_divergence, status_of, threshold_mm
from ghostline.ghost.forecast import forecast_trial, freeze
from ghostline.ghost.geometry import make_geometry
from ghostline.ghost.model import GhostError
from ghostline.ghost.specs import library_spec
from ghostline.physics.runner import InProcessRunner

RUNNER = InProcessRunner()
G = 9.81


@pytest.fixture(scope="module")
def pendulum_pair():
    """Trial 1 (0.5 rad, 8 s) and a longer trial 2 (0.3 rad, 10 s) of one undamped pendulum."""
    s1, c1 = series_for("pendulum", 8.0, 60.0, 1, {"theta0": 0.5})
    s2, _ = series_for("pendulum", 10.0, 60.0, 2, {"theta0": 0.3})
    frozen = freeze(RUNNER, library_spec("P1"), "pendulum", s1, c1)
    forecast = forecast_trial(RUNNER, library_spec("P1"), "pendulum", frozen, s2)
    return s1, s2, frozen, forecast


def test_the_ghost_stays_within_tau_for_at_least_three_periods(pendulum_pair):
    _, s2, _, forecast = pendulum_pair
    div = measure_divergence(forecast, s2)
    period = 2 * np.pi * np.sqrt(s2.summary["pendulum"]["length_eff_m"] / G)
    after = div.t >= forecast.launch_end_t
    assert div.onset_t is None
    assert div.t[-1] - forecast.launch_end_t > 3 * period
    assert div.error_mm[after].max() < div.tau_mm
    # Far inside the threshold, not just under it: the law is the true one.
    assert div.error_mm[after].max() < 0.25 * div.tau_mm


def test_only_the_initial_conditions_come_from_trial_2(pendulum_pair):
    s1, _, frozen, forecast = pendulum_pair
    assert forecast.params["g"] == frozen.values["g"]  # trial 1's g, to the last bit
    assert set(forecast.ic.values) == {"theta0", "omega0"}
    assert frozen.values["g"] == pytest.approx(G, rel=0.01)
    # trial 2 swings from 0.3 rad, trial 1 from 0.5: the launch window found the new start
    assert abs(forecast.ic.values["theta0"]) == pytest.approx(0.3, abs=0.02)
    assert abs(frozen.initial["theta0"]) == pytest.approx(0.5, abs=0.02)


def test_the_launch_window_is_ten_percent_of_the_motion_or_a_quarter_second(pendulum_pair):
    _, s2, _, forecast = pendulum_pair
    motion = s2.t_end - s2.t_start
    assert forecast.launch_window_s == pytest.approx(0.1 * motion, abs=0.03)
    short, _ = series_for("pendulum", 2.0, 60.0, 3, {"theta0": 0.3})
    assert short.launch_window_s() == pytest.approx(0.25, abs=0.02)  # 10% would be 0.16 s


def test_the_spread_is_small_early_and_grows_with_the_horizon(pendulum_pair):
    _, _, _, forecast = pendulum_pair
    assert forecast.n_draws == 200
    assert np.all(np.isfinite(forecast.spread_mm)) and forecast.spread_mm.min() >= 0
    assert forecast.spread_mm[-1] > forecast.spread_mm[0]
    assert forecast.spread_mm.max() < 2.0  # millimetres: a clean fit predicts tightly


def test_a_wrong_law_drifts_and_is_measured_in_millimetres():
    """Trial 2 swings under a different gravity: the frozen law must not adapt to it."""
    s1, c1 = series_for("pendulum", 8.0, 60.0, 1, {"theta0": 0.4})
    s2, _ = series_for("pendulum", 10.0, 60.0, 2, {"theta0": 0.4, "g": 7.0})
    frozen = freeze(RUNNER, library_spec("P1"), "pendulum", s1, c1)
    forecast = forecast_trial(RUNNER, library_spec("P1"), "pendulum", frozen, s2)
    div = measure_divergence(forecast, s2)
    assert forecast.params["g"] == frozen.values["g"] == pytest.approx(G, rel=0.01)
    assert div.onset_t is not None and div.onset_t > forecast.launch_end_t
    assert div.max_error_mm > 5 * div.tau_mm
    assert "red" in div.status


@pytest.mark.parametrize(
    ("system", "model", "dur", "fps", "first", "second"),
    [
        ("free_fall", "FF0", 0.8, 120.0, {}, {}),
        ("ramp", "R1", 2.0, 60.0, {}, {}),
        ("spin_down", "S2", 4.0, 60.0, {"omega0": 30.0, "coulomb": 3.0, "viscous": 0.4},
         {"omega0": 24.0, "coulomb": 3.0, "viscous": 0.4}),
        ("bounce", "B0", 3.0, 120.0, {}, {}),
    ],
)  # fmt: skip
def test_every_system_gets_a_ghost_that_tracks_a_second_trial(
    system, model, dur, fps, first, second
):
    s1, c1 = series_for(system, dur, fps, 1, first)
    s2, _ = series_for(system, dur, fps, 2, second)
    frozen = freeze(RUNNER, library_spec(model), system, s1, c1)
    forecast = forecast_trial(RUNNER, library_spec(model), system, frozen, s2)
    div = measure_divergence(forecast, s2)
    assert div.onset_t is None
    assert forecast.xy.shape == (len(forecast.frames), 2)
    assert div.max_error_mm < div.tau_mm


def test_a_spinner_ghost_is_rotated_by_the_predicted_angle():
    s1, c1 = series_for("spin_down", 4.0, 60.0, 1, {"omega0": 30.0, "coulomb": 3.0, "viscous": 0.4})
    s2, _ = series_for("spin_down", 4.0, 60.0, 2, {"omega0": 24.0, "coulomb": 3.0, "viscous": 0.4})
    frozen = freeze(RUNNER, library_spec("S2"), "spin_down", s1, c1)
    forecast = forecast_trial(RUNNER, library_spec("S2"), "spin_down", frozen, s2)
    assert forecast.angle is not None
    turned = abs(forecast.angle[-1] - forecast.angle[0])
    assert turned > 5.0  # several radians of spin over the clip
    assert make_geometry(("phi",), s2).kind == "spinner"


def test_a_pendulum_ghost_has_a_pivot_for_its_string(pendulum_pair):
    _, s2, _, forecast = pendulum_pair
    pivot = np.asarray(s2.summary["pendulum"]["pivot_xy_m"])
    assert forecast.anchor_xy == pytest.approx(pivot)
    radius = np.hypot(*(forecast.xy - pivot).T)
    assert radius == pytest.approx(s2.summary["pendulum"]["length_m"], rel=1e-9)


def test_a_missing_frozen_parameter_is_a_clear_error():
    s1, c1 = series_for("pendulum", 4.0, 60.0, 1, {"theta0": 0.4})
    s2, _ = series_for("pendulum", 4.0, 60.0, 2, {"theta0": 0.4})
    frozen = freeze(RUNNER, library_spec("P2"), "pendulum", s1, c1)
    frozen.values.pop("gamma")
    with pytest.raises(GhostError, match="gamma"):
        forecast_trial(RUNNER, library_spec("P2"), "pendulum", frozen, s2)


# --- divergence ----------------------------------------------------------------------------
def test_tau_is_three_sigma_or_half_the_radius_whichever_is_larger():
    assert threshold_mm(0.4, 20.0) == 10.0  # the radius rules a clean measurement
    assert threshold_mm(5.0, 20.0) == 15.0  # noise rules a poor one


def test_the_outline_is_green_below_tau_amber_up_to_2_tau_and_red_beyond():
    assert [status_of(e, 10.0) for e in (0.0, 9.99, 10.0, 19.9, 20.0, 50.0)] == [
        "green", "green", "amber", "amber", "red", "red",
    ]  # fmt: skip


def test_onset_needs_five_frames_in_a_row_over_tau_after_the_launch_window():
    t = np.arange(40) / 10.0
    e = np.zeros(40)
    e[8:12] = 20.0  # four frames over: not an onset
    e[20:] = 20.0  # then a sustained run
    assert find_onset(t, e, 10.0, after_t=0.0) == pytest.approx(2.0)
    e[22] = 0.0  # a dip breaks the run; the next five start at index 23
    assert find_onset(t, e, 10.0, after_t=0.0) == pytest.approx(2.3)
    # an exceedance inside the launch window never counts
    early = np.zeros(40)
    early[2:9] = 20.0
    assert find_onset(t, early, 10.0, after_t=1.0) is None
