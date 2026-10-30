"""Running a model for the What-if on the server (build plan Part 18)."""

import math

import export_helpers as h
import numpy as np
import pytest
from test_whatif_context import LENGTH, pendulum, swing_summary

from ghostline.jobs.errors import StageError
from ghostline.jobs.stage_types import SimulateRequest
from ghostline.physics.library import get_model
from ghostline.physics.runner import SubprocessRunner
from ghostline.store.core_schemas import ParamSpec as CoreParamSpec
from ghostline.store.records import FitParam, FitRecord, ModelRecord
from ghostline.whatif import simulate_custom_model
from ghostline.whatif.simulate import readouts, stop_time


def run(repo, experiment, model_id, **kwargs):
    model = repo.models[model_id]
    return simulate_custom_model(
        repo, experiment, model, SimulateRequest(model_id=model_id, **kwargs)
    )


def library_model(repo, experiment, model_id):
    repo.models[model_id] = ModelRecord(experiment_id=experiment.id, id=model_id, name=model_id)


def test_a_library_model_runs_from_its_fit_and_the_clips_setup():
    repo, experiment = pendulum()

    result = run(repo, experiment, "P2", t_end_s=6.0, n_points=361)

    assert result.model_id == "P2"
    assert len(result.t) == 361 and result.t[-1] == pytest.approx(6.0)
    expected = get_model("P2", "pendulum").predict(
        np.array(result.t),
        dict(theta0=result.series["theta"][0], omega0=0.0, g=9.79, L=LENGTH, gamma=0.05),
    )[0]
    assert result.series["theta"][0] == pytest.approx(0.66, abs=0.03)  # trial 1's start
    assert result.series["theta"][1:] != []
    assert max(abs(a - b) for a, b in zip(result.series["theta"], expected, strict=True)) < 0.05
    assert result.derived["period_small_angle"] == pytest.approx(
        2 * math.pi * math.sqrt(LENGTH / 9.79)
    )


def test_overrides_change_the_law_and_the_start():
    repo, experiment = pendulum()

    base = run(repo, experiment, "P2", t_end_s=20.0, n_points=2001)
    damped = run(repo, experiment, "P2", t_end_s=20.0, n_points=2001, params={"gamma": 0.5})
    kicked = run(repo, experiment, "P2", t_end_s=5.0, n_points=501, params={"theta0": 0.3})

    assert max(abs(v) for v in damped.series["theta"][-300:]) < 0.1 * max(
        abs(v) for v in base.series["theta"][-300:]
    )
    assert kicked.series["theta"][0] == pytest.approx(0.3)


def test_the_mars_pendulum_period_is_sqrt_9_81_over_3_71_times_earths():
    repo, experiment = pendulum("P1")
    library_model(repo, experiment, "P1")

    rest = {"omega0": 0.0}  # released from rest, so the swing only changes its time scale
    earth = run(repo, experiment, "P1", t_end_s=30.0, n_points=5000, params={"g": 9.81, **rest})
    mars = run(repo, experiment, "P1", t_end_s=30.0, n_points=5000, params={"g": 3.71, **rest})

    expected = math.sqrt(9.81 / 3.71)
    assert expected == pytest.approx(1.6261, abs=1e-4)
    # The small-angle formula and the period measured on the simulated swing both obey it.
    assert mars.derived["period_small_angle"] / earth.derived[
        "period_small_angle"
    ] == pytest.approx(expected, rel=1e-12)
    assert mars.derived["period"] / earth.derived["period"] == pytest.approx(expected, rel=1e-4)


def test_a_bounce_reports_the_first_rebound_height_and_a_fall_the_terminal_velocity():
    repo, experiment = h.build(
        "bounce",
        "B0",
        [h.quantity("e", 0.8)],
        summary={"samples": {"t": [0.0, 0.2, 0.4], "x": [0, 0, 0], "y": [1.0, 0.8, 0.2]}},
    )
    library_model(repo, experiment, "B0")
    fall_repo, fall = h.build(
        "free_fall",
        "FF2",
        [h.quantity("k", 0.35)],
        summary={"samples": {"t": [0.0, 0.2, 0.4], "x": [0, 0, 0], "y": [1.0, 0.8, 0.2]}},
    )
    library_model(fall_repo, fall, "FF2")

    bounce = run(
        repo, experiment, "B0", t_end_s=4.0, n_points=4001, params={"g": 9.81, "y0": 1.0, "v0": 0.0}
    )
    drop = run(fall_repo, fall, "FF2", t_end_s=2.0, params={"g": 9.81, "k": 0.35})

    assert bounce.derived["bounce_height"] == pytest.approx(0.64, abs=2e-3)
    assert drop.derived["terminal_velocity"] == pytest.approx(math.sqrt(9.81 / 0.35))


def test_a_spinner_reports_when_it_stops():
    t = [i / 10 for i in range(101)]
    phi = [10 * x - x * x for x in t[:51]] + [25.0] * 50  # stops at t = 5

    assert stop_time(t, phi) == pytest.approx(5.0, abs=0.11)
    assert stop_time(t, [10 * x for x in t]) is None  # still turning at the end
    assert readouts("spin_down", t, {"phi": phi})["time_to_stop"] == stop_time(t, phi)
    assert readouts("bounce", t, {"y": [1, 0.5, 0.0, 0.4, 0.6, 0.4, 0.0]})[
        "bounce_height"
    ] == pytest.approx(0.6)
    assert readouts("ramp", t, {"s": phi}) == {}


def test_a_law_that_needs_what_the_clip_did_not_measure_says_so():
    repo, experiment = h.build("ramp", "R1", [h.quantity("mu_k", 0.2)], summary={})
    library_model(repo, experiment, "R1")

    with pytest.raises(StageError) as raised:
        run(repo, experiment, "R1")

    assert raised.value.code == "missing_setup"


WONDER_CODE = """
import math

def initial_state(p):
    return [p['theta0'], p['omega0']]

def rhs(t, s, p):
    return [s[1], -p['g'] / p['L'] * math.sin(s[0]) - p['gamma'] * s[1]]
"""


def wonder_model(repo, experiment, code=WONDER_CODE):
    spec = {
        "theta0": CoreParamSpec(init=0.5, lower=-3, upper=3, unit="rad", kind="initial_condition"),
        "omega0": CoreParamSpec(
            init=0.0, lower=-9, upper=9, unit="rad/s", kind="initial_condition"
        ),
        "g": CoreParamSpec(init=9.8, lower=1, upper=30, unit="m/s^2", kind="physical"),
        "gamma": CoreParamSpec(init=0.1, lower=0, upper=5, unit="1/s", kind="physical"),
    }
    repo.models["W1"] = ModelRecord(
        experiment_id=experiment.id,
        id="W1",
        name="Wonder",
        source="wonder",
        code=code,
        params_spec=spec,
    )
    repo.fits.append(
        FitRecord(
            experiment_id=experiment.id,
            model_id="W1",
            trial_id=repo.trials[0].id,
            split={"kind": "final"},
            params=[
                FitParam(name="g", value=9.79, ci_low=9.7, ci_high=9.9, unit="m/s^2"),
                FitParam(name="gamma", value=0.05, ci_low=0.04, ci_high=0.06, unit="1/s"),
            ],
        )
    )


def test_wonders_code_runs_through_the_harness_and_matches_the_library_twin():
    repo, experiment = h.build(
        "pendulum", "W1", [h.quantity("gamma", 0.05)], summary=swing_summary()
    )
    wonder_model(repo, experiment)
    request = SimulateRequest(
        model_id="W1", t_end_s=5.0, n_points=251, params={"theta0": 0.66, "omega0": 0.0}
    )

    result = simulate_custom_model(
        repo, experiment, repo.models["W1"], request, runner=SubprocessRunner()
    )

    expected = get_model("P2", "pendulum").predict(
        np.linspace(0, 5.0, 251),
        dict(theta0=0.66, omega0=0.0, g=9.79, L=LENGTH, gamma=0.05),
    )[0]
    assert max(abs(a - b) for a, b in zip(result.series["theta"], expected, strict=True)) < 1e-5
    assert result.derived["period"] == pytest.approx(
        2 * math.pi * math.sqrt(LENGTH / 9.79), rel=0.04
    )  # a 38 degree swing is a few percent slower than the small-angle period


def test_wonders_code_that_breaks_the_rules_is_refused():
    repo, experiment = h.build(
        "pendulum", "W1", [h.quantity("gamma", 0.05)], summary=swing_summary()
    )
    wonder_model(repo, experiment, code=WONDER_CODE + "\nopen('/etc/passwd').read()\n")

    with pytest.raises(StageError) as raised:
        simulate_custom_model(
            repo,
            experiment,
            repo.models["W1"],
            SimulateRequest(model_id="W1"),
            runner=SubprocessRunner(),
        )

    assert raised.value.code == "invalid_model"
