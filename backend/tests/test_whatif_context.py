"""The What-if tab's data, built from a finished analysis (build plan Part 18)."""

import math

import export_helpers as h
import numpy as np
import pytest

from ghostline.api.views import WhatIfView
from ghostline.physics.library import get_model
from ghostline.store.core_schemas import ParamSpec as CoreParamSpec
from ghostline.store.records import FitParam, FitRecord, ModelRecord
from ghostline.whatif import build_whatif

PIVOT = (0.05, 0.7)
LENGTH = 0.62


def swing_summary(g=9.79, gamma=0.05, theta0=0.66, window=4.0):
    """60 samples of a damped swing, shaped like the trajectory summary's `samples`."""
    t = np.linspace(0.0, window, 60)
    theta = get_model("P2", "pendulum").predict(
        t, dict(theta0=theta0, omega0=0.0, g=g, L=LENGTH, gamma=gamma)
    )[0]
    return {
        "pendulum": {
            "length_m": LENGTH - 0.001,
            "length_eff_m": LENGTH,
            "bob_radius_m": 0.012,
            "pivot_xy_m": list(PIVOT),
        },
        "samples": {
            "t": t.tolist(),
            "x": (PIVOT[0] + LENGTH * np.sin(theta)).tolist(),
            "y": (PIVOT[1] - LENGTH * np.cos(theta)).tolist(),
            "theta": theta.tolist(),
        },
        "ranges": {"theta": [float(theta.min()), float(theta.max())]},
    }


def pendulum(model="P2", **kwargs):
    repo, experiment = h.build(
        "pendulum",
        model,
        [h.quantity("g", 9.79, 0.01), h.quantity("gamma", 0.05, 0.04)],
        summary=swing_summary(),
        mover="steel nut",
        surface=None,
        size_cm=2.4,
        **kwargs,
    )
    trial = repo.trials[0]
    trial.paths["poster"] = "trials/x/poster.jpg"
    final = FitRecord(
        experiment_id=experiment.id,
        model_id=model,
        trial_id=trial.id,
        split={"kind": "final"},
        params=[
            FitParam(name="g", value=9.79, ci_low=9.7, ci_high=9.9, unit="m/s^2"),
            FitParam(name="gamma", value=0.05, ci_low=0.04, ci_high=0.06, unit="1/s"),
            FitParam(name="period_small_angle", value=1.58, ci_low=1.5, ci_high=1.6, unit="s"),
        ],
    )
    repo.fits.append(final)
    return repo, experiment


def test_a_pendulum_gets_its_fitted_law_its_setup_and_its_anchor():
    repo, experiment = pendulum()

    ctx = build_whatif(repo, experiment)

    assert ctx is not None
    view = WhatIfView.model_validate(ctx)  # the shape the API serves
    assert (view.system, view.model_ref, view.sim_supported) == ("pendulum", "P2", True)
    assert view.params == {"g": 9.79, "gamma": 0.05}  # physical parameters only, no derived ones
    assert view.fixed == {"L": LENGTH}
    assert view.initial["theta0"] == pytest.approx(0.66, abs=0.03)
    assert abs(view.initial["omega0"]) < 0.3
    assert view.anchor_m == PIVOT
    assert view.object.name.startswith("steel nut")
    assert view.object.radius_m == pytest.approx(0.012)
    assert view.still_url == f"{experiment.id}/trials/x/poster.jpg"
    low, high = view.measured_range["amplitude_deg"]
    assert high == pytest.approx(math.degrees(0.66), abs=0.5) and low < high


def test_a_ramp_gets_its_angle_travel_and_direction():
    t = np.linspace(0.0, 1.2, 60)
    alpha = math.radians(22)
    s = 0.5 * 2.1 * t**2
    summary = {
        "samples": {
            "t": t.tolist(),
            "x": (0.1 + s * math.cos(alpha)).tolist(),
            "y": (0.9 - s * math.sin(alpha)).tolist(),
        },
    }
    repo, experiment = h.build(
        "ramp",
        "R1",
        [h.quantity("mu_k", 0.25)],
        calibration={"ramp": {"angle_deg": 22.0, "incline_deg": 22.0, "sigma_deg": 0.3}},
        summary=summary,
    )
    repo.fits.append(
        FitRecord(
            experiment_id=experiment.id,
            model_id="R1",
            trial_id=repo.trials[0].id,
            split={"kind": "final"},
            params=[FitParam(name="mu_k", value=0.25, ci_low=0.2, ci_high=0.3, unit="")],
        )
    )

    view = WhatIfView.model_validate(build_whatif(repo, experiment))

    assert view.params == {"mu_k": 0.25}
    assert view.fixed["alpha"] == pytest.approx(alpha)
    assert view.fixed["g"] == pytest.approx(9.80665) and view.fixed["dir_x"] == 1.0
    assert view.fixed["s_max"] == pytest.approx(1.2 * s[-1], rel=1e-3)
    assert view.initial["v0"] == pytest.approx(0.0, abs=0.2) and view.initial["s0"] == 0.0
    assert view.measured_range["angle_deg"] == (17.0, 27.0)
    assert view.anchor_m == pytest.approx((0.1, 0.9))


def test_a_bounce_is_measured_from_the_floor_up():
    t = np.linspace(0.0, 0.4, 60)
    summary = {
        "samples": {"t": t.tolist(), "x": [0.3] * 60, "y": (1.2 - 0.5 * 9.81 * t**2).tolist()}
    }
    repo, experiment = h.build("bounce", "B0", [h.quantity("e", 0.8)], summary=summary)

    view = WhatIfView.model_validate(build_whatif(repo, experiment))

    floor = 1.2 - 0.5 * 9.81 * 0.4**2
    assert view.anchor_m == pytest.approx((0.3, floor))
    assert view.initial["y0"] == pytest.approx(1.2 - floor, abs=0.01)
    assert view.initial["v0"] == pytest.approx(0.0, abs=0.2)
    assert (
        view.measured_range["height_m"][0] < view.initial["y0"] < view.measured_range["height_m"][1]
    )


def test_a_spinner_gets_its_direction_hub_and_spin_range():
    t = np.linspace(0.0, 6.0, 60)
    phi = -(30 * t - 2.0 * t**2)  # clockwise, slowing down
    summary = {
        "samples": {"t": t.tolist(), "phi": phi.tolist()},
        "spinner": {"hub_xy_m": [0.2, 0.4], "marker_radius_m": 0.03, "direction": "cw"},
    }
    repo, experiment = h.build("spin_down", "S1", [h.quantity("c", 4.0)], summary=summary)

    view = WhatIfView.model_validate(build_whatif(repo, experiment))

    assert view.fixed["direction"] == -1.0 and view.fixed["marker_radius_m"] == 0.03
    assert view.initial["omega0"] == pytest.approx(30.0, rel=0.1)
    assert view.anchor_m == (0.2, 0.4)
    low, high = view.measured_range["spin_rev_s"]
    assert 0 < low < high and high == pytest.approx(30 * 1.1 / (2 * math.pi), rel=0.1)


def test_a_wonder_written_winner_is_played_on_the_server():
    repo, experiment = h.build(
        "pendulum", "W1", [h.quantity("gamma", 0.05)], summary=swing_summary(), source="wonder"
    )
    repo.models["W1"] = ModelRecord(
        experiment_id=experiment.id,
        id="W1",
        name="Wonder's model",
        source="wonder",
        code="def rhs(t, s, p): ...",
        params_spec={
            "gamma": CoreParamSpec(init=0.05, lower=0, upper=1, unit="1/s", kind="physical"),
            "theta0": CoreParamSpec(
                init=0.0, lower=-3, upper=3, unit="rad", kind="initial_condition"
            ),
        },
    )
    repo.fits.append(
        FitRecord(
            experiment_id=experiment.id,
            model_id="W1",
            trial_id=repo.trials[0].id,
            split={"kind": "final"},
            params=[
                FitParam(name="gamma", value=0.05, ci_low=0.04, ci_high=0.06, unit="1/s"),
                FitParam(name="theta0", value=0.66, ci_low=0.6, ci_high=0.7, unit="rad"),
            ],
        )
    )

    view = WhatIfView.model_validate(build_whatif(repo, experiment))

    assert (view.model_ref, view.sim_supported, view.system) == ("W1", False, "pendulum")
    assert view.params == {"gamma": 0.05}  # the initial condition is not a physical parameter


def test_systems_the_what_if_cannot_play_give_nothing():
    repo, oscillator = h.build("oscillator", "O1", [h.quantity("gamma", 0.1)])
    none_yet = h.build("pendulum", "P2", [h.quantity("gamma", 0.1)])
    none_yet[1].conclusion = None

    assert build_whatif(repo, oscillator) is None
    assert build_whatif(*none_yet) is None
    assert (
        build_whatif(*pendulum()[:1], pendulum()[1].model_copy(update={"winner_model_id": None}))
        is None
    )


def test_a_clip_without_the_geometry_gives_nothing():
    summary = swing_summary()
    summary["pendulum"] = {}
    repo, experiment = h.build("pendulum", "P2", [h.quantity("gamma", 0.1)], summary=summary)

    assert build_whatif(repo, experiment) is None
