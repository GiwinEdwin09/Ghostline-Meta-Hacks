import struct

import numpy as np
import pytest

from ghostline.physics.harness.schema import (
    BootstrapSpec,
    FitJob,
    FitResult,
    ModelSpec,
    Perturbation,
    SeriesData,
    SimulateSpec,
)
from ghostline.physics.runner import InProcessRunner, SubprocessRunner
from ghostline.physics.synthetic import make_case

G = 9.81


def drop_case(seed=0, sigma=5e-4, model_id="FF0", **extra):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 0.5, 1 / 240)
    params = dict(y0=1.2, vy0=0.0, g=G, **extra)
    return make_case("free_fall", model_id, params, {}, t, sigma, rng)


def param(model_result, name, which="full"):
    fit = model_result.full if which == "full" else model_result.splits[0]
    return next(p for p in fit.params if p.name == name)


def test_recovers_g_within_two_sigma():
    case = drop_case()

    result = InProcessRunner().run(case.job(["FF0"], plots=False))

    g = param(result.model("FF0"), "g")
    assert g.se and abs(g.value - G) < 2 * g.se
    assert g.ci_low < G < g.ci_high
    assert result.model("FF0").full.train.chi2_red == pytest.approx(1.0, abs=0.3)


def test_select_result_has_heldout_scores_for_each_split():
    result = InProcessRunner().run(drop_case().job(["FF0", "FF1"], plots=False))

    for model in result.models:
        assert model.status == "ok"
        assert [s.split for s in model.splits] == [0.65, 0.5, 0.8]
        for s in model.splits:
            assert s.n_train + s.n_test == s.n_train + len(s.heldout_errors)
            assert s.heldout_rmse == pytest.approx(np.sqrt(np.mean(np.square(s.heldout_errors))))
            assert len(s.heldout_draws) == 24 and len(s.heldout_draws[0]) == s.n_test
            assert min(s.t_test) > 0
        assert model.full.split == 1.0 and model.aic is not None and model.bic is not None
    assert result.model("FF0").n_physical == 1 and result.model("FF1").n_physical == 2


def test_chronological_split_trains_on_the_first_65_percent():
    result = InProcessRunner().run(drop_case().job(["FF0"], plots=False))

    split = result.model("FF0").splits[0]

    assert split.t_split == pytest.approx(0.65 * 119 / 240, abs=1 / 240)
    assert split.n_train == pytest.approx(0.65 * 120, abs=2)


def test_plots_are_1000_by_600_pngs_with_an_agent_variant():
    runner = InProcessRunner()
    result = runner.run(drop_case().job(["FF0", "FF1"], plots=True))

    for model in result.models:
        assert len(model.plots) == 4
        audiences = {p.id: p.audience for p in model.plots}
        assert audiences[f"{model.id}.fit_train"] == "agent"
        assert audiences[f"{model.id}.fit"] == "reviewer"
        for ref in model.plots:
            with open(ref.path, "rb") as fh:
                head = fh.read(24)
            assert head[:8] == b"\x89PNG\r\n\x1a\n"
            width, height = struct.unpack(">II", head[16:24])
            assert (width, height) == (1000, 600)


def test_subprocess_runner_follows_the_sandbox_contract():
    job = drop_case().job(["FF0", "FF2"], plots=False)

    in_process = InProcessRunner().run(job)
    subprocess_result = SubprocessRunner().run(job)

    assert isinstance(subprocess_result, FitResult)
    for model_id in ("FF0", "FF2"):
        a, b = in_process.model(model_id), subprocess_result.model(model_id)
        assert a.splits[0].heldout_rmse == pytest.approx(b.splits[0].heldout_rmse, rel=1e-6)
        assert param(a, "g").value == pytest.approx(param(b, "g").value, rel=1e-6)


def test_worker_pool_gives_the_same_answer_as_one_process():
    job = drop_case().job(["FF0", "FF1", "FF2"], plots=False)

    one = InProcessRunner().run(job)
    two = InProcessRunner().run(job.model_copy(update={"workers": 2}))

    for model_id in ("FF0", "FF1", "FF2"):
        assert one.model(model_id).splits[0].heldout_rmse == pytest.approx(
            two.model(model_id).splits[0].heldout_rmse, rel=1e-6
        )


def test_custom_model_code_fits_like_the_library_model():
    code = """
import numpy as np

def initial_state(p):
    return [p['y0'], p['vy0']]

def rhs(t, s, p):
    return [s[1], -p['g'] - p['b'] * s[1]]
"""
    specs = {
        "y0": dict(init=1.0, lower=-5, upper=5, unit="m", kind="initial_condition"),
        "vy0": dict(init=0.0, lower=-10, upper=10, unit="m/s", kind="initial_condition"),
        "g": dict(init=9.0, lower=1, upper=30, unit="m/s^2"),
        "b": dict(init=0.1, lower=0, upper=10, unit="1/s"),
    }
    case = drop_case(model_id="FF1", b=1.0)
    job = case.job(["FF1"], plots=False)
    job.models.append(
        ModelSpec(id="W1", code=code, params=specs, observables=["y"], state=["y", "vy"])
    )

    result = InProcessRunner().run(job)

    assert result.model("W1").status == "ok"
    assert param(result.model("W1"), "g").value == pytest.approx(
        param(result.model("FF1"), "g").value, rel=1e-3
    )


def test_gaps_nan_and_zero_weights_are_dropped_without_interpolating():
    case = drop_case()
    y = case.observed["y"].copy()
    y[10:14] = np.nan
    weights = np.ones_like(y)
    weights[30] = 0.0
    y[30] = 1e3  # a wild value that has zero weight must not matter
    data = SeriesData(
        t=case.t.tolist(),
        series={"y": [None if np.isnan(v) else float(v) for v in y]},
        sigma=case.sigma,
        weights=weights.tolist(),
    )
    job = case.job(["FF0"], plots=False).model_copy(update={"data": data})

    result = InProcessRunner().run(job)

    g = param(result.model("FF0"), "g")
    assert abs(g.value - G) < 3 * g.se
    assert result.model("FF0").full.train.n == len(y) - 5


def test_final_mode_adds_bootstrap_intervals_and_perturbed_refits():
    case = drop_case()
    perts = [
        Perturbation(scale=1.01, eps={"scale": 0.01}),
        Perturbation(scale=0.99, eps={"scale": -0.01}),
        Perturbation(time_scale=1.01, eps={"timing": 0.01}),
        Perturbation(time_scale=0.99, eps={"timing": -0.01}),
    ]
    job = case.job(
        ["FF0"], mode="final", plots=False, bootstrap=BootstrapSpec(n=60, block=10, seed=3),
        perturbations=perts,
    )  # fmt: skip

    model = InProcessRunner().run(job).model("FF0")

    g = param(model, "g")
    assert model.bootstrap_done == 60
    assert g.ci_low < g.value < g.ci_high
    base = g.value
    reps = [r["g"] for r in model.replicates]
    assert reps[0] == pytest.approx(base * 1.01, rel=2e-3)  # lengths scale g
    assert reps[1] == pytest.approx(base * 0.99, rel=2e-3)
    assert reps[2] == pytest.approx(base / 1.01**2, rel=2e-3)  # time^-2
    assert model.replicate_eps[2] == {"timing": 0.01}


def test_bootstrap_with_a_fixed_seed_is_reproducible():
    job = drop_case().job(["FF0"], mode="final", plots=False, bootstrap=BootstrapSpec(n=30, seed=5))

    a = param(InProcessRunner().run(job).model("FF0"), "g")
    b = param(InProcessRunner().run(job).model("FF0"), "g")

    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_fixed_parameters_are_held_and_constants_are_required():
    case = drop_case(model_id="FF1", b=1.0)
    job = case.job(["FF1"], plots=False)
    job.models[0] = ModelSpec(id="FF1", library_ref="FF1", fixed={"g": G})
    fixed_run = InProcessRunner().run(job).model("FF1")
    assert "g" not in {p.name for p in fixed_run.full.params}
    assert fixed_run.n_physical == 1

    pend = make_case(
        "pendulum", "P1", dict(theta0=0.5, omega0=0.0, g=G), {"L": 0.5},
        np.arange(300) / 60, 2e-3, np.random.default_rng(0),
    )  # fmt: skip
    job = pend.job(["P1"], plots=False).model_copy(update={"constants": {}})
    missing = InProcessRunner().run(job).model("P1")
    assert missing.status == "invalid" and "constants" in missing.error


def test_a_model_that_overruns_the_time_limit_reports_a_timeout():
    job = drop_case().job(["FF2"], plots=False, time_limit_s=1e-6)

    result = InProcessRunner().run(job)

    assert result.model("FF2").status == "timeout"


def test_simulate_mode_evaluates_a_model_at_requested_times():
    job = FitJob(
        mode="simulate",
        system="free_fall",
        models=[ModelSpec(id="FF0", library_ref="FF0")],
        simulate=SimulateSpec(
            times=[0.0, 0.1, 0.2], params={"FF0": {"y0": 1.0, "vy0": 0.0, "g": 10.0}}
        ),
    )

    out = InProcessRunner().run(job).simulations["FF0"]["y"]

    assert out == pytest.approx([1.0, 1.0 - 5 * 0.01, 1.0 - 5 * 0.04])


def test_pendulum_through_the_ode_path_picks_the_nonlinear_model_at_large_amplitude():
    rng = np.random.default_rng(2)
    t = np.arange(0, 6, 1 / 40)
    case = make_case("pendulum", "P1", dict(theta0=1.0, omega0=0.0, g=G), {"L": 0.5}, t, 2e-3, rng)

    result = InProcessRunner().run(case.job(["P0", "P1"], plots=False))

    p0, p1 = result.model("P0"), result.model("P1")
    assert p1.splits[0].heldout_rmse < 0.5 * p0.splits[0].heldout_rmse
    g = param(p1, "g")
    assert abs(g.value - G) < 3 * g.se
    assert param(p0, "g").value < G - 3 * param(p0, "g").se  # small-angle g is biased low
    assert p0.full.train.systematic and not p1.full.train.systematic
    assert param(p1, "period_small_angle").value == pytest.approx(
        2 * np.pi * np.sqrt(0.5 / g.value)
    )


def test_bounce_splits_by_flight_and_predicts_the_next_one():
    rng = np.random.default_rng(1)
    t = np.arange(0, 3.0, 1 / 240)
    case = make_case(
        "bounce", "B0", dict(y0=1.0, v0=0.0, y_floor=0.0, g=G, e=0.8), {}, t, 5e-4, rng
    )

    result = InProcessRunner().run(case.job(["B0"], plots=False))

    model = result.model("B0")
    assert model.status == "ok", model.error
    e = param(model, "e")
    assert abs(e.value - 0.8) < 3 * e.se + 1e-3
    flights = len(set(case.segment_id.tolist()))
    train_flights = {round(s.split * flights) for s in model.splits}
    assert all(2 <= k <= flights - 1 for k in train_flights)
    assert model.splits[0].heldout_rmse < 5 * 5e-4
