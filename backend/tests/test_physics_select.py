import numpy as np
import pytest

from ghostline.physics.analysis import analyze
from ghostline.physics.harness.schema import (
    BootstrapSpec,
    Diagnostics,
    FitResult,
    ModelResult,
    ParamEstimate,
    SplitFit,
)
from ghostline.physics.recovery_suite import BUDGET, inject_calibration_error
from ghostline.physics.runner import InProcessRunner
from ghostline.physics.select import (
    CalibrationBudget,
    calibration_perturbations,
    comparison_table,
    final_quantities,
    select_models,
)
from ghostline.physics.synthetic import make_case

G = 9.81
SPLITS = (0.65, 0.5, 0.8)
N_TEST = 80


def diag():
    return Diagnostics(n=100, n_params=3, rmse=0.001, chi2=100.0, chi2_red=1.0)


def fake_model(model_id, n_physical, rmse_by_split, rng, draw_spread=0.0, ref=None):
    """A ModelResult whose held-out errors have a chosen RMSE on each split."""
    splits = []
    for split in SPLITS:
        base = rmse_by_split[split] if isinstance(rmse_by_split, dict) else rmse_by_split
        t = np.linspace(0.01, 1.0, N_TEST)
        shared = np.random.default_rng(int(split * 100)).normal(size=N_TEST)  # same for all models
        own = rng.normal(size=N_TEST)  # different for each model
        errors = np.sqrt((0.7 * shared) ** 2 + (0.7 * own) ** 2)
        errors *= base / np.sqrt(np.mean(errors**2))
        draws = (
            [(errors * (1 + draw_spread * rng.normal())).tolist() for _ in range(24)]
            if draw_spread
            else None
        )
        splits.append(
            SplitFit(
                split=split, t_split=0.0, n_train=100, n_test=N_TEST, converged=True, params=[],
                train=diag(), heldout_rmse=float(np.sqrt(np.mean(errors**2))),
                heldout_errors=errors.tolist(), heldout_draws=draws, t_test=t.tolist(),
            )
        )  # fmt: skip
    return ModelResult(
        id=model_id, status="ok", library_ref=ref, n_physical=n_physical, splits=splits,
        full=splits[0].model_copy(update={"split": 1.0}), aic=1.0, bic=1.0,
    )  # fmt: skip


def result_of(*models):
    return FitResult(job_id="j", mode="select", system="free_fall", models=list(models))


def test_clear_winner_is_decided_in_every_split():
    rng = np.random.default_rng(0)
    result = result_of(
        fake_model("A", 1, 0.010, rng),
        fake_model("B", 2, 0.004, rng),
        fake_model("C", 1, 0.020, rng),
    )

    sel = select_models(result, seed=1)

    assert (sel.status, sel.selected_id, sel.splits_won, sel.n_splits) == ("decided", "B", 3, 3)
    assert sel.tied_with == [] and not sel.parsimony
    assert sel.score("B").heldout_rmse["0.65"] == pytest.approx(0.004)


def test_a_statistical_tie_goes_to_the_model_with_fewer_parameters():
    rng = np.random.default_rng(0)
    # B has the lowest RMSE by a hair, but A is 0.5% behind and has fewer parameters.
    result = result_of(fake_model("A", 1, 0.00402, rng), fake_model("B", 3, 0.00400, rng))

    sel = select_models(result, seed=1)

    assert sel.status == "tied" and sel.selected_id == "A" and sel.parsimony
    assert sel.tied_with == ["B"] and sel.tied_equal_or_simpler == []
    assert "can't distinguish" in sel.message and "fewer parameters" in sel.message


def test_a_tie_between_equally_complex_models_is_a_genuine_can_not_tell():
    rng = np.random.default_rng(0)
    result = result_of(fake_model("A", 2, 0.00402, rng), fake_model("B", 2, 0.00400, rng))

    sel = select_models(result, seed=1)

    other = ({"A", "B"} - {sel.selected_id}).pop()
    assert sel.status == "tied" and sel.tied_equal_or_simpler == [other]


def test_the_tie_rank_prefers_the_nonlinear_pendulum_over_its_small_angle_twin():
    rng = np.random.default_rng(0)
    result = result_of(
        fake_model("P0", 1, 0.00400, rng, ref="P0"), fake_model("P1", 1, 0.00402, rng, ref="P1")
    ).model_copy(update={"system": "pendulum"})

    assert select_models(result, seed=1).selected_id == "P1"


def test_splits_that_disagree_make_the_result_inconclusive():
    rng = np.random.default_rng(0)
    a = {0.65: 0.004, 0.5: 0.012, 0.8: 0.012}
    b = {0.65: 0.012, 0.5: 0.004, 0.8: 0.004}
    result = result_of(fake_model("A", 1, a, rng), fake_model("B", 1, b, rng))

    sel = select_models(result, seed=1)

    assert sel.status == "inconclusive"
    assert [o.selected for o in sel.splits].count("loses") >= 2


def test_parameter_uncertainty_keeps_noisy_forecasts_from_looking_like_wins():
    rng = np.random.default_rng(3)
    # Same typical error, but B's forecast swings widely with its parameters.
    result = result_of(
        fake_model("A", 1, 0.0040, rng, draw_spread=0.02),
        fake_model("B", 2, 0.0036, rng, draw_spread=0.6),
    )

    sel = select_models(result, seed=1)

    assert sel.selected_id == "A" and sel.status != "decided" or sel.splits_won < 3


def test_models_that_failed_are_reported_but_not_scored():
    rng = np.random.default_rng(0)
    broken = ModelResult(id="X", status="failed", error="boom")
    sel = select_models(result_of(fake_model("A", 1, 0.004, rng), broken), seed=1)
    assert sel.status == "inconclusive" and sel.selected_id == "A"
    assert sel.score("X").error == "boom"
    assert select_models(result_of(broken), seed=1).selected_id is None


def test_comparison_table_hides_heldout_arrays_and_held_out_plots():
    rng = np.random.default_rng(0)
    model = fake_model("A", 1, 0.004, rng)
    model.splits[0].params = [
        ParamEstimate(name="g", value=9.8, ci_low=9.7, ci_high=9.9, unit="m/s^2"),
        ParamEstimate(name="y0", value=1.0, kind="initial_condition"),
    ]
    sel = select_models(result_of(model, fake_model("B", 2, 0.010, rng)), seed=1)

    row = comparison_table(sel)[0]

    assert set(row) >= {
        "model_id",
        "parameters",
        "heldout_rmse",
        "heldout_rmse_ci95",
        "diagnostics",
    }
    assert [p["name"] for p in row["parameters"]] == ["g"]
    assert "heldout_errors" not in str(row) and "t_test" not in str(row)
    assert row["heldout_rmse"] == pytest.approx(0.004)


# ----------------------------------------------------------------------------------------
# On simulated data


def test_foam_ball_is_decided_for_quadratic_drag_and_dense_ball_falls_back_to_constant_g():
    rng = np.random.default_rng(4)
    foam = make_case(
        "free_fall",
        "FF2",
        dict(y0=2.0, vy0=0.0, g=G, k=0.9),
        {},
        np.arange(0, 0.65, 1 / 240),
        5e-4,
        rng,
    )
    dense = make_case(
        "free_fall", "FF0", dict(y0=1.2, vy0=0.0, g=G), {}, np.arange(0, 0.5, 1 / 240), 5e-4, rng
    )

    foam_out = analyze(
        foam.t,
        foam.observed,
        foam.sigma,
        system="free_fall",
        review="accepted",
        n_boot=40,
        n_perturb=10,
    )
    dense_out = analyze(
        dense.t,
        dense.observed,
        dense.sigma,
        system="free_fall",
        review="accepted",
        n_boot=40,
        n_perturb=10,
    )

    assert foam_out.selection.status == "decided" and foam_out.selection.selected_id == "FF2"
    assert foam_out.confidence.level == "high"
    assert dense_out.selection.selected_id == "FF0"
    assert dense_out.selection.status in ("tied", "decided")
    k = next(q for q in foam_out.quantities if q.name == "k")
    assert k.ci_low < 0.9 < k.ci_high


def test_spin_down_rejects_exponential_decay_with_held_out_numbers():
    rng = np.random.default_rng(5)
    case = make_case(
        "spin_down",
        "S1",
        dict(phi0=0.0, omega0=40.0, c=3.0),
        {},
        np.arange(0, 15, 1 / 60),
        0.03,
        rng,
    )

    out = analyze(
        case.t,
        case.observed,
        case.sigma,
        system="spin_down",
        review="accepted",
        n_boot=40,
        n_perturb=10,
    )

    assert out.selection.selected_id == "S1"
    s0 = out.selection.score("S0")
    assert s0.heldout_rmse["0.65"] > 100 * out.selection.score("S1").heldout_rmse["0.65"]
    stop = next(q for q in out.quantities if q.name == "time_to_stop")
    assert stop.ci_low < 40 / 3 < stop.ci_high


def test_ramp_angle_dominates_the_friction_budget():
    rng = np.random.default_rng(6)
    consts = {"alpha": 0.45, "g": 9.80665}
    case = make_case(
        "ramp", "R1", dict(s0=0.0, v0=0.0, mu_k=0.3), consts, np.arange(0, 1.2, 1 / 60), 1e-3, rng
    )

    out = analyze(
        case.t, case.observed, case.sigma, system="ramp", constants=consts,
        library_ids=["R0", "R1"],
        budget=BUDGET, review="accepted", n_boot=40, n_perturb=30,
    )  # fmt: skip

    mu = next(q for q in out.quantities if q.name == "mu_k")
    assert out.selection.selected_id == "R1"
    assert mu.budget["angle"] > 10 * mu.budget["scale"]
    assert mu.budget["angle"] > 5 * mu.budget["statistical"]
    assert mu.budget["angle"] == pytest.approx(0.005 / 0.9 * 1.0, rel=0.5)  # d mu / d alpha = sec^2


# ----------------------------------------------------------------------------------------
# Uncertainty budget


def test_calibration_draws_are_antithetic_and_carry_their_components():
    budget = CalibrationBudget(
        sigma_scale=0.01, sigma_offset=0.02, sigma_angle=0.01, sigma_timing=0.005
    )

    draws = calibration_perturbations(budget, {"alpha": 0.4}, n=50, seed=2)

    assert len(draws) == 50
    for d in draws:
        assert d.scale == pytest.approx((1 + d.eps["scale"]) * (1 + d.eps["offset"]))
        assert d.time_scale == pytest.approx(1 + d.eps["timing"])
        assert d.constants["alpha"] == pytest.approx(0.4 + d.eps["angle"])
    assert abs(np.mean([d.eps["scale"] for d in draws])) < 1e-12
    assert np.std([d.eps["scale"] for d in draws]) == pytest.approx(0.01, rel=0.3)
    assert calibration_perturbations(budget, {}, n=4)[0].constants == {}


def test_budget_attributes_the_spread_to_the_right_calibration_input():
    rng = np.random.default_rng(0)
    budget = CalibrationBudget(
        sigma_scale=0.01, sigma_offset=0.0, sigma_angle=0.0, sigma_timing=0.004
    )
    draws = calibration_perturbations(budget, {}, n=60, seed=1)
    g0 = 9.81
    reps = [{"g": g0 * d.scale / d.time_scale**2 + rng.normal(0, 1e-4)} for d in draws]
    est = ParamEstimate(name="g", value=g0, ci_low=g0 - 0.02, ci_high=g0 + 0.02, unit="m/s^2")
    final = ModelResult(
        id="FF0", status="ok", n_physical=1, replicates=reps,
        replicate_eps=[d.eps for d in draws], full=SplitFit(
            split=1.0, t_split=0, n_train=1, n_test=0, converged=True, params=[est], train=diag()
        ),
    )  # fmt: skip

    (q,) = final_quantities(final, budget)

    assert q.budget["scale"] == pytest.approx(0.0981, rel=0.1)
    assert q.budget["timing"] == pytest.approx(2 * 0.004 * 9.81, rel=0.1)
    assert q.budget["offset"] == 0.0 and q.budget["angle"] == 0.0
    assert q.budget["other"] < 0.01
    assert q.budget["statistical"] == pytest.approx(0.02 / 1.96, rel=0.01)
    assert q.sigma_total == pytest.approx(np.sqrt(sum(v**2 for v in q.budget.values())))
    assert (q.ci_high - q.ci_low) / 2 == pytest.approx(1.959964 * q.sigma_total, rel=1e-6)


def test_model_choice_term_is_the_spread_across_tied_models():
    def model(model_id, g):
        est = ParamEstimate(name="g", value=g, ci_low=g - 0.1, ci_high=g + 0.1)
        fit = SplitFit(
            split=1.0, t_split=0, n_train=1, n_test=0, converged=True, params=[est], train=diag()
        )
        return ModelResult(id=model_id, status="ok", full=fit)

    (q,) = final_quantities(
        model("A", 9.8), CalibrationBudget(), [model("B", 9.6), model("C", 9.9)]
    )

    assert q.budget["model_choice"] == pytest.approx(0.15)


def test_ninety_five_percent_intervals_cover_the_truth_about_95_percent_of_the_time():
    runner = InProcessRunner()
    hits, runs = 0, 100
    for i in range(runs):
        rng = np.random.default_rng(900 + i)
        case = make_case(
            "free_fall",
            "FF0",
            dict(y0=1.2, vy0=0.0, g=G),
            {},
            np.arange(0, 0.5, 1 / 240),
            5e-4,
            rng,
        )
        t, observed, _ = inject_calibration_error(case, rng)
        data_case = case.job(["FF0"], mode="final", plots=False)
        data_case.data.t = t.tolist()
        data_case.data.series = {"y": observed["y"].tolist()}
        data_case.bootstrap = BootstrapSpec(n=0)
        data_case.perturbations = calibration_perturbations(BUDGET, {}, n=16, seed=i)
        final = runner.run(data_case).model("FF0")
        (g,) = [q for q in final_quantities(final, BUDGET) if q.name == "g"]
        hits += g.ci_low <= G <= g.ci_high

    assert 0.88 <= hits / runs <= 1.0


def test_incomplete_calibration_refits_are_refused_instead_of_understating_the_budget():
    est = ParamEstimate(name="g", value=9.8, ci_low=9.7, ci_high=9.9)
    fit = SplitFit(
        split=1.0, t_split=0, n_train=1, n_test=0, converged=True, params=[est], train=diag()
    )
    cut_short = ModelResult(
        id="P1", status="ok", full=fit, perturbations_requested=50, replicates=[{"g": 9.8}] * 3
    )

    with pytest.raises(ValueError, match="calibration refits"):
        final_quantities(cut_short, CalibrationBudget(sigma_scale=0.01))
    # With no calibration uncertainty to estimate there is nothing to understate.
    assert final_quantities(cut_short, CalibrationBudget())[0].name == "g"
