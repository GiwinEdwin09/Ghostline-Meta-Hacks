import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import solve_ivp

from ghostline.physics.library import ParamSpec, catalog, get_model, model_ids, models_for

G = 9.81
ALPHA = 0.4

# (system, model id, parameters and constants, duration)
CLOSED_FORM_CASES = [
    ("free_fall", "FF0", dict(y0=1.5, vy0=0.0, g=G), 0.5),
    ("free_fall", "FF1", dict(y0=1.5, vy0=-1.0, g=G, b=0.8), 0.5),
    ("free_fall", "FF1", dict(y0=1.5, vy0=2.0, g=G, b=1e-9), 0.5),
    ("free_fall", "FF2", dict(y0=1.5, vy0=-0.5, g=G, k=0.9), 0.6),
    ("projectile", "FF0", dict(x0=0, y0=1.5, vx0=2.0, vy0=3.0, g=G, tilt=0.05), 0.5),
    ("projectile", "FF1", dict(x0=0, y0=1.5, vx0=2.0, vy0=3.0, g=G, tilt=0.05, b=0.6), 0.5),
    ("pendulum", "P0", dict(theta0=0.2, omega0=0.1, g=G, L=0.6), 6.0),
    ("ramp", "R0", dict(s0=0.0, v0=0.1, alpha=ALPHA, g=G), 1.0),
    ("ramp", "R1", dict(s0=0.0, v0=0.1, alpha=ALPHA, g=G, mu_k=0.1), 1.0),
    ("ramp", "R2", dict(s0=0.0, v0=0.1, alpha=ALPHA, g=G, beta=0.4), 1.0),
    ("ramp", "R3", dict(s0=0.0, v0=0.1, alpha=ALPHA, g=G, beta=0.4, c_rr=0.02), 1.0),
    ("spin_down", "S0", dict(phi0=0.2, omega0=12.0, direction=1.0, b=0.3), 7.0),
    ("spin_down", "S1", dict(phi0=0.2, omega0=12.0, direction=-1.0, c=1.5), 7.0),
    ("spin_down", "S2", dict(phi0=0.2, omega0=12.0, direction=1.0, b=0.2, c=1.0), 7.0),
    ("spin_down", "S3", dict(phi0=0.2, omega0=12.0, direction=1.0, c=1.0, k=0.02), 7.0),
    ("oscillator", "O0", dict(y0=0.1, v0=0.0, y_eq=0.0, k_over_m=100.0), 3.0),
    ("oscillator", "O1", dict(y0=0.1, v0=0.2, y_eq=0.0, k_over_m=100.0, gamma=0.5), 3.0),
]
ODE_ONLY_CASES = [
    ("pendulum", "P1", dict(theta0=0.7, omega0=0.1, g=G, L=0.6), 6.0),
    ("pendulum", "P2", dict(theta0=0.7, omega0=0.1, g=G, L=0.6, gamma=0.1), 6.0),
    ("pendulum", "P3", dict(theta0=0.7, omega0=0.1, g=G, L=0.6, kappa=0.1), 6.0),
    ("pendulum", "P4", dict(theta0=0.7, omega0=0.1, g=G, L=0.6, mu_p=0.1), 6.0),
    ("projectile", "FF2", dict(x0=0, y0=1.5, vx0=2.0, vy0=3.0, g=G, tilt=0.05, k=0.6), 0.5),
]


@pytest.mark.parametrize("system,model_id,p,duration", CLOSED_FORM_CASES)
def test_closed_form_matches_the_ode(system, model_id, p, duration):
    model = get_model(model_id, system)
    tau = np.linspace(0, duration, 200)

    closed = model.closed_form(tau, p)
    simulated = model.simulate(tau, p)

    assert closed.shape == (len(model.observables), tau.size)
    scale = max(float(np.ptp(closed)), 1e-9)
    assert np.max(np.abs(closed - simulated)) < 1e-6 * scale


@pytest.mark.parametrize("system,model_id,p,duration", ODE_ONLY_CASES)
def test_ode_models_predict_through_the_integrator(system, model_id, p, duration):
    model = get_model(model_id, system)
    tau = np.linspace(0, duration, 100)

    assert model.closed_form(tau, p) is None
    pred = model.predict(tau, p)

    assert pred.shape == (len(model.observables), tau.size)
    assert np.all(np.isfinite(pred))


def test_undamped_pendulum_conserves_energy_and_lengthens_with_amplitude():
    p1 = get_model("P1")
    tau = np.linspace(0, 20, 4000)
    theta = p1.predict(tau, dict(theta0=1.0, omega0=0.0, g=G, L=0.5))[0]
    assert np.max(np.abs(theta)) == pytest.approx(1.0, abs=1e-6)

    def first_return(amplitude):
        th = p1.predict(tau, dict(theta0=amplitude, omega0=0.0, g=G, L=0.5))[0]
        return tau[np.argmax(th < 0)] * 4

    small_angle_period = 2 * math.pi * math.sqrt(0.5 / G)
    assert first_return(0.05) == pytest.approx(small_angle_period, rel=2e-3)
    assert first_return(1.0) > 1.06 * small_angle_period


def test_bounce_heights_shrink_by_e_squared_and_b2_matches_b0_without_drag():
    b0, b2 = get_model("B0"), get_model("B2")
    p = dict(y0=1.0, v0=0.0, y_floor=0.0, g=G, e=0.8)
    tau = np.linspace(0, 2.5, 5000)
    y = b0.closed_form(tau, p)[0]

    peaks = [y[i] for i in range(1, y.size - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]
    assert peaks[0] == pytest.approx(0.8**2, rel=1e-3)
    assert peaks[1] / peaks[0] == pytest.approx(0.8**2, rel=1e-3)
    assert np.allclose(b2.closed_form(tau, {**p, "k": 0.0}), b0.closed_form(tau, p))


def test_bounce_with_drag_matches_a_numerical_reference():
    b2 = get_model("B2")
    p = dict(y0=1.0, v0=0.0, y_floor=0.0, g=G, e=0.8, k=0.3)
    tau = np.linspace(0, 2.0, 300)
    y = b2.closed_form(tau, p)[0]

    reference = np.zeros_like(tau)
    t0, state = 0.0, [1.0, 0.0]
    for _ in range(20):
        hit = lambda t, s: s[0]  # noqa: E731
        hit.terminal, hit.direction = True, -1
        sol = solve_ivp(
            lambda t, s: [s[1], -G - 0.3 * abs(s[1]) * s[1]],
            (t0, 2.0), state, events=hit, dense_output=True, rtol=1e-11, atol=1e-12,
        )  # fmt: skip
        inside = (tau >= t0) & (tau <= sol.t[-1])
        reference[inside] = sol.sol(tau[inside])[0]
        if sol.status != 1:
            break
        state, t0 = [0.0, 0.8 * -sol.y_events[0][0][1]], sol.t[-1]

    assert np.max(np.abs(y - reference)) < 1e-6


def test_spin_down_clamps_at_zero_and_keeps_its_direction():
    s1 = get_model("S1")
    p = dict(phi0=0.0, omega0=10.0, c=2.0, direction=-1.0)
    tau = np.linspace(0, 10, 100)

    phi = s1.predict(tau, p)[0]

    assert phi[-1] == pytest.approx(-25.0)  # stops at t = 5 after turning 25 rad
    assert phi[tau > 5][0] == pytest.approx(-25.0)
    assert np.all(np.diff(phi) <= 1e-12)


def test_derived_quantities():
    ff2 = get_model("FF2")
    assert ff2.derived(dict(g=G, k=0.9))["terminal_velocity"] == pytest.approx(math.sqrt(G / 0.9))
    pend = get_model("P1")
    out = pend.derived(dict(g=G, L=0.5, L_tape=0.52))
    assert out["g_tape"] == pytest.approx(G * 0.52 / 0.5)
    assert out["period_small_angle"] == pytest.approx(2 * math.pi * math.sqrt(0.5 / G))
    assert get_model("S1").derived(dict(omega0=30.0, c=3.0))["time_to_stop"] == pytest.approx(10.0)


def test_registry_and_catalog():
    assert model_ids("pendulum") == ["P0", "P1", "P2", "P3", "P4"]
    assert [m.id for m in models_for("ramp", "block")] == ["R0", "R1"]
    assert [m.id for m in models_for("ramp", "ball")] == ["R0", "R2", "R3"]
    assert get_model("FF1", "projectile").observables == ("x", "y")
    assert get_model("FF1", "free_fall").observables == ("y",)
    with pytest.raises(KeyError):
        get_model("XX9")
    entries = catalog("spin_down")
    assert [e["id"] for e in entries] == ["S0", "S1", "S2", "S3"]
    for entry in entries:
        assert entry["mechanism"] and entry["equations_latex"] and entry["signature"]


def test_every_model_has_documented_parameters_inside_their_bounds():
    for system in (
        "free_fall",
        "projectile",
        "pendulum",
        "ramp",
        "bounce",
        "spin_down",
        "oscillator",
    ):
        for model in models_for(system):
            assert model.mechanism and model.equations_latex and model.signature
            assert model.params, model.id
            for name, spec in model.params.items():
                assert spec.lower <= spec.init <= spec.upper, (model.id, name)
                assert spec.kind in ("physical", "initial_condition", "nuisance")


def test_param_spec_rejects_an_init_outside_its_bounds():
    with pytest.raises(ValidationError):
        ParamSpec(init=5.0, lower=0.0, upper=1.0, unit="m")


@pytest.mark.parametrize(
    "system,model_id,truth,constants",
    [
        ("free_fall", "FF2", dict(y0=1.5, vy0=0.0, g=G, k=0.9), {}),
        ("pendulum", "P0", dict(theta0=0.3, omega0=0.0, g=G), {"L": 0.6}),
        ("ramp", "R2", dict(s0=0.0, v0=0.0, beta=0.4), {"alpha": ALPHA, "g": G}),
        ("spin_down", "S2", dict(phi0=0.0, omega0=30.0, b=0.2, c=1.0), {}),
        ("oscillator", "O1", dict(y0=0.05, v0=0.0, y_eq=0.0, k_over_m=90.0, gamma=0.3), {}),
    ],
)
def test_initial_guess_heuristics_land_near_the_truth(system, model_id, truth, constants):
    model = get_model(model_id, system)
    tau = np.linspace(0, 6.0 if system != "free_fall" else 0.6, 240)
    p = {**constants, **truth}
    clean = model.predict(tau, p)
    obs = {o: clean[i] for i, o in enumerate(model.observables)}
    constants = model.prepare(tau, obs, constants)

    specs = model.fit_setup(tau, obs, constants)

    for spec in specs.values():
        assert spec.lower <= spec.init <= spec.upper
    physical = {n: s for n, s in specs.items() if s.kind == "physical" and n in truth}
    for name, spec in physical.items():
        if name in ("b", "gamma", "c"):  # decay constants only need the right order of magnitude
            assert 0.1 * truth[name] < spec.init < 10 * truth[name], name
        else:
            assert spec.init == pytest.approx(truth[name], rel=0.35), name
