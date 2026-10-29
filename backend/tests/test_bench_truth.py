"""Benchmark ground truth (build plan Part 25): local gravity, friction, rolling, sigma algebra,
and the manifest."""

import math
import textwrap

import pytest
from bench.truth import (
    DEFAULT_MANIFEST,
    ManifestRow,
    beta_from_acceleration,
    combine_sigma,
    g_from_pendulum,
    gravity_truth,
    interval_contains,
    load_manifest,
    local_gravity,
    local_gravity_sigma,
    mu_k_from_tilts,
    pendulum_period_correction,
    resolve_truth,
    rolling_acceleration,
    rolling_truth,
    sigma_from_interval,
    wilson_interval,
    z_score,
)
from scipy.special import ellipk


# --- local gravity -------------------------------------------------------------------------
def test_gravity_matches_published_values():
    # IGF 1980 at the equator, and the polar value it implies (both are printed in the literature).
    assert local_gravity(0) == pytest.approx(9.780327, abs=1e-9)
    assert local_gravity(90) == pytest.approx(9.832186, abs=1e-6)
    # The conventional standard gravity, 9.80665, is the formula's value near 45.5 degrees.
    assert local_gravity(45.5) == pytest.approx(9.80665, abs=2e-4)


def test_gravity_falls_by_the_free_air_gradient_with_altitude():
    assert local_gravity(40, 0) - local_gravity(40, 1000) == pytest.approx(3.086e-3, rel=1e-9)
    assert local_gravity(40, 100) < local_gravity(40, 0)


def test_gravity_is_symmetric_between_hemispheres_and_grows_poleward():
    assert local_gravity(35) == pytest.approx(local_gravity(-35))
    assert local_gravity(10) < local_gravity(40) < local_gravity(70)


@pytest.mark.parametrize("bad", [-91, 90.5, 200])
def test_gravity_rejects_impossible_latitudes(bad):
    with pytest.raises(ValueError, match="latitude"):
        local_gravity(bad)


def test_gravity_sigma_is_dominated_by_anomalies_and_grows_with_altitude_doubt():
    assert local_gravity_sigma(0) == pytest.approx(0.002)
    assert local_gravity_sigma(10) == pytest.approx(0.002, abs=1e-4)
    assert local_gravity_sigma(5000) > local_gravity_sigma(10)


def test_gravity_truth_carries_a_method_string():
    truth = gravity_truth(40.0, 120.0)
    assert truth.value == pytest.approx(local_gravity(40.0, 120.0))
    assert "40.000" in truth.method and "120" in truth.method


# --- friction --------------------------------------------------------------------------------
def test_mu_k_is_the_mean_and_sample_sd_of_tan_theta():
    angles = [17.0, 17.4, 16.6, 17.2, 16.8]  # mean 17.0 degrees
    truth = mu_k_from_tilts(angles)
    values = [math.tan(math.radians(a)) for a in angles]
    mean = sum(values) / 5
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 4)
    assert truth.value == pytest.approx(mean)
    assert truth.sigma == pytest.approx(sd)
    assert truth.value == pytest.approx(0.3057, abs=1e-3)  # tan(17 degrees) is 0.3057
    assert "5 repeats" in truth.method


def test_mu_k_needs_repeats_and_sane_angles():
    with pytest.raises(ValueError, match="at least two"):
        mu_k_from_tilts([17.0])
    with pytest.raises(ValueError, match="between 0 and 90"):
        mu_k_from_tilts([17.0, 95.0])
    with pytest.raises(ValueError, match="between 0 and 90"):
        mu_k_from_tilts([0.0, 10.0])


# --- rolling -------------------------------------------------------------------------------
def test_rolling_ball_theory_is_five_sevenths_g_sin_alpha():
    g, alpha = 9.80665, 20.0
    accel = rolling_acceleration(g, alpha)
    assert accel == pytest.approx(5 / 7 * g * math.sin(math.radians(alpha)))
    assert rolling_truth().value == pytest.approx(0.4)
    assert rolling_truth().sigma == 0.0


def test_beta_round_trips_through_the_acceleration():
    for beta in (0.4, 0.5, 2 / 3):
        accel = rolling_acceleration(9.81, 15.0, beta)
        assert beta_from_acceleration(accel, 9.81, 15.0) == pytest.approx(beta)
    # A frictionless slide (beta = 0) is the upper bound on the acceleration.
    assert rolling_acceleration(9.81, 15.0, 0.0) > rolling_acceleration(9.81, 15.0)
    with pytest.raises(ValueError):
        beta_from_acceleration(0.0, 9.81, 15.0)


def test_other_rolling_shapes_have_their_textbook_beta():
    assert rolling_truth("hollow_sphere").value == pytest.approx(2 / 3)
    assert rolling_truth("solid_cylinder").value == pytest.approx(0.5)


# --- pendulum sub-checks ---------------------------------------------------------------------
def test_pendulum_cross_check_recovers_g_from_tape_and_stopwatch():
    g_true, length = 9.8066, 1.000
    period = 2 * math.pi * math.sqrt(length / g_true)
    truth = g_from_pendulum(length, 20 * period, 20)
    assert truth.value == pytest.approx(g_true, rel=1e-9)
    # +-1 mm on 1 m and +-0.2 s on 40 s: 0.5% from the stopwatch, doubled for T squared.
    assert truth.sigma / truth.value == pytest.approx(math.hypot(0.001, 2 * 0.2 / 40.13), rel=0.01)


def test_finite_amplitude_correction_lengthens_the_period():
    assert pendulum_period_correction(0) == 1.0
    # The series against the exact answer, T / T0 = 2 K(sin^2(theta0 / 2)) / pi.
    for amplitude in (10, 25, 40):
        exact = 2 * ellipk(math.sin(math.radians(amplitude) / 2) ** 2) / math.pi
        assert pendulum_period_correction(amplitude) == pytest.approx(exact, abs=1e-4)
    g_naive = g_from_pendulum(1.0, 40.0, 20, amplitude_deg=None).value
    g_fixed = g_from_pendulum(1.0, 40.0, 20, amplitude_deg=40).value
    assert g_fixed > g_naive


# --- sigma algebra -----------------------------------------------------------------------------
def test_combine_sigma_adds_in_quadrature_and_ignores_none():
    assert combine_sigma(3, 4) == pytest.approx(5)
    assert combine_sigma(3, None, 4) == pytest.approx(5)
    assert combine_sigma() == 0
    with pytest.raises(ValueError):
        combine_sigma(1, -2)


def test_sigma_from_a_95_percent_interval():
    assert sigma_from_interval(9.0, 11.0) == pytest.approx(1 / 1.959964, rel=1e-5)
    assert sigma_from_interval(9.0, 11.0, level=0.6827) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError):
        sigma_from_interval(11.0, 9.0)


def test_z_uses_the_combined_sigma():
    assert z_score(10.3, 10.0, 0.3, 0.4) == pytest.approx(0.3 / 0.5)
    assert z_score(9.0, 10.0, 0.5, None) == pytest.approx(-2.0)
    assert z_score(10.0, 10.0, 0.0, 0.0) is None


def test_interval_contains_is_inclusive():
    assert interval_contains(9.0, 11.0, 9.0)
    assert interval_contains(9.0, 11.0, 10.0)
    assert not interval_contains(9.0, 11.0, 11.01)


def test_wilson_interval_is_honest_at_small_n():
    low, high = wilson_interval(23, 25)
    assert 0.74 < low < 0.78 and 0.97 < high < 0.99
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(5, 5)[1] == pytest.approx(1.0)
    assert wilson_interval(5, 5)[0] < 0.6  # five for five still leaves a lot of doubt


# --- the manifest ------------------------------------------------------------------------------
def test_shipped_manifest_has_the_planned_clips():
    rows = load_manifest()
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row.kind, []).append(row)
    assert len(rows) == 25
    assert len(by_kind["scored"]) == 20 and len(by_kind["story"]) == 2
    assert len(by_kind["failure"]) == 3

    def count(**match):
        return sum(all(getattr(r, k) == v for k, v in match.items()) for r in rows)

    assert count(system="free_fall", expected_model="FF0") == 4
    assert count(system="free_fall", expected_model="FF1|FF2") == 2
    assert count(system="pendulum", kind="scored") == 5
    assert count(system="ramp", expected_model="R1") == 3
    assert count(system="ramp", expected_model="R2") == 2
    assert count(system="bounce") == 4
    assert count(system="spin_down") == 2
    assert all(r.failure_expected for r in by_kind["failure"])
    assert {r.failure_expected for r in by_kind["failure"]} == {
        "out_of_plane",
        "camera_motion",
        "time_base_uncertain",
    }


def test_shipped_manifest_needs_filming_so_truth_is_blank_except_theory():
    rows = load_manifest()
    assert all(r.truth is None for r in rows)
    resolved = {r.clip_id: resolve_truth(r) for r in rows}
    # With no location, tilt repeats or audio yet, only the rolling-ball theory is known.
    assert {k for k, v in resolved.items() if v is not None} == {"RA-R1", "RA-R2"}
    assert resolved["RA-R1"].value == pytest.approx(0.4)


def test_shipped_manifest_files_are_unique_and_relative():
    rows = load_manifest()
    files = [r.file for r in rows]
    assert len(set(files)) == len(files)
    assert all(f and not f.startswith("/") and ".." not in f for f in files)


def test_gravity_truth_comes_from_the_default_location_or_the_rows_own():
    rows = {r.clip_id: r for r in load_manifest()}
    row = rows["FF-D1"]
    assert resolve_truth(row) is None
    truth = resolve_truth(row, latitude=40.0, altitude_m=100.0)
    assert truth.value == pytest.approx(local_gravity(40.0, 100.0))
    own = row.model_copy(update={"latitude": 60.0, "altitude_m": 0.0})
    assert resolve_truth(own, latitude=40.0).value == pytest.approx(local_gravity(60.0))


def test_a_written_truth_wins_over_the_method(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(
        textwrap.dedent(
            """\
            clip_id,system,quantity,truth,truth_sigma,method,tilt_deg
            A,ramp,mu_k,0.31,0.02,tape,17;18
            B,ramp,mu_k,,,tilt,17.0;17.4;16.6;17.2;16.8
            C,ramp,mu_k,,,tilt,17.0
            """
        )
    )
    a, b, c = (resolve_truth(r) for r in load_manifest(path))
    assert (a.value, a.sigma, a.method) == (0.31, 0.02, "tape")
    assert b.value == pytest.approx(0.3057, abs=1e-3)
    assert c is None  # one repeat has no spread


def test_manifest_rejects_duplicate_ids_and_parses_blank_cells(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("clip_id,system,truth\nA,ramp,\nA,ramp,1\n")
    with pytest.raises(ValueError, match="unique"):
        load_manifest(path)
    row = ManifestRow.model_validate({"clip_id": "X", "system": "ramp", "truth": "", "file": ""})
    assert row.truth is None and row.file == "" and row.expected_models == []
    assert DEFAULT_MANIFEST.exists()


def test_capture_fps_decides_the_declared_mode():
    normal = ManifestRow(clip_id="a", system="ramp")
    sixty = ManifestRow(clip_id="b", system="ramp", capture_fps=60)
    slow = ManifestRow(clip_id="c", system="ramp", capture_fps=240)
    assert normal.declared_capture == {"mode": "normal"}
    assert sixty.declared_capture == {"mode": "normal"}
    assert slow.declared_capture == {"mode": "slow_mo", "fps": 240.0}
