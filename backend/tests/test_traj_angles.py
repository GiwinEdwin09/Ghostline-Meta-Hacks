import numpy as np
import pytest

from ghostline.traj.angles import (
    effective_length,
    fit_circle,
    kasa_fit,
    pendulum_theta,
    spinner_phi,
)


def arc(centre, radius, span_deg, n=120, sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.radians(-90 + np.linspace(-span_deg / 2, span_deg / 2, n))
    pts = np.asarray(centre) + radius * np.column_stack([np.cos(a), np.sin(a)])
    return pts + rng.normal(0, sigma, pts.shape)


def test_kasa_fit_is_exact_without_noise():
    c, r = kasa_fit(arc((0.38, 0.3), 0.6, 60))
    assert np.allclose(c, [0.38, 0.3], atol=1e-9) and r == pytest.approx(0.6)


def test_geometric_fit_recovers_the_pivot_and_length_from_a_swing():
    fit = fit_circle(arc((0.38, 0.3), 0.6, 70, sigma=0.0015))
    assert fit.radius == pytest.approx(0.6, rel=0.01)
    assert np.allclose(fit.centre, [0.38, 0.3], atol=0.008)
    assert fit.rms == pytest.approx(0.0015, rel=0.3)
    assert fit.arc_deg == pytest.approx(70, abs=0.5)
    assert 0 < fit.sigma_radius < 0.01


def test_a_small_swing_leaves_the_pivot_poorly_constrained_and_says_so():
    wide = fit_circle(arc((0, 0), 0.6, 70, sigma=0.0015))
    narrow = fit_circle(arc((0, 0), 0.6, 8, sigma=0.0015))
    assert narrow.sigma_radius > 10 * wide.sigma_radius
    assert narrow.arc_deg == pytest.approx(8, abs=0.5)


def test_pendulum_angle_is_zero_hanging_down_and_positive_to_the_right():
    pivot = np.array([0.4, 0.3])
    theta = np.array([-0.5, 0.0, 0.3, 1.2])
    pos = pivot + 0.6 * np.column_stack([np.sin(theta), -np.cos(theta)])
    assert np.allclose(pendulum_theta(pos, pivot), theta)


def test_effective_length_of_a_spherical_bob():
    assert effective_length(0.6, 0.02) == pytest.approx(0.6 + 0.4 * 0.02**2 / 0.6)
    assert effective_length(0.5, 0.0) == 0.5


def circle_points(phi, hub=(0.1, -0.2), rho=0.035):
    return np.asarray(hub) + rho * np.column_stack([np.cos(phi), np.sin(phi)]), np.asarray(hub)


def test_spinner_angle_is_unwrapped_over_many_revolutions():
    phi = 0.5 + np.arange(200) * 0.6  # 34 degrees per frame, about 19 turns
    pos, hub = circle_points(phi)
    out = spinner_phi(pos, hub, np.arange(200))
    assert np.allclose(out.phi - out.phi[0], phi - phi[0], atol=1e-9)
    assert out.direction == 1 and out.direction_consistent and not out.alias_risk.any()
    assert out.max_step_deg == pytest.approx(np.degrees(0.6))


def test_clockwise_spin_is_reported_as_negative():
    pos, hub = circle_points(-np.arange(100) * 0.3)
    out = spinner_phi(pos, hub, np.arange(100))
    assert out.direction == -1 and out.phi[-1] < out.phi[0]


def test_steps_above_120_degrees_per_frame_are_flagged_as_aliasing_risk():
    phi = np.cumsum(np.r_[0.0, np.full(50, np.radians(100)), np.full(20, np.radians(130))])
    pos, hub = circle_points(phi)
    out = spinner_phi(pos, hub, np.arange(71))
    assert not out.alias_risk[:51].any() and out.alias_risk[51:].all()
    assert out.max_step_deg == pytest.approx(130)


def test_dropped_frames_do_not_look_like_a_faster_spin():
    phi = np.arange(0, 100, 2) * np.radians(70)  # every second frame: 140 degrees between samples
    pos, hub = circle_points(phi)
    out = spinner_phi(pos, hub, np.arange(0, 100, 2))
    assert out.max_step_deg == pytest.approx(70)
    assert not out.alias_risk.any()


def test_direction_flips_and_rest_are_recognised():
    phi = np.r_[np.arange(60) * 0.2, 12.0 - np.arange(40) * 0.2]
    pos, hub = circle_points(phi)
    out = spinner_phi(pos, hub, np.arange(100))
    assert not out.direction_consistent and out.n_reversals > 5
    rest = np.full(50, 1.0) + np.random.default_rng(0).normal(0, 1e-4, 50)
    pos, hub = circle_points(rest)
    still = spinner_phi(pos, hub, np.arange(50), sigma_phi=1e-4 / 0.035)
    assert still.direction == 0 and still.direction_consistent
