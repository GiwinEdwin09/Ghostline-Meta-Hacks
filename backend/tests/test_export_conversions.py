"""The unit conversions behind the simulator export (build plan Part 19) and Bullet's damping
formula."""

import math

import pytest

from ghostline.export import conversions as c


def test_bullet_damping_makes_the_speed_decay_at_the_measured_rate():
    for rate in (0.05, 0.8, 3.0):
        d = c.bullet_linear_damping(rate)

        assert d == pytest.approx(1 - math.exp(-rate), rel=1e-12)
        # Bullet scales the velocity by (1 - d) ** dt every step: after one second, by e^-rate.
        dt = 1 / 240
        speed = 1.0
        for _ in range(240):
            speed *= (1 - d) ** dt
        assert speed == pytest.approx(math.exp(-rate), rel=1e-9)
        assert c.decay_rate_from_bullet_damping(d) == pytest.approx(rate, rel=1e-9)


def test_bullet_damping_edge_cases():
    assert c.bullet_linear_damping(0.0) == 0.0
    assert c.bullet_linear_damping(20.0) < 1.0
    with pytest.raises(ValueError):
        c.bullet_linear_damping(-0.1)
    with pytest.raises(ValueError):
        c.decay_rate_from_bullet_damping(1.0)
    # The default damping of 0.04 is a decay of about 4% per second, not nothing.
    assert c.decay_rate_from_bullet_damping(c.BULLET_DEFAULT_DAMPING) == pytest.approx(
        0.0408, abs=1e-4
    )


def test_rolling_friction_is_the_coefficient_times_the_ball_radius():
    assert c.bullet_rolling_friction(0.012, 0.02) == pytest.approx(0.00024)
    assert c.bullet_rolling_friction(0.0, 0.5) == 0.0


def test_pendulum_inertia_is_m_times_l_squared_plus_two_fifths_r_squared():
    m, length, r = 0.05, 0.62, 0.012
    expected = m * (length**2 + 0.4 * r**2)

    assert c.pendulum_inertia(m, length, r) == pytest.approx(expected, rel=1e-12)
    assert c.pendulum_inertia(m, length, 0.0) == pytest.approx(m * length**2)  # a point mass
    assert c.pendulum_inertia(2 * m, length, r) == pytest.approx(2 * expected)  # linear in mass


def test_joint_damping_is_the_rate_times_the_inertia():
    inertia = c.pendulum_inertia(0.05, 0.62, 0.012)

    assert c.joint_damping(0.12, inertia) == pytest.approx(0.12 * inertia)
    # The torque it gives per unit of inertia is the measured rate times the angular speed.
    omega = 2.0
    assert c.joint_damping(0.12, inertia) * omega / inertia == pytest.approx(0.12 * omega)


def test_coulomb_motor_force_is_the_deceleration_times_the_inertia():
    inertia = 4.2e-3

    assert c.coulomb_motor_force(0.25, inertia) == pytest.approx(0.25 * inertia)
    assert c.quadratic_drag_coefficient(0.35, 0.1) == pytest.approx(0.035)


def test_restitution_at_speed_and_its_interval():
    assert c.restitution_at_speed(0.9, 0.02, 3.0) == pytest.approx(0.84)
    assert c.restitution_at_speed(0.9, 0.5, 3.0) == 0.0  # clipped to a physical value
    assert c.restitution_at_speed(1.2, 0.0, 1.0) == 1.0

    value, low, high = c.restitution_interval_at_speed((0.9, 0.88, 0.92), (0.02, 0.015, 0.025), 3.0)

    assert value == pytest.approx(0.84)
    assert low < value < high
    # A wider e1 interval widens the result at a higher speed.
    wide = c.restitution_interval_at_speed((0.9, 0.88, 0.92), (0.02, 0.015, 0.025), 6.0)
    assert (wide[2] - wide[1]) > (high - low)


def test_bounce_heights_shrink_by_e_squared():
    assert c.bounce_height_ratio(0.8) == pytest.approx(0.64)
