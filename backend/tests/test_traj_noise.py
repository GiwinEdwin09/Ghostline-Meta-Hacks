import numpy as np
import pytest

from ghostline.traj.noise import local_fit, noise_sigma, robust_std


def smooth(t):
    return 0.4 * np.sin(2 * np.pi * t / 1.4) + 0.1 * t**2


def test_robust_std_ignores_outliers_and_nans():
    x = np.random.default_rng(0).normal(0, 2.0, 5000)
    x[:50] = 1e3
    x[50:60] = np.nan
    assert robust_std(x) == pytest.approx(2.0, rel=0.05)
    assert np.isnan(robust_std(np.array([np.nan])))


@pytest.mark.parametrize("sigma", [0.0005, 0.002, 0.01])
def test_noise_estimate_is_within_20_percent_with_gaps_and_jittered_times(sigma):
    errors = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        t = np.sort(np.arange(240) / 60 + rng.uniform(-0.001, 0.001, 240))
        keep = rng.random(240) > 0.1  # dropped frames
        t = t[keep]
        y = smooth(t) + rng.normal(0, sigma, len(t))
        est = noise_sigma(local_fit(t, y))[0]
        errors.append(est / sigma - 1)
    assert np.abs(np.mean(errors)) < 0.1 and np.abs(errors).max() < 0.2


def test_both_channels_are_estimated_at_once():
    rng = np.random.default_rng(1)
    t = np.arange(300) / 60
    y = np.column_stack(
        [smooth(t) + rng.normal(0, 0.001, 300), 0.5 * t + rng.normal(0, 0.004, 300)]
    )
    sx, sy = noise_sigma(local_fit(t, y))
    assert sx == pytest.approx(0.001, rel=0.2) and sy == pytest.approx(0.004, rel=0.2)


def test_the_slope_matches_the_derivative_and_its_uncertainty_is_right():
    rng = np.random.default_rng(2)
    t = np.arange(300) / 60
    sigma = 0.002
    slopes = []
    for _ in range(40):
        y = smooth(t) + rng.normal(0, sigma, len(t))
        slopes.append(local_fit(t, y).slope[150])
    truth = 0.4 * 2 * np.pi / 1.4 * np.cos(2 * np.pi * t[150] / 1.4) + 0.2 * t[150]
    fit = local_fit(t, smooth(t) + rng.normal(0, sigma, len(t)))
    assert np.mean(slopes) == pytest.approx(truth, rel=0.02)
    assert np.std(slopes) == pytest.approx(sigma * fit.slope_factor[150], rel=0.35)


def test_a_kink_inflates_the_noise_unless_the_segments_are_separated():
    t = np.arange(240) / 60
    y = np.abs(t - 2.0) * 3.0 + np.random.default_rng(3).normal(0, 0.001, 240)  # V shape
    seg = (t >= 2.0).astype(int)
    plain = noise_sigma(local_fit(t, y))[0]
    split = noise_sigma(local_fit(t, y, seg))[0]
    assert split == pytest.approx(0.001, rel=0.25)
    assert plain >= split


def test_short_segments_are_left_invalid():
    t = np.arange(40) / 60
    seg = np.r_[np.zeros(36, int), np.ones(4, int)]
    fit = local_fit(t, t**2, seg)
    assert fit.valid[:36].all() and not fit.valid[36:].any()
    assert np.isnan(fit.value[36:]).all()


def test_exact_polynomials_have_no_residual():
    t = np.arange(60) / 30
    fit = local_fit(t, 1 + 2 * t - 3 * t**2 + t**3)
    assert np.abs(fit.resid).max() < 1e-9
    assert np.allclose(fit.slope, 2 - 6 * t + 3 * t**2, atol=1e-7)
