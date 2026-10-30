"""Measurements on a simulated trajectory, in plain Python (build plan Part 19).

These functions have no imports. The PyBullet snippet embeds their source so that it stands
alone. Each takes lists of floats (sample times and values).
"""


def rebound_heights(z):
    """Height of each rebound: its apex above the lowest point reached just before it.

    Measured from the impact point, not from the floor, because that is what the speed after
    the impact sets (v^2 / 2g). A simulator that turns the ball around a little above the
    floor, as Bullet's speculative contacts do, still gives successive heights in the ratio e^2.
    """
    heights, low = [], z[0]
    for i in range(1, len(z) - 1):
        low = min(low, z[i])
        if z[i - 1] < z[i] >= z[i + 1]:
            heights.append(z[i] - low)
            low = z[i]
    return heights


def geometric_ratio(heights, n=3):
    """Mean ratio of successive heights, from the first n: (last / first) ** (1 / steps)."""
    use = [h for h in heights[:n] if h > 0]
    if len(use) < 2:
        return None
    return (use[-1] / use[0]) ** (1.0 / (len(use) - 1))


def mean_period(t, y):
    """Mean time between upward zero crossings of y, with linear interpolation."""
    crossings = []
    for i in range(1, len(y)):
        if y[i - 1] < 0 <= y[i]:
            crossings.append(t[i - 1] + (t[i] - t[i - 1]) * (-y[i - 1]) / (y[i] - y[i - 1]))
    if len(crossings) < 2:
        return None
    return (crossings[-1] - crossings[0]) / (len(crossings) - 1)


def half_life(t, y):
    """Time for the swing's peak |y| to fall to half of the release amplitude |y[0]|."""
    peaks = [(t[0], abs(y[0]))]
    for i in range(1, len(y) - 1):
        if abs(y[i]) >= abs(y[i - 1]) and abs(y[i]) > abs(y[i + 1]):
            peaks.append((t[i], abs(y[i])))
    target = peaks[0][1] / 2.0
    for k in range(len(peaks) - 1):
        (t0, a0), (t1, a1) = peaks[k], peaks[k + 1]
        if a1 <= target < a0:
            return t0 + (t1 - t0) * (a0 - target) / (a0 - a1)
    return None
