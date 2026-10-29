"""Generate parity fixtures for the TypeScript physics twins (build-plan Part 18).

Each case is integrated with scipy's DOP853 at tight tolerances, independently of the
TypeScript RK4 code, and sampled on the simulator's 1/240 s grid. `sim.parity.test.ts`
compares the TypeScript output against these within 1e-6 relative.

When the Python physics library (backend/ghostline/physics/library) lands, point this
script at it instead of the equations written out below, so that the fixtures come from the
same code the harness fits with. Run from web/:

    uv run --with numpy --with scipy python scripts/gen_sim_parity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

DT = 1 / 240
STRIDE = 24  # keep every 24th sample (10 Hz) to keep the fixture small
REST_SPEED = 0.02
RTOL, ATOL = 1e-12, 1e-14


def rhs_for(model: str, p: dict):
    g = p.get("g", 9.81)

    def pend(t, s):
        th, om = s
        a = -(g / p["L_eff"]) * np.sin(th)
        if model == "P2":
            a -= p["gamma"] * om
        elif model == "P3":
            a -= p["kappa"] * abs(om) * om
        elif model == "P4":
            a -= p["mu_p"] * np.tanh(om / p["eps"])
        return [om, a]

    def fall(t, s):
        y, v = s
        if model == "FF0":
            a = -g
        elif model == "FF1":
            a = -g - p["b_over_m"] * v
        elif model == "FF2":
            a = -g - p["k_over_m"] * abs(v) * v
        elif model == "B2":
            a = -g - p["k_over_m"] * abs(v) * v
        else:
            a = -g
        return [v, a]

    def ramp(t, s):
        sa, ca = np.sin(p["alpha"]), np.cos(p["alpha"])
        if model == "R0":
            a = g * sa
        elif model == "R1":
            a = g * (sa - p["mu_k"] * ca)
        elif model == "R2":
            a = g * sa / (1 + p["beta"])
        else:
            a = g * (sa - p["c_rr"] * ca) / (1 + p["beta"])
        return [s[1], a]

    def spin(t, s):
        w = s[1]
        if model == "S0":
            d = p["b"] * w
        elif model == "S1":
            d = p["c"]
        elif model == "S2":
            d = p["c"] + p["b"] * w
        else:
            d = p["c"] + p["k"] * w * w
        return [w, -d]

    return {"P": pend, "F": fall, "B": fall, "R": ramp, "S": spin}[model[0]]


def sample_grid(t_end: float) -> np.ndarray:
    n = int(round(t_end / DT))
    return np.arange(n + 1) * DT


def continuous_case(model, params, init, t_end, y0_names):
    f = rhs_for(model, params)
    s0 = [init[y0_names[0]], init[y0_names[1]]]
    grid = sample_grid(t_end)
    sol = solve_ivp(f, (0, t_end), s0, method="DOP853", t_eval=grid, rtol=RTOL, atol=ATOL)
    return grid, sol.y[0]


def stopping_case(model, params, init, t_end, names, direction_index):
    """Integrate until the state component `direction_index` crosses zero downward, then hold."""
    f = rhs_for(model, params)
    s0 = [init[names[0]], init[names[1]]]

    def stop(t, s):
        return s[direction_index]

    stop.terminal = True
    stop.direction = -1
    grid = sample_grid(t_end)
    sol = solve_ivp(f, (0, t_end), s0, method="DOP853", events=stop, dense_output=True, rtol=RTOL, atol=ATOL)
    t_stop = sol.t_events[0][0] if len(sol.t_events[0]) else None
    out = np.empty_like(grid)
    for i, t in enumerate(grid):
        if t_stop is not None and t >= t_stop:
            out[i] = sol.sol(t_stop)[0]
        else:
            out[i] = sol.sol(t)[0]
    return grid, out


def bounce_case(model, params, init, t_end):
    f = rhs_for(model, params)
    grid = sample_grid(t_end)
    out = np.empty_like(grid)
    t0, y, v = 0.0, init["y0"], init["v0"]
    segments = []
    stopped_at = None
    while t0 < t_end:
        def hit(t, s):
            return s[0]

        hit.terminal = True
        hit.direction = -1
        sol = solve_ivp(f, (t0, t_end), [y, v], method="DOP853", events=hit, dense_output=True, rtol=RTOL, atol=ATOL)
        t_hit = sol.t_events[0][0] if len(sol.t_events[0]) else None
        segments.append((t0, t_hit if t_hit is not None else t_end, sol))
        if t_hit is None:
            break
        v_in = abs(sol.sol(t_hit)[1])
        if model == "B1":
            e = max(0.0, params["e0"] - params["e1"] * v_in)
        else:
            e = params["e"]
        v_out = e * v_in
        t0, y, v = t_hit, 0.0, v_out
        if v_out < REST_SPEED:
            stopped_at = t_hit
            break
    for i, t in enumerate(grid):
        if stopped_at is not None and t >= stopped_at:
            out[i] = 0.0
            continue
        for a, b, sol in segments:
            if a <= t <= b:
                out[i] = sol.sol(t)[0]
                break
    return grid, out


CASES = [
    ("P1", {"g": 9.81, "L_eff": 0.62}, {"theta0": 0.66, "omega0": 0.0}, 12.0),
    ("P2", {"g": 9.79, "L_eff": 0.62, "gamma": 0.052}, {"theta0": 0.66, "omega0": 0.0}, 12.0),
    ("P3", {"g": 9.79, "L_eff": 0.62, "kappa": 0.04}, {"theta0": 0.9, "omega0": 0.1}, 12.0),
    ("P4", {"g": 9.79, "L_eff": 0.62, "mu_p": 0.05, "eps": 0.02}, {"theta0": 0.5, "omega0": 0.0}, 12.0),
    ("FF1", {"g": 9.81, "b_over_m": 0.9}, {"y0": 5.0, "v0": 1.0}, 1.2),
    ("FF2", {"g": 9.81, "k_over_m": 0.3}, {"y0": 8.0, "v0": 0.0}, 2.0),
    ("R1", {"g": 9.81, "alpha": 0.35, "mu_k": 0.18}, {"s0": 0.0, "v0": 0.4}, 1.5),
    ("R3", {"g": 9.81, "alpha": 0.35, "beta": 0.4, "c_rr": 0.01}, {"s0": 0.0, "v0": 0.0}, 1.5),
    ("R1", {"g": 9.81, "alpha": 0.2, "mu_k": 0.6}, {"s0": 0.0, "v0": 1.0}, 2.0),  # decelerates to rest
    ("B0", {"g": 9.81, "e": 0.82}, {"y0": 1.1, "v0": 0.0}, 4.0),
    ("B1", {"g": 9.81, "e0": 0.92, "e1": 0.02}, {"y0": 1.1, "v0": 0.0}, 4.0),
    ("B2", {"g": 9.81, "e": 0.8, "k_over_m": 0.2}, {"y0": 1.5, "v0": 0.0}, 4.0),
    ("S0", {"b": 0.06}, {"phi0": 0.0, "omega0": 55.0}, 30.0),
    ("S1", {"c": 1.1}, {"phi0": 0.0, "omega0": 55.0}, 60.0),
    ("S2", {"c": 0.9, "b": 0.035}, {"phi0": 0.0, "omega0": 55.0}, 40.0),
    ("S3", {"c": 0.9, "k": 0.004}, {"phi0": 0.0, "omega0": 55.0}, 40.0),
]


def run(model, params, init, t_end):
    kind = model[0]
    if kind == "P":
        return continuous_case(model, params, init, t_end, ["theta0", "omega0"])
    if model.startswith("FF"):
        return continuous_case(model, params, init, t_end, ["y0", "v0"])
    if kind == "R":
        return stopping_case(model, params, init, t_end, ["s0", "v0"], 1)
    if kind == "B":
        return bounce_case(model, params, init, t_end)
    return stopping_case(model, params, init, t_end, ["phi0", "omega0"], 1)


def main():
    out = []
    for model, params, init, t_end in CASES:
        grid, y = run(model, params, init, t_end)
        out.append(
            {
                "model": model,
                "params": params,
                "init": init,
                "t_end": t_end,
                "dt": DT,
                "stride": STRIDE,
                "obs": [float(f"{v:.13g}") for v in y[::STRIDE]],
            }
        )
    dest = Path(__file__).resolve().parent.parent / "src" / "lib" / "sim" / "__fixtures__" / "parity.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {len(out)} cases to {dest}")


if __name__ == "__main__":
    main()
