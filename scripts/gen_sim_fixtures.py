"""Write the fixture trajectories the TypeScript simulator is tested against (build plan Part 18).

Each fixture runs one library model from `ghostline.physics.library` (the closed form where it
has one, else the ODE at 1e-12 tolerance) and samples it every `stride` steps of the simulator's
1/240 s grid. `web/src/lib/sim/sim.parity.test.ts` integrates the same model with the RK4
twin in `web/src/lib/sim/models` and checks that it agrees within 1e-6 of the signal's size.
Parameter and initial-condition names are the Python library's.

    uv run python scripts/gen_sim_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ghostline.physics.library import get_model
from ghostline.physics.library.base import ode_tolerance

OUT = Path(__file__).resolve().parents[1] / "web/src/lib/sim/__fixtures__/parity.json"
DT = 1 / 240
STRIDE = 24  # one fixture sample every 0.1 s

G = 9.81
# (model id, system, parameters and constants, initial conditions, duration in s)
CASES: list[tuple[str, str, dict[str, float], dict[str, float], float]] = [
    ("FF0", "free_fall", dict(g=G), dict(y0=2.0, vy0=1.5), 0.6),
    ("FF1", "free_fall", dict(g=G, b=1.7), dict(y0=2.0, vy0=1.5), 2.0),
    ("FF2", "free_fall", dict(g=G, k=0.4), dict(y0=5.0, vy0=0.0), 3.0),
    ("FF2", "free_fall", dict(g=G, k=0.4), dict(y0=1.0, vy0=3.0), 1.5),
    ("P0", "pendulum", dict(g=G, L=0.62), dict(theta0=0.2, omega0=0.3), 12.0),
    ("P1", "pendulum", dict(g=G, L=0.62), dict(theta0=0.66, omega0=0.0), 12.0),
    ("P2", "pendulum", dict(g=9.79, L=0.62, gamma=0.052), dict(theta0=0.66, omega0=0.0), 30.0),
    ("P3", "pendulum", dict(g=G, L=0.62, kappa=0.08), dict(theta0=0.9, omega0=0.0), 20.0),
    ("P4", "pendulum", dict(g=G, L=0.62, mu_p=0.12, eps=0.05), dict(theta0=0.7, omega0=0.0), 15.0),
    ("R0", "ramp", dict(g=G, alpha=0.4), dict(s0=0.0, v0=0.1), 1.5),
    ("R1", "ramp", dict(g=G, alpha=0.4, mu_k=0.2), dict(s0=0.0, v0=0.1), 1.5),
    ("R2", "ramp", dict(g=G, alpha=0.4, beta=0.4), dict(s0=0.0, v0=0.1), 1.5),
    ("R3", "ramp", dict(g=G, alpha=0.4, beta=0.4, c_rr=0.01), dict(s0=0.0, v0=0.1), 1.5),
    ("B0", "bounce", dict(g=G, e=0.8), dict(y0=1.0, v0=0.0), 6.0),
    ("B1", "bounce", dict(g=G, e0=0.9, e1=0.02), dict(y0=1.0, v0=0.0), 6.0),
    ("B2", "bounce", dict(g=G, e=0.8, k=0.05), dict(y0=1.0, v0=0.0), 6.0),
    ("S0", "spin_down", dict(b=0.06, direction=1.0), dict(phi0=0.5, omega0=40.0), 30.0),
    ("S1", "spin_down", dict(c=1.1, direction=-1.0), dict(phi0=0.5, omega0=40.0), 45.0),
    ("S2", "spin_down", dict(c=0.9, b=0.035, direction=1.0), dict(phi0=0.5, omega0=40.0), 45.0),
    ("S3", "spin_down", dict(c=0.9, k=0.004, direction=-1.0), dict(phi0=0.5, omega0=40.0), 45.0),
]


def fixture(model_id: str, system: str, params, init, duration: float) -> dict:
    model = get_model(model_id, system)
    steps = round(duration / DT)
    tau = np.arange(steps + 1) * DT
    # Defaults for the library's own bookkeeping parameters (a bounce's floor at 0, and so on).
    full = {name: spec.init for name, spec in model.params.items()} | params | init
    with ode_tolerance(1e-12, 1e-13):
        observed = model.predict(tau, full)[0]
    return {
        "model": model_id,
        "params": params,
        "init": init,
        "t_end": duration,
        "dt": DT,
        "stride": STRIDE,
        "obs": [float(f"{v:.13g}") for v in observed[::STRIDE]],
    }


def main() -> None:
    fixtures = [fixture(*case) for case in CASES]
    OUT.write_text(json.dumps(fixtures, separators=(",", ":")) + "\n")
    print(f"wrote {len(fixtures)} fixtures to {OUT.relative_to(OUT.parents[4])}")


if __name__ == "__main__":
    main()
