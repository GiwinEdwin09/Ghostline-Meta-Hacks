"""Part 10 and Part 12 "done when" checks on synthetic data with known truth.

    uv run python -m ghostline.physics.recovery_suite --n 50            # recovery + selection
    uv run python -m ghostline.physics.recovery_suite --coverage 100    # interval coverage

For each scenario it simulates N noisy datasets from one library model and reports
  - parameter recovery: how often the fitted truth model lands within 2 sigma of the truth, and
  - selection: how often the held-out comparison picks the truth (or a model that the data
    cannot tell apart from it), and how often it reports a tie or an inconclusive result.
Coverage injects calibration and timing errors into the data and checks that the 95% intervals
from `analyze` contain the true value.
"""

from __future__ import annotations

import argparse
import multiprocessing
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .analysis import analyze
from .runner import InProcessRunner
from .select import CalibrationBudget, select_models
from .synthetic import SyntheticCase, make_case

G = 9.81


@dataclass
class Scenario:
    name: str
    system: str
    truth_id: str
    params: dict[str, float]
    t: np.ndarray
    sigma: float
    candidates: list[str]
    constants: dict[str, float] = field(default_factory=dict)
    accept: tuple[str, ...] = ()
    note: str = ""

    def accepted(self) -> set[str]:
        return set(self.accept) | {self.truth_id}

    def case(self, rng: np.random.Generator) -> SyntheticCase:
        return make_case(
            self.system, self.truth_id, self.params, self.constants, self.t, self.sigma, rng
        )


def _t(duration: float, fps: float) -> np.ndarray:
    return np.arange(0.0, duration, 1.0 / fps)


def _scn(name, system, truth, params, duration, fps, sigma, candidates, constants=None, **kw):
    t = _t(duration, fps)
    return Scenario(name, system, truth, params, t, sigma, candidates, constants or {}, **kw)


def scenarios() -> list[Scenario]:
    ff = ["FF0", "FF1", "FF2"]
    pend = ["P0", "P1", "P2", "P3", "P4"]
    spin = ["S0", "S1", "S2", "S3"]
    bounce = ["B0", "B1", "B2"]
    length = {"L": 0.6}
    ramp = {"alpha": 0.45, "g": 9.80665}
    drop = dict(y0=1.2, vy0=0.0, g=G)
    swing = dict(omega0=0.0, g=G)
    bb = dict(y0=1.0, v0=0.0, y_floor=0.0, g=G)
    still = dict(s0=0.0, v0=0.0)
    return [
        _scn("free_fall dense ball", "free_fall", "FF0", drop, 0.5, 240, 5e-4, ff),
        _scn(
            "free_fall linear drag",
            "free_fall",
            "FF1",
            dict(drop, y0=2.0, b=1.5),
            0.7,
            240,
            5e-4,
            ff,
        ),
        _scn(
            "free_fall foam ball",
            "free_fall",
            "FF2",
            dict(drop, y0=2.0, k=0.9),
            0.65,
            240,
            5e-4,
            ff,
        ),
        _scn(
            "pendulum 10 deg",
            "pendulum",
            "P0",
            dict(swing, theta0=0.17),
            10,
            60,
            2e-3,
            pend,
            length,
            accept=("P1",),
            note="P0 and P1 are the same physics at small angle",
        ),
        _scn(
            "pendulum 40 deg", "pendulum", "P1", dict(swing, theta0=0.7), 10, 60, 2e-3, pend, length
        ),
        _scn(
            "pendulum viscous",
            "pendulum",
            "P2",
            dict(swing, theta0=0.7, gamma=0.08),
            10,
            60,
            2e-3,
            pend,
            length,
        ),
        _scn(
            "pendulum quadratic drag",
            "pendulum",
            "P3",
            dict(swing, theta0=0.9, kappa=0.15),
            10,
            60,
            2e-3,
            pend,
            length,
        ),
        _scn(
            "pendulum pivot friction",
            "pendulum",
            "P4",
            dict(swing, theta0=0.7, mu_p=0.1),
            10,
            60,
            2e-3,
            pend,
            length,
        ),
        _scn("ramp frictionless", "ramp", "R0", still, 1.2, 60, 1e-3, ["R0", "R1"], ramp),
        _scn("ramp block", "ramp", "R1", dict(still, mu_k=0.3), 1.2, 60, 1e-3, ["R0", "R1"], ramp),
        _scn(
            "ramp ball",
            "ramp",
            "R2",
            dict(still, beta=0.4),
            1.2,
            60,
            1e-3,
            ["R0", "R2", "R3"],
            ramp,
        ),
        _scn(
            "ramp ball + rolling resistance",
            "ramp",
            "R3",
            dict(still, beta=0.4, c_rr=0.03),
            1.2,
            60,
            1e-3,
            ["R0", "R2", "R3"],
            ramp,
            accept=("R2",),
            note="c_rr and beta are not separable at one angle",
        ),
        _scn("bounce constant e", "bounce", "B0", dict(bb, e=0.8), 3.0, 240, 5e-4, bounce),
        _scn(
            "bounce speed-dependent e",
            "bounce",
            "B1",
            dict(bb, e0=0.9, e1=0.04),
            3.0,
            240,
            5e-4,
            bounce,
        ),
        _scn("bounce with drag", "bounce", "B2", dict(bb, e=0.8, k=0.3), 3.0, 240, 5e-4, bounce),
        _scn(
            "spin viscous",
            "spin_down",
            "S0",
            dict(phi0=0.0, omega0=40.0, b=0.25),
            15,
            60,
            0.03,
            spin,
        ),
        _scn(
            "spin Coulomb",
            "spin_down",
            "S1",
            dict(phi0=0.0, omega0=40.0, c=3.0),
            15,
            60,
            0.03,
            spin,
        ),
        _scn(
            "spin viscous + Coulomb",
            "spin_down",
            "S2",
            dict(phi0=0.0, omega0=40.0, b=0.15, c=1.5),
            15,
            60,
            0.03,
            spin,
        ),
        _scn(
            "spin Coulomb + air",
            "spin_down",
            "S3",
            dict(phi0=0.0, omega0=40.0, c=1.5, k=0.0015),
            15,
            60,
            0.03,
            spin,
        ),
        _scn(
            "spring undamped",
            "oscillator",
            "O0",
            dict(y0=0.05, v0=0.0, y_eq=0.0, k_over_m=100.0),
            6,
            60,
            5e-4,
            ["O0", "O1"],
        ),
        _scn(
            "spring damped",
            "oscillator",
            "O1",
            dict(y0=0.05, v0=0.0, y_eq=0.0, k_over_m=100.0, gamma=0.4),
            6,
            60,
            5e-4,
            ["O0", "O1"],
        ),
    ]


SCENARIOS = {s.name: s for s in scenarios()}


def _task(args: tuple[str, int]) -> dict:
    name, index = args
    scn = SCENARIOS[name]
    rng = np.random.default_rng(10_000 + index)
    case = scn.case(rng)
    runner = InProcessRunner()
    started = time.monotonic()
    select_result = runner.run(case.job(scn.candidates, plots=False, job_id=f"{name}-{index}"))
    selection = select_models(select_result, sigma=scn.sigma, seed=index)
    fit = runner.run(case.job([scn.truth_id], mode="final", plots=False)).models[0]
    z_scores = {}
    if fit.full is not None:
        for est in fit.full.params:
            truth = case.truth.get(est.name)
            if truth is not None and est.se and (est.kind == "physical" or est.derived):
                z_scores[est.name] = (est.value - truth) / est.se
    return {
        "selected": selection.selected_id,
        "status": selection.status,
        "z": z_scores,
        "seconds": time.monotonic() - started,
    }


def run_recovery(names: list[str], n: int, workers: int) -> list[dict]:
    jobs = [(name, i) for name in names for i in range(n)]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers) as pool:
        outcomes = pool.map(_task, jobs, chunksize=1)
    rows = []
    for name in names:
        runs = [o for (nm, _), o in zip(jobs, outcomes, strict=True) if nm == name]
        scn = SCENARIOS[name]
        chosen = Counter(r["selected"] for r in runs)
        right = sum(chosen[i] for i in scn.accepted())
        exact = chosen[scn.truth_id]
        status = Counter(r["status"] for r in runs)
        names_z = sorted({k for r in runs for k in r["z"]})
        within = {k: np.mean([abs(r["z"][k]) <= 2 for r in runs if k in r["z"]]) for k in names_z}
        scored = [all(abs(v) <= 2 for v in r["z"].values()) for r in runs if r["z"]]
        every = float(np.mean(scored)) if scored else float("nan")
        rows.append(
            {
                "scenario": name,
                "truth": scn.truth_id,
                "n": n,
                "selected_truth": exact / n,
                "selected_accepted": right / n,
                "chosen": dict(chosen),
                "status": dict(status),
                "within_2sigma": within,
                "all_within_2sigma": every,
                "seconds_per_run": float(np.mean([r["seconds"] for r in runs])),
                "note": scn.note,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Interval coverage with injected calibration and timing errors

BUDGET = CalibrationBudget(
    sigma_scale=0.003, sigma_offset=0.003, sigma_angle=0.005, sigma_timing=0.001
)


def inject_calibration_error(case: SyntheticCase, rng: np.random.Generator, budget=BUDGET):
    """Return (t, observed, constants) as a mis-calibrated measurement of the true motion."""
    from .library import get_model

    model = get_model(case.model_id, case.system)
    eps = {k: rng.normal(0.0, budget.sigma(k)) for k in ("scale", "offset", "angle", "timing")}
    scale = (1 + eps["scale"]) * (1 + eps["offset"])
    t = case.t * (1 + eps["timing"])
    observed = {
        o: v * scale if model.observable_units.get(o) == "m" else v
        for o, v in case.observed.items()
    }
    constants = {
        k: v * scale if k in model.length_constants else v for k, v in case.constants.items()
    }
    if "alpha" in constants:
        constants["alpha"] += eps["angle"]
    return t, observed, constants


COVERAGE_TARGETS = {
    # scenario -> (quantity, candidates for the selection, library ids)
    "free_fall dense ball": "g",
    "free_fall foam ball": "g",
    "pendulum 40 deg": "g",
    "ramp block": "mu_k",
    "bounce constant e": "e",
    "spin Coulomb": "c",
}


def _coverage_task(args: tuple[str, int, int, int]) -> dict:
    name, index, n_boot, n_perturb = args
    scn = SCENARIOS[name]
    quantity = COVERAGE_TARGETS[name]
    rng = np.random.default_rng(50_000 + index)
    case = scn.case(rng)
    t, observed, constants = inject_calibration_error(case, rng)
    sigma = dict(case.sigma)
    result = analyze(
        t, observed, sigma, system=scn.system, constants=constants, library_ids=[scn.truth_id],
        segment_id=case.segment_id, budget=BUDGET, review="accepted", n_boot=n_boot,
        n_perturb=n_perturb, seed=index, job_id=f"cov-{name}-{index}",
    )  # fmt: skip
    q = next((q for q in result.quantities if q.name == quantity), None)
    truth = case.truth[quantity]
    return {"covered": q is not None and q.ci_low <= truth <= q.ci_high, "ok": q is not None}


def run_coverage(names: list[str], n: int, workers: int, n_boot: int, n_perturb: int) -> list[dict]:
    jobs = [(name, i, n_boot, n_perturb) for name in names for i in range(n)]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers) as pool:
        outcomes = pool.map(_coverage_task, jobs, chunksize=1)
    rows = []
    for name in names:
        runs = [o for (nm, *_), o in zip(jobs, outcomes, strict=True) if nm == name]
        rows.append(
            {
                "scenario": name,
                "quantity": COVERAGE_TARGETS[name],
                "n": len(runs),
                "coverage": float(np.mean([r["covered"] for r in runs])),
                "failed": sum(not r["ok"] for r in runs),
            }
        )
    return rows


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.0%}"


def _print_recovery(rows: list[dict]) -> None:
    print(
        "| scenario | truth | picked truth | picked truth or equivalent | status "
        "| all params within 2 sigma | s/run |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        status = ", ".join(f"{k} {v}" for k, v in sorted(r["status"].items()))
        print(
            f"| {r['scenario']} | {r['truth']} | {r['selected_truth']:.0%} "
            f"| {r['selected_accepted']:.0%} | {status} | {_pct(r['all_within_2sigma'])} "
            f"| {r['seconds_per_run']:.1f} |"
        )
    print()
    for r in rows:
        per = ", ".join(f"{k} {v:.0%}" for k, v in r["within_2sigma"].items())
        print(f"- {r['scenario']}: chose {r['chosen']}; within 2 sigma: {per}. {r['note']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(multiprocessing.cpu_count() - 1, 1))
    parser.add_argument("--only", nargs="*", help="scenario names (default: all)")
    parser.add_argument("--coverage", type=int, default=0, help="run N coverage trials per target")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--n-perturb", type=int, default=40)
    args = parser.parse_args()
    if args.coverage:
        names = args.only or list(COVERAGE_TARGETS)
        for row in run_coverage(names, args.coverage, args.workers, args.n_boot, args.n_perturb):
            print(row)
        return
    _print_recovery(run_recovery(args.only or list(SCENARIOS), args.n, args.workers))


if __name__ == "__main__":
    main()
