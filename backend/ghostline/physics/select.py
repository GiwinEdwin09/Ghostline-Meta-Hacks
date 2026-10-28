"""Held-out model selection and the final uncertainty budget (build plan Part 12).

This runs on the trusted side, on the harness output. The harness fits on the first 50%, 65% and
80% of the motion and returns each model's forecast errors on the rest. Here a paired block
bootstrap decides which differences are real, parsimony breaks ties, and the final numbers get
statistical, calibration, timing and model-choice uncertainty.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .harness.fitting import Z_95, block_bootstrap_indices
from .harness.schema import (
    Diagnostics,
    FitResult,
    ModelResult,
    ParamEstimate,
    Perturbation,
    PlotRef,
)
from .library import get_model

N_BOOT = 1000
DIVERGENCE_SIGMAS = 3.0
CALIBRATION_KEYS = ("scale", "offset", "angle", "timing")


class CalibrationBudget(BaseModel):
    """Calibration and timing uncertainty from Parts 5 and 8, as one-sigma values."""

    sigma_scale: float = 0.0  # relative, from the sheet corners
    sigma_offset: float = 0.0  # relative effect of the plane offset d and the distance Z
    sigma_angle: float = 0.0  # radians, the ramp angle
    sigma_timing: float = 0.0  # relative error of the time base

    def sigma(self, key: str) -> float:
        return getattr(self, f"sigma_{key}")


class ModelScore(BaseModel):
    id: str
    status: str
    error: str | None = None
    n_physical: int = 0
    heldout_rmse: dict[str, float] = Field(default_factory=dict)
    heldout_ci: dict[str, tuple[float, float]] = Field(default_factory=dict)
    nrmse: float | None = None
    divergence_time: float | None = None
    train_rmse: float | None = None
    train_diagnostics: Diagnostics | None = None
    full_diagnostics: Diagnostics | None = None
    params: list[ParamEstimate] = Field(default_factory=list)
    aic: float | None = None
    bic: float | None = None
    plots: list[PlotRef] = Field(default_factory=list)


class SplitOutcome(BaseModel):
    split: float
    leader: str
    runner_up: str | None
    selected: Literal["wins", "tied", "loses"]


class Selection(BaseModel):
    status: Literal["decided", "tied", "inconclusive"]
    selected_id: str | None
    splits_won: int = 0
    n_splits: int = 0
    primary_split: float = 0.65
    tied_with: list[str] = Field(default_factory=list)
    tied_equal_or_simpler: list[str] = Field(default_factory=list)
    parsimony: bool = False
    message: str = ""
    models: list[ModelScore] = Field(default_factory=list)
    splits: list[SplitOutcome] = Field(default_factory=list)

    def score(self, model_id: str) -> ModelScore:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)


def _key(split: float) -> str:
    return f"{split:.2f}"


def _tie_rank(model: ModelResult, system: str) -> int:
    if model.library_ref is None:
        return 0
    try:
        return get_model(model.library_ref, system).tie_rank
    except KeyError:
        return 0


def _aligned_errors(models: list[ModelResult], split: float):
    """Realized squared errors and parameter-draw pools on the test times every model shares."""
    picked = []
    for m in models:
        fit = next((s for s in m.splits if abs(s.split - split) < 1e-9), None)
        if fit is None or not fit.heldout_errors or fit.t_test is None:
            return None
        draws = np.asarray(fit.heldout_draws) if fit.heldout_draws else None
        picked.append((np.round(fit.t_test, 9), np.asarray(fit.heldout_errors), draws))
    common = picked[0][0]
    for times, _, _ in picked[1:]:
        common = np.intersect1d(common, times)
    if common.size < 3:
        return None
    realized, pools = [], []
    for times, errs, draws in picked:
        keep = np.isin(times, common)
        realized.append(errs[keep] ** 2)
        rows = errs[keep][None, :]
        if draws is not None:
            rows = np.vstack([rows, draws[:, keep]])
        pools.append(rows**2)
    return np.vstack(realized), pools


def _bootstrap_rmse(pools: list[np.ndarray], n_boot: int, block: int, rng) -> np.ndarray:
    """RMSE per model per resample: shared blocks of test points, one parameter draw per model."""
    n = pools[0].shape[1]
    block = max(1, min(block, n // 4))
    idx = np.stack([block_bootstrap_indices(n, block, rng) for _ in range(n_boot)])
    out = np.empty((len(pools), n_boot))
    for i, pool in enumerate(pools):
        draw = rng.integers(0, pool.shape[0], size=n_boot)
        out[i] = np.sqrt(pool[draw[:, None], idx].mean(axis=1))
    return out


def _divergence(t_test: list[float] | None, errors: list[float] | None, sigma: float | None):
    if not t_test or not errors or not sigma:
        return None
    above = np.where(np.asarray(errors) > DIVERGENCE_SIGMAS * sigma)[0]
    return float(t_test[above[0]]) if above.size else None


def _score_for(m: ModelResult, motion_range, sigma) -> ModelScore:
    score = ModelScore(id=m.id, status=m.status, error=m.error, n_physical=m.n_physical)
    if not m.splits:
        return score
    primary = m.splits[0]
    score.params = primary.params
    score.train_rmse = primary.train.rmse
    score.train_diagnostics = primary.train
    score.full_diagnostics = m.full.train if m.full else None
    score.aic, score.bic = m.aic, m.bic
    score.plots = m.plots
    if primary.heldout_rmse is not None:
        if motion_range:
            score.nrmse = primary.heldout_rmse / motion_range
        score.divergence_time = _divergence(primary.t_test, primary.heldout_errors, sigma)
    return score


def select_models(
    result: FitResult,
    *,
    primary_split: float = 0.65,
    sigma: float | None = None,
    motion_range: float | None = None,
    n_boot: int = N_BOOT,
    block: int = 10,
    seed: int = 0,
) -> Selection:
    """Choose a winner from a harness `select` result, or say the data can't separate the models."""
    scores = [_score_for(m, motion_range, sigma) for m in result.models]
    ok = [m for m in result.models if m.status == "ok" and m.splits and m.full]
    system = result.system
    if len(ok) < 2:
        only = ok[0].id if ok else None
        why = "only one model could be fitted" if ok else "no model could be fitted"
        return Selection(status="inconclusive", selected_id=only, message=why, models=scores)

    rng = np.random.default_rng(seed)
    split_list = sorted({s.split for s in ok[0].splits})
    primary = min(split_list, key=lambda s: abs(s - primary_split))
    ids = [m.id for m in ok]
    index = {mid: i for i, mid in enumerate(ids)}
    rmse_by_split: dict[float, np.ndarray] = {}
    boot_by_split: dict[float, np.ndarray] = {}
    for split in split_list:
        aligned = _aligned_errors(ok, split)
        if aligned is None:
            continue
        realized, pools = aligned
        rmse_by_split[split] = np.sqrt(realized.mean(axis=1))
        boot = _bootstrap_rmse(pools, n_boot, block, rng)
        boot_by_split[split] = boot
        for m_id, i in index.items():
            score = next(s for s in scores if s.id == m_id)
            score.heldout_rmse[_key(split)] = float(rmse_by_split[split][i])
            lo, hi = np.percentile(boot[i], [2.5, 97.5])
            score.heldout_ci[_key(split)] = (float(lo), float(hi))
    if primary not in rmse_by_split:
        return Selection(
            status="inconclusive", selected_id=None, message="no held-out data to compare",
            models=scores,
        )  # fmt: skip

    def beats(split: float, a: int, b: int) -> bool:
        """Model a predicts significantly better than b on this split."""
        boot = boot_by_split[split]
        return float(np.percentile(boot[b] - boot[a], 2.5)) > 0

    n_models = len(ids)
    prim_rmse = rmse_by_split[primary]
    best = int(np.argmin(prim_rmse))
    near_best = {best} | {j for j in range(n_models) if j != best and not beats(primary, best, j)}
    by_id = {m.id: m for m in ok}

    def parsimony_key(j: int):
        m = by_id[ids[j]]
        return (m.n_physical, _tie_rank(m, system), prim_rmse[j])

    winner = min(near_best, key=parsimony_key)

    outcomes: list[SplitOutcome] = []
    unseparated: set[int] = set(near_best) - {winner}
    for split in sorted(rmse_by_split):
        rm = rmse_by_split[split]
        order = np.argsort(rm)
        leader = int(order[0])
        rivals = [j for j in range(n_models) if j != winner]
        beaten_by = [j for j in rivals if beats(split, j, winner)]
        runner = int(order[1]) if leader == winner else leader
        if beaten_by:
            state = "loses"
        elif leader == winner and beats(split, winner, runner):
            state = "wins"
        else:
            state = "tied"
            unseparated.add(runner)
        outcomes.append(
            SplitOutcome(
                split=split,
                leader=ids[leader],
                runner_up=ids[runner],
                selected=state,  # type: ignore[arg-type]
            )
        )
    won = sum(o.selected == "wins" for o in outcomes)
    lost = sum(o.selected == "loses" for o in outcomes)
    n_splits = len(outcomes)
    need = 2 if n_splits >= 3 else n_splits
    if lost >= max(2, n_splits // 2 + 1):
        status = "inconclusive"
    elif won >= need:
        status = "decided"
    else:
        status = "tied"
    tied_with = [ids[j] for j in sorted(unseparated)]
    n_winner = by_id[ids[winner]].n_physical
    equal_or_simpler = [i for i in tied_with if by_id[i].n_physical <= n_winner]
    parsimony = winner != best and status != "inconclusive"
    return Selection(
        status=status,
        selected_id=ids[winner],
        splits_won=won,
        n_splits=n_splits,
        primary_split=primary,
        tied_with=tied_with,
        tied_equal_or_simpler=equal_or_simpler,
        parsimony=parsimony,
        message=_message(status, ids[winner], tied_with, equal_or_simpler, won, n_splits),
        models=scores,
        splits=outcomes,
    )


def _message(status, winner, tied_with, equal_or_simpler, won, n_splits) -> str:
    if status == "decided":
        return (
            f"{winner} predicts the held-out motion better than every rival "
            f"in {won} of {n_splits} splits."
        )
    if status == "inconclusive":
        return f"The held-out splits disagree about {winner}; the result is inconclusive."
    rivals = " and ".join(tied_with) if tied_with else "its rivals"
    tail = (
        f"{winner} is preferred because it has fewer parameters."
        if not equal_or_simpler
        else f"{winner} is listed first, but {rivals} fits as well."
    )
    return f"The data can't distinguish {winner} from {rivals}. {tail}"


def comparison_table(selection: Selection) -> list[dict]:
    """The `run_fits` table Wonder reads: training-window numbers and held-out scores only."""
    rows = []
    for m in selection.models:
        prim = _key(selection.primary_split)
        rows.append(
            {
                "model_id": m.id,
                "status": m.status,
                "error": m.error,
                "parameters": [
                    {
                        "name": p.name,
                        "value": p.value,
                        "ci95": [p.ci_low, p.ci_high],
                        "unit": p.unit,
                        "at_bound": p.at_bound,
                    }
                    for p in m.params
                    if p.kind == "physical"
                ],
                "train_rmse": m.train_rmse,
                "heldout_rmse": m.heldout_rmse.get(prim),
                "heldout_rmse_ci95": m.heldout_ci.get(prim),
                "nrmse": m.nrmse,
                "time_to_divergence_s": m.divergence_time,
                "diagnostics": m.train_diagnostics.model_dump() if m.train_diagnostics else None,
                "plot_ids": [p.id for p in m.plots if p.audience == "agent"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Final numbers and the uncertainty budget


def calibration_perturbations(
    budget: CalibrationBudget,
    constants: dict[str, float],
    n: int = 50,
    seed: int = 0,
) -> list[Perturbation]:
    """Antithetic draws of the calibration and timing errors, as harness perturbations."""
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((max(n // 2, 1), len(CALIBRATION_KEYS)))
    z = np.vstack([half, -half])[:n]
    draws = []
    for row in z:
        eps = {k: float(row[i]) * budget.sigma(k) for i, k in enumerate(CALIBRATION_KEYS)}
        override = {}
        if "alpha" in constants and eps["angle"]:
            override["alpha"] = constants["alpha"] + eps["angle"]
        draws.append(
            Perturbation(
                scale=(1 + eps["scale"]) * (1 + eps["offset"]),
                time_scale=1 + eps["timing"],
                constants=override,
                eps=eps,
            )
        )
    return draws


class FinalQuantity(BaseModel):
    name: str
    value: float
    unit: str
    ci_low: float
    ci_high: float
    sigma_total: float
    budget: dict[str, float]
    derived: bool = False
    at_bound: bool = False


def _decompose(
    deltas: np.ndarray, eps: list[dict[str, float]], budget: CalibrationBudget
) -> dict[str, float]:
    """Split the spread of the refit values across the calibration inputs by linear regression."""
    keys = [k for k in CALIBRATION_KEYS if budget.sigma(k) > 0 and all(k in e for e in eps)]
    out = dict.fromkeys(CALIBRATION_KEYS, 0.0)
    out["other"] = 0.0
    if deltas.size < 8:
        return out
    total_var = float(np.var(deltas, ddof=1))
    if keys:
        x = np.array([[e[k] for k in keys] for e in eps])
        x = x - x.mean(axis=0)
        coef, *_ = np.linalg.lstsq(x, deltas - deltas.mean(), rcond=None)
        for k, c in zip(keys, coef, strict=True):
            out[k] = abs(float(c)) * budget.sigma(k)
    explained = sum(out[k] ** 2 for k in CALIBRATION_KEYS)
    out["other"] = math.sqrt(max(total_var - explained, 0.0))
    return out


def final_quantities(
    final: ModelResult,
    budget: CalibrationBudget,
    tied: list[ModelResult] | None = None,
) -> list[FinalQuantity]:
    """Headline numbers for the winner with the full uncertainty budget.

    `final` is the winner from a harness `final` job (bootstrap and calibration refits).
    `tied` are the other models' results from the selection job; the spread of each shared
    quantity across them is the model-choice term.
    """
    if final.full is None:
        raise ValueError("final result has no fit")
    quantities = []
    for est in final.full.params:
        if est.kind != "physical" and not est.derived:
            continue
        if est.ci_low is not None and est.ci_high is not None:
            stat = (est.ci_high - est.ci_low) / (2 * Z_95)
        else:
            stat = est.se or 0.0
        reps = [r[est.name] for r in final.replicates if est.name in r]
        parts = dict.fromkeys((*CALIBRATION_KEYS, "other"), 0.0)
        if len(reps) == len(final.replicates) and reps:
            parts = _decompose(np.asarray(reps) - est.value, final.replicate_eps, budget)
        choice = 0.0
        values = [est.value]
        for other in tied or []:
            match = next(
                (p for p in (other.full.params if other.full else []) if p.name == est.name), None
            )
            if match is not None:
                values.append(match.value)
        if len(values) > 1:
            choice = (max(values) - min(values)) / 2
        budget_row = {"statistical": stat, **parts, "model_choice": choice}
        total = math.sqrt(sum(v * v for v in budget_row.values()))
        quantities.append(
            FinalQuantity(
                name=est.name,
                value=est.value,
                unit=est.unit,
                ci_low=est.value - Z_95 * total,
                ci_high=est.value + Z_95 * total,
                sigma_total=total,
                budget=budget_row,
                derived=est.derived,
                at_bound=est.at_bound,
            )
        )
    return quantities
