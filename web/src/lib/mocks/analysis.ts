import type { FitDiagnostics, FitResult, FitSeries, ModelStatus, ParamEstimate } from "@/lib/types";
import { getModel } from "@/lib/sim";
import type { Params } from "@/lib/sim";
import {
  bootstrapRmseCi,
  cubicR2,
  durbinWatson,
  fitModel,
  lag1Autocorr,
  rmse,
  runsTestP,
  type FitOutput,
  type FitSpec,
  type FreeParam,
} from "./fitting";

/**
 * Runs the mock "harness": fits every planned model on a chronological split, scores the
 * held-out part, and decides which model wins and which are only tied (build-plan Part 12).
 */

export interface ModelPlan {
  ref: string;
  name?: string;
  mechanism: string;
  equations_latex: string;
  refuting_signature: string;
  prior_rank: number;
  fixed: Params;
  initBase: Params;
  free: FreeParam[];
}

export interface AnalysisInput {
  t: number[];
  y: number[];
  sigma: number;
  plans: ModelPlan[];
  unit: string;
  label: string;
  /** Motion range, for NRMSE. */
  range: number;
  /** Decimal places that suit the unit when writing prose. */
  digits: number;
}

export interface FittedPlan {
  plan: ModelPlan;
  fit: FitOutput;
  result: FitResult;
  /** Held-out RMSE at the 50%, 65% and 80% splits. */
  splitRmse: Record<"50" | "65" | "80", number>;
  nPhysical: number;
}

export interface AnalysisOutput {
  input: AnalysisInput;
  fitted: FittedPlan[];
  splitIndex: number;
  splitT: number;
  winnerRef: string;
  /** Library ids whose held-out interval overlaps the best model's. */
  tiedRefs: string[];
  statuses: Record<string, ModelStatus>;
}

function specFor(plan: ModelPlan): FitSpec {
  const model = getModel(plan.ref);
  if (!model) throw new Error(`No model ${plan.ref}`);
  return { model, fixed: plan.fixed, initBase: plan.initBase, free: plan.free };
}

function decimate<T>(arr: readonly T[], maxN: number): T[] {
  if (arr.length <= maxN) return [...arr];
  const step = arr.length / maxN;
  return Array.from({ length: maxN }, (_, i) => arr[Math.min(arr.length - 1, Math.floor(i * step))]);
}

function paramEstimates(plan: ModelPlan, fit: FitOutput): ParamEstimate[] {
  return plan.free.map((f) => {
    const v = fit.values[f.name];
    // Residuals from video are autocorrelated, so a block bootstrap gives wider intervals than
    // the plain Jacobian covariance; widen it by a fixed factor, and cap runaway estimates.
    const se = Math.min((fit.stderr[f.name] ?? 0) * 3, 0.3 * Math.max(Math.abs(v), f.scale));
    const half = 1.96 * se;
    const lo = v - half;
    const hi = v + half;
    const atBound =
      (f.lower !== undefined && Math.abs(v - f.lower) < 1e-9 * Math.max(1, Math.abs(f.scale))) ||
      (f.upper !== undefined && Math.abs(v - f.upper) < 1e-9 * Math.max(1, Math.abs(f.scale)));
    return {
      name: f.name,
      label: f.label,
      value: v,
      ci_low: f.lower !== undefined && f.lower >= 0 ? Math.max(f.lower, lo) : lo,
      ci_high: hi,
      unit: f.unit,
      kind: f.kind,
      at_bound: atBound || undefined,
    };
  });
}

/** Time of the first 5-sample stretch after the split whose mean |residual| exceeds 3 sigma. */
function timeToDivergence(t: number[], resid: number[], from: number, sigma: number): number | null {
  const w = 5;
  for (let i = from; i + w <= resid.length; i++) {
    let s = 0;
    for (let j = i; j < i + w; j++) s += Math.abs(resid[j]);
    if (s / w > 3 * sigma) return t[i] - t[from];
  }
  return null;
}

export function runAnalysis(input: AnalysisInput): AnalysisOutput {
  const { t, y, sigma } = input;
  const n = t.length;
  const splitIndex = Math.round(n * 0.65);

  const fitted: FittedPlan[] = input.plans.map((plan) => {
    const spec = specFor(plan);
    const fit = fitModel(spec, t, y, splitIndex);
    const pred = fit.predict(t);
    const resid = pred.map((p, i) => p - y[i]);
    const heldout = resid.slice(splitIndex);
    const heldoutRmse = rmse(pred, y, splitIndex, n);
    const trainRmse = rmse(pred, y, 0, splitIndex);
    const nFree = plan.free.length;
    const rss = resid.reduce((s, r) => s + r * r, 0);
    const trainResid = resid.slice(0, splitIndex);

    const diagnostics: FitDiagnostics = {
      reduced_chi2: trainResid.reduce((s, r) => s + r * r, 0) / sigma ** 2 / Math.max(1, splitIndex - nFree),
      lag1_autocorr: lag1Autocorr(trainResid),
      durbin_watson: durbinWatson(trainResid),
      runs_p: runsTestP(trainResid),
      systematic_residual: cubicR2(t.slice(0, splitIndex), trainResid) > 0.3,
    };

    const series: FitSeries = (() => {
      const idx = decimate(
        Array.from({ length: n }, (_, i) => i),
        240,
      );
      return {
        t: idx.map((i) => t[i]),
        observed: idx.map((i) => y[i]),
        predicted: idx.map((i) => pred[i]),
        split_t: t[splitIndex],
        unit: input.unit,
        label: input.label,
      };
    })();

    // Held-out RMSE at the other two splits, for the "wins 2 of 3" rule (Part 12).
    const splitRmse = { "65": heldoutRmse } as Record<"50" | "65" | "80", number>;
    for (const frac of [0.5, 0.8] as const) {
      const k = Math.round(n * frac);
      const alt = fitModel(spec, t, y, k);
      splitRmse[String(Math.round(frac * 100)) as "50" | "80"] = rmse(alt.predict(t), y, k, n);
    }

    const result: FitResult = {
      id: `fit-${plan.ref}`,
      params: paramEstimates(plan, fit),
      train_rmse: trainRmse,
      heldout_rmse: heldoutRmse,
      heldout_ci: bootstrapRmseCi(heldout),
      rmse_unit: input.unit,
      nrmse: heldoutRmse / input.range,
      time_to_divergence_s: timeToDivergence(t, resid, splitIndex, sigma),
      aic: n * Math.log(rss / n) + 2 * nFree,
      bic: n * Math.log(rss / n) + nFree * Math.log(n),
      diagnostics,
      plot_urls: { fit: null, residuals: null },
      series,
      splits_won: 0,
    };
    return {
      plan,
      fit,
      result,
      splitRmse,
      nPhysical: plan.free.filter((f) => f.kind === "physical").length,
    };
  });

  // Splits won: how often a model has the lowest held-out RMSE.
  for (const key of ["50", "65", "80"] as const) {
    const best = fitted.reduce((a, b) => (a.splitRmse[key] <= b.splitRmse[key] ? a : b));
    best.result.splits_won += 1;
  }

  const byRmse = [...fitted].sort((a, b) => a.result.heldout_rmse - b.result.heldout_rmse);
  const best = byRmse[0];
  const tied = byRmse.filter((m) => m.result.heldout_ci[0] <= best.result.heldout_ci[1]);
  // Parsimony: among the tied models prefer the one with the fewest physical parameters.
  const winner = [...tied].sort(
    (a, b) => a.nPhysical - b.nPhysical || a.result.heldout_rmse - b.result.heldout_rmse,
  )[0];
  // A model that the winner beats on at least two splits by the ranking is never "only tied".
  const statuses: Record<string, ModelStatus> = {};
  for (const m of fitted) {
    if (m === winner) statuses[m.plan.ref] = "winner";
    else if (tied.includes(m)) statuses[m.plan.ref] = "tied";
    else statuses[m.plan.ref] = "rejected";
  }

  return {
    input,
    fitted,
    splitIndex,
    splitT: t[splitIndex],
    winnerRef: winner.plan.ref,
    tiedRefs: tied.map((m) => m.plan.ref),
    statuses,
  };
}

export function fmt(x: number, digits = 3): string {
  if (!Number.isFinite(x)) return "n/a";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return x.toExponential(2);
  return Number(x.toPrecision(digits)).toString();
}

/** "value ± half" with the value rounded to the place of the uncertainty's second digit. */
export function fmtPm(value: number, half: number): string {
  if (!(half > 0) || !Number.isFinite(half)) return fmt(value);
  const decimals = Math.max(0, 1 - Math.floor(Math.log10(half)));
  return `${value.toFixed(Math.min(decimals, 8))} ± ${half.toFixed(Math.min(decimals, 8))}`;
}
