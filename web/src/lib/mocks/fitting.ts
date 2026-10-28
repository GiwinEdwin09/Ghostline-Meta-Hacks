import { integrate } from "@/lib/sim";
import type { Params, SimModel } from "@/lib/sim";
import { mulberry32, quantile } from "./rng";

/**
 * A tiny fitting harness for the mock experiments.
 *
 * The real harness (build-plan Part 11) runs in a Modal Sandbox with scipy. The mocks only
 * need numbers that are consistent with the curves they plot, so this fits each model to the
 * training window with Nelder-Mead and reports held-out error and rough intervals computed
 * from the same residuals (Jacobian covariance for parameters, a block bootstrap for RMSE).
 * It is mock-only code: nothing in the production path imports it.
 */

export interface FreeParam {
  name: string;
  init: number;
  /** Typical size, used for the starting simplex and finite differences. */
  scale: number;
  lower?: number;
  upper?: number;
  kind: "physical" | "initial_condition";
  unit: string;
  label: string;
}

export interface FitSpec {
  model: SimModel;
  /** Held-fixed physical parameters. */
  fixed: Params;
  /** Initial conditions that are not being fitted. */
  initBase: Params;
  free: FreeParam[];
}

export interface FitOutput {
  /** Fitted values by name (free parameters only). */
  values: Params;
  cost: number;
  /** Prediction at the given times. */
  predict(t: readonly number[]): number[];
  /** 1-sigma standard errors by name, from the Jacobian covariance. */
  stderr: Record<string, number>;
}

const GRID_DT = 1 / 240;

function simulateObs(spec: FitSpec, values: Params, tEnd: number): (t: readonly number[]) => number[] {
  const physical: Params = { ...spec.fixed };
  const init: Params = { ...spec.initBase };
  for (const f of spec.free) {
    if (f.kind === "physical") physical[f.name] = values[f.name];
    else init[f.name] = values[f.name];
  }
  const cf = spec.model.closedForm;
  if (cf) return (t) => t.map((x) => cf(x, physical, init)[0]);
  const res = integrate(spec.model, physical, init, { tEnd: tEnd + GRID_DT, dt: GRID_DT });
  return (t) =>
    t.map((x) => {
      const f = x / GRID_DT;
      const i = Math.min(res.obs[0].length - 2, Math.max(0, Math.floor(f)));
      const w = f - i;
      return res.obs[0][i] * (1 - w) + res.obs[0][i + 1] * w;
    });
}

function clampToBounds(spec: FitSpec, x: number[]): { values: Params; penalty: number } {
  const values: Params = {};
  let penalty = 0;
  spec.free.forEach((f, i) => {
    let v = x[i];
    if (f.lower !== undefined && v < f.lower) {
      penalty += ((f.lower - v) / f.scale) ** 2;
      v = f.lower;
    }
    if (f.upper !== undefined && v > f.upper) {
      penalty += ((v - f.upper) / f.scale) ** 2;
      v = f.upper;
    }
    values[f.name] = v;
  });
  return { values, penalty };
}

export function nelderMead(
  f: (x: number[]) => number,
  x0: number[],
  scale: number[],
  maxIter = 600,
): { x: number[]; fx: number } {
  const n = x0.length;
  let simplex: number[][] = [x0.slice()];
  for (let i = 0; i < n; i++) {
    const p = x0.slice();
    p[i] += scale[i];
    simplex.push(p);
  }
  let fv = simplex.map(f);
  for (let iter = 0; iter < maxIter; iter++) {
    const order = fv.map((v, i) => i).sort((a, b) => fv[a] - fv[b]);
    simplex = order.map((i) => simplex[i]);
    fv = order.map((i) => fv[i]);
    if (Math.abs(fv[n] - fv[0]) < 1e-14 * (1 + Math.abs(fv[0]))) break;
    const centroid = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) centroid[j] += simplex[i][j] / n;
    const along = (t: number) => centroid.map((c, j) => c + t * (simplex[n][j] - c));
    const xr = along(-1);
    const fr = f(xr);
    if (fr < fv[0]) {
      const xe = along(-2);
      const fe = f(xe);
      if (fe < fr) {
        simplex[n] = xe;
        fv[n] = fe;
      } else {
        simplex[n] = xr;
        fv[n] = fr;
      }
    } else if (fr < fv[n - 1]) {
      simplex[n] = xr;
      fv[n] = fr;
    } else {
      const xc = fr < fv[n] ? along(-0.5) : along(0.5);
      const fc = f(xc);
      if (fc < Math.min(fr, fv[n])) {
        simplex[n] = xc;
        fv[n] = fc;
      } else {
        for (let i = 1; i <= n; i++) {
          simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]));
          fv[i] = f(simplex[i]);
        }
      }
    }
  }
  const best = fv.indexOf(Math.min(...fv));
  return { x: simplex[best], fx: fv[best] };
}

function invert(a: number[][]): number[][] | null {
  const n = a.length;
  const m = a.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    if (Math.abs(m[p][c]) < 1e-300) return null;
    [m[c], m[p]] = [m[p], m[c]];
    const d = m[c][c];
    for (let j = 0; j < 2 * n; j++) m[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const k = m[r][c];
      for (let j = 0; j < 2 * n; j++) m[r][j] -= k * m[c][j];
    }
  }
  return m.map((row) => row.slice(n));
}

/** Least-squares fit on the first `trainCount` samples. */
export function fitModel(
  spec: FitSpec,
  t: readonly number[],
  y: readonly number[],
  trainCount: number,
): FitOutput {
  const tTrain = t.slice(0, trainCount);
  const yTrain = y.slice(0, trainCount);
  const tEnd = t[t.length - 1];

  const cost = (x: number[]) => {
    const { values, penalty } = clampToBounds(spec, x);
    const pred = simulateObs(spec, values, tEnd)(tTrain);
    let s = 0;
    for (let i = 0; i < tTrain.length; i++) {
      const r = pred[i] - yTrain[i];
      s += r * r;
    }
    return s + penalty * 1e3 * (s + 1);
  };

  let x = spec.free.map((f) => f.init);
  const scale = spec.free.map((f) => f.scale);
  let fx = Infinity;
  for (let restart = 0; restart < 3; restart++) {
    const out = nelderMead(cost, x, scale.map((s) => s / (1 + restart)), 500);
    x = out.x;
    fx = out.fx;
  }

  const { values } = clampToBounds(spec, x);
  const predictAll = simulateObs(spec, values, tEnd);

  // Jacobian covariance on the training window.
  const k = spec.free.length;
  const n = tTrain.length;
  const J: number[][] = Array.from({ length: n }, () => new Array<number>(k).fill(0));
  for (let j = 0; j < k; j++) {
    const h = spec.free[j].scale * 1e-3;
    const up = { ...values, [spec.free[j].name]: values[spec.free[j].name] + h };
    const dn = { ...values, [spec.free[j].name]: values[spec.free[j].name] - h };
    const pu = simulateObs(spec, up, tEnd)(tTrain);
    const pd = simulateObs(spec, dn, tEnd)(tTrain);
    for (let i = 0; i < n; i++) J[i][j] = (pu[i] - pd[i]) / (2 * h);
  }
  const JtJ = Array.from({ length: k }, (_, a) =>
    Array.from({ length: k }, (_, b) => J.reduce((s, row) => s + row[a] * row[b], 0)),
  );
  const inv = invert(JtJ);
  const s2 = fx / Math.max(1, n - k);
  const stderr: Record<string, number> = {};
  spec.free.forEach((f, j) => {
    stderr[f.name] = inv ? Math.sqrt(Math.max(0, inv[j][j]) * s2) : Math.abs(values[f.name]) * 0.1;
  });

  return { values, cost: fx, predict: predictAll, stderr };
}

/* ------------------------------------------------------------------------------------ */
/* Scoring helpers                                                                       */
/* ------------------------------------------------------------------------------------ */

export function rmse(a: readonly number[], b: readonly number[], from = 0, to = a.length): number {
  let s = 0;
  for (let i = from; i < to; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / Math.max(1, to - from));
}

/** Block bootstrap of the held-out RMSE (blocks of about 10 samples, fixed seed). */
export function bootstrapRmseCi(residuals: readonly number[], seed = 7, draws = 300, block = 10): [number, number] {
  const n = residuals.length;
  if (n < 2) return [0, 0];
  const rng = mulberry32(seed);
  const out: number[] = [];
  const nBlocks = Math.max(1, Math.ceil(n / block));
  for (let d = 0; d < draws; d++) {
    let s = 0;
    let count = 0;
    for (let b = 0; b < nBlocks; b++) {
      const start = Math.floor(rng() * Math.max(1, n - block + 1));
      for (let i = start; i < Math.min(n, start + block) && count < n; i++, count++) s += residuals[i] ** 2;
    }
    out.push(Math.sqrt(s / Math.max(1, count)));
  }
  out.sort((a, b) => a - b);
  return [quantile(out, 0.025), quantile(out, 0.975)];
}

export function lag1Autocorr(r: readonly number[]): number {
  const m = r.reduce((s, v) => s + v, 0) / r.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < r.length; i++) {
    den += (r[i] - m) ** 2;
    if (i > 0) num += (r[i] - m) * (r[i - 1] - m);
  }
  return den > 0 ? num / den : 0;
}

export function durbinWatson(r: readonly number[]): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < r.length; i++) {
    den += r[i] ** 2;
    if (i > 0) num += (r[i] - r[i - 1]) ** 2;
  }
  return den > 0 ? num / den : 2;
}

/** Wald-Wolfowitz runs test, normal approximation; returns a two-sided p-value. */
export function runsTestP(r: readonly number[]): number {
  const signs = r.map((v) => v >= 0);
  const n1 = signs.filter(Boolean).length;
  const n2 = signs.length - n1;
  if (n1 === 0 || n2 === 0) return 0;
  let runs = 1;
  for (let i = 1; i < signs.length; i++) if (signs[i] !== signs[i - 1]) runs++;
  const mu = (2 * n1 * n2) / (n1 + n2) + 1;
  const varr = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1));
  const z = (runs - mu) / Math.sqrt(Math.max(varr, 1e-12));
  return erfc(Math.abs(z) / Math.SQRT2);
}

function erfc(x: number): number {
  // Abramowitz-Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return 1 - y;
}

/** Share of the residual variance explained by a cubic in time (the "systematic residual" flag). */
export function cubicR2(t: readonly number[], r: readonly number[]): number {
  const n = t.length;
  if (n < 8) return 0;
  const t0 = t[0];
  const span = t[n - 1] - t0 || 1;
  const X = t.map((v) => {
    const u = (2 * (v - t0)) / span - 1;
    return [1, u, u * u, u * u * u];
  });
  const A = Array.from({ length: 4 }, (_, a) =>
    Array.from({ length: 4 }, (_, b) => X.reduce((s, row) => s + row[a] * row[b], 0)),
  );
  const bvec = Array.from({ length: 4 }, (_, a) => X.reduce((s, row, i) => s + row[a] * r[i], 0));
  const inv = invert(A);
  if (!inv) return 0;
  const coef = inv.map((row) => row.reduce((s, v, j) => s + v * bvec[j], 0));
  const mean = r.reduce((s, v) => s + v, 0) / n;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const fit = coef.reduce((s, c, j) => s + c * X[i][j], 0);
    ssTot += (r[i] - mean) ** 2;
    ssRes += (r[i] - fit) ** 2;
  }
  return ssTot > 0 ? 1 - ssRes / ssTot : 0;
}
