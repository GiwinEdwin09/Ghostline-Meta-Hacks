import { integrate } from "./rk4";
import { getModel } from "./models";
import { floorEvent } from "./models/free-fall";
import { SHAPE_BETA, type RampShape } from "./models/ramp";
import { spinStopTime } from "./models/spin";
import { exactPeriod, smallAnglePeriod } from "./models/pendulum";
import type { ModelEvent, Params, SimContext, SimModel, SimResult } from "./types";

/**
 * Turns an experiment's fitted model plus a set of slider values into something the
 * integrator can run (build-plan Part 18). The same functions give the "fitted" baseline
 * (default slider values) and the "what-if" (edited values), so the two ghosts always
 * differ only by what the user changed.
 */

export const GRAVITY_PRESETS = [
  { id: "earth", label: "Earth", value: 9.81 },
  { id: "moon", label: "Moon", value: 1.62 },
  { id: "mars", label: "Mars", value: 3.71 },
  { id: "jupiter", label: "Jupiter", value: 24.79 },
] as const;

export const AIR_PRESETS = [
  { id: "earth", label: "Earth", value: 1 },
  { id: "mars", label: "Mars", value: 0.01 },
  { id: "vacuum", label: "Vacuum", value: 0 },
] as const;

export type ControlValues = Record<string, number | string>;

export interface ControlSpec {
  id: string;
  label: string;
  unit: string;
  type: "range" | "choice";
  min: number;
  max: number;
  step: number;
  decimals: number;
  default: number | string;
  options?: { value: string; label: string }[];
  presets?: { label: string; value: number }[];
  /** A one-line note under the control (what it does, or what it does not do). */
  help?: string;
}

export interface Scenario {
  model: SimModel;
  params: Params;
  init: Params;
  tEnd: number;
  extraEvents: ModelEvent[];
  /** Geometry the renderer needs. */
  geometry: { alpha?: number; sMax?: number; length?: number; markerRadius?: number };
}

const DEG = Math.PI / 180;

const HAS_AIR = new Set(["P2", "P3", "FF1", "FF2", "B2", "S3"]);

function g0(ctx: SimContext): number {
  return ctx.params.g ?? ctx.fixed.g ?? 9.81;
}

function rampShapeOf(ctx: SimContext): RampShape {
  if (ctx.model_ref === "R0" || ctx.model_ref === "R1") return "block";
  const beta = ctx.params.beta ?? ctx.fixed.beta ?? 0.4;
  if (Math.abs(beta - 2 / 3) < 0.05) return "hollow_ball";
  if (Math.abs(beta - 0.5) < 0.05) return "solid_cylinder";
  return "solid_ball";
}

export function controlsFor(ctx: SimContext): ControlSpec[] {
  const gravity: ControlSpec = {
    id: "g",
    label: "Gravity",
    unit: "m/s²",
    type: "range",
    min: 0.5,
    max: 30,
    step: 0.01,
    decimals: 2,
    default: g0(ctx),
    presets: GRAVITY_PRESETS.map((p) => ({ label: p.label, value: p.value })),
  };
  const air: ControlSpec = {
    id: "air",
    label: "Air density",
    unit: "× Earth",
    type: "range",
    min: 0,
    max: 2,
    step: 0.01,
    decimals: 2,
    default: 1,
    presets: AIR_PRESETS.map((p) => ({ label: p.label, value: p.value })),
    help: "Drag coefficients scale with the density of the air.",
  };
  const list: ControlSpec[] = [gravity];

  switch (ctx.system) {
    case "pendulum": {
      list.push({
        id: "length_cm",
        label: "String length",
        unit: "cm",
        type: "range",
        min: 10,
        max: 200,
        step: 1,
        decimals: 0,
        default: (ctx.fixed.L_eff ?? 0.6) * 100,
      });
      list.push({
        id: "amplitude_deg",
        label: "Starting angle",
        unit: "°",
        type: "range",
        min: 1,
        max: 90,
        step: 1,
        decimals: 0,
        default: Math.abs(ctx.initial.theta0 ?? 0.3) / DEG,
      });
      if (HAS_AIR.has(ctx.model_ref)) {
        list.push(air);
        list.push({
          id: "mass",
          label: "Bob mass",
          unit: "× yours",
          type: "range",
          min: 0.25,
          max: 4,
          step: 0.05,
          decimals: 2,
          default: 1,
          help: "Mass does not change the period. Damping was fitted as b/m, so a heavier bob of the same size damps more slowly.",
        });
      }
      break;
    }
    case "free_fall": {
      list.push({
        id: "height_m",
        label: "Drop height",
        unit: "m",
        type: "range",
        min: 0.2,
        max: 5,
        step: 0.05,
        decimals: 2,
        default: ctx.initial.y0 ?? 1.2,
      });
      if (HAS_AIR.has(ctx.model_ref)) {
        list.push(air);
        list.push({
          id: "mass",
          label: "Mass",
          unit: "× yours",
          type: "range",
          min: 0.25,
          max: 4,
          step: 0.05,
          decimals: 2,
          default: 1,
          help: "Drag was fitted as a rate per unit mass, so a heavier object of the same size feels less of it.",
        });
      }
      break;
    }
    case "ramp": {
      list.push({
        id: "angle_deg",
        label: "Ramp angle",
        unit: "°",
        type: "range",
        min: 5,
        max: 60,
        step: 0.5,
        decimals: 1,
        default: (ctx.fixed.alpha ?? ctx.params.alpha ?? 0.35) / DEG,
      });
      if (ctx.model_ref === "R1") {
        list.push({
          id: "mu_k",
          label: "Friction coefficient μ",
          unit: "",
          type: "range",
          min: 0,
          max: 1,
          step: 0.01,
          decimals: 2,
          default: ctx.params.mu_k ?? 0.3,
        });
      }
      list.push({
        id: "shape",
        label: "Shape",
        unit: "",
        type: "choice",
        min: 0,
        max: 0,
        step: 0,
        decimals: 0,
        default: rampShapeOf(ctx),
        options: [
          { value: "block", label: "Block" },
          { value: "solid_ball", label: "Solid ball" },
          { value: "hollow_ball", label: "Hollow ball" },
          { value: "solid_cylinder", label: "Solid cylinder" },
        ],
        help: "Rolling shapes share the pull of gravity with their spin: a solid ball gets 5/7 of the block's acceleration when friction is zero.",
      });
      break;
    }
    case "bounce": {
      list.push({
        id: "height_m",
        label: "Drop height",
        unit: "m",
        type: "range",
        min: 0.2,
        max: 3,
        step: 0.05,
        decimals: 2,
        default: ctx.initial.y0 ?? 1,
      });
      list.push({
        id: "e",
        label: "Bounciness e",
        unit: "",
        type: "range",
        min: 0.1,
        max: 0.99,
        step: 0.01,
        decimals: 2,
        default: restitutionAtMedian(ctx),
        help: "The fraction of speed kept after each bounce.",
      });
      if (HAS_AIR.has(ctx.model_ref)) list.push(air);
      break;
    }
    case "spin_down": {
      list.push({
        id: "spin_rev_s",
        label: "Starting spin",
        unit: "rev/s",
        type: "range",
        min: 0.5,
        max: 20,
        step: 0.1,
        decimals: 1,
        default: (ctx.initial.omega0 ?? 20) / (2 * Math.PI),
      });
      if (HAS_AIR.has(ctx.model_ref)) list.push(air);
      // Gravity matters here only through the bearing load.
      list[0] = {
        ...gravity,
        help: "Assumes bearing friction scales with g, because the load is the spinner's own weight.",
      };
      break;
    }
  }
  return list;
}

export function defaultControls(ctx: SimContext): ControlValues {
  const out: ControlValues = {};
  for (const c of controlsFor(ctx)) out[c.id] = c.default;
  return out;
}

function restitutionAtMedian(ctx: SimContext): number {
  if (ctx.model_ref === "B1") {
    const v = Math.sqrt(2 * g0(ctx) * (ctx.initial.y0 ?? 1));
    return Math.max(0, (ctx.params.e0 ?? 0.8) - (ctx.params.e1 ?? 0) * v);
  }
  return ctx.params.e ?? 0.8;
}

function num(values: ControlValues, id: string, fallback: number): number {
  const v = values[id];
  return typeof v === "number" && Number.isFinite(v) ? v : fallback;
}

/** Build the model, parameters and initial state for a set of control values. */
export function buildScenario(ctx: SimContext, values: ControlValues): Scenario {
  const base = getModel(ctx.model_ref);
  if (!base) throw new Error(`No TypeScript twin for model ${ctx.model_ref}`);
  const d = defaultControls(ctx);
  const get = (id: string) => num(values, id, num(d, id, 0));

  const params: Params = { ...ctx.fixed, ...ctx.params };
  const init: Params = { ...ctx.initial };
  const extra: ModelEvent[] = [];
  let model = base;
  let tEnd = 12;
  const geometry: Scenario["geometry"] = {};

  const g = get("g");
  params.g = g;
  const airScale = values.air === undefined ? 1 : get("air");

  switch (ctx.system) {
    case "pendulum": {
      const L = get("length_cm") / 100;
      params.L_eff = L;
      geometry.length = L;
      const massScale = 1 / Math.max(0.05, get("mass"));
      if (base.paramNames.includes("gamma")) params.gamma = (ctx.params.gamma ?? 0) * airScale * massScale;
      if (base.paramNames.includes("kappa")) params.kappa = (ctx.params.kappa ?? 0) * airScale * massScale;
      const amp = get("amplitude_deg");
      const defAmp = num(d, "amplitude_deg", amp);
      if (Math.abs(amp - defAmp) > 1e-9) {
        // A new starting angle means a new release from rest.
        init.theta0 = Math.sign(ctx.initial.theta0 || 1) * amp * DEG;
        init.omega0 = 0;
      }
      const T = smallAnglePeriod(g, L);
      tEnd = clamp(8 * T, 8, 40);
      break;
    }
    case "free_fall": {
      const h = get("height_m");
      const defH = num(d, "height_m", h);
      init.y0 = h;
      if (Math.abs(h - defH) > 1e-9) init.v0 = 0;
      const massScale = 1 / Math.max(0.05, get("mass"));
      if (base.paramNames.includes("b_over_m")) params.b_over_m = (ctx.params.b_over_m ?? 0) * airScale * massScale;
      if (base.paramNames.includes("k_over_m")) params.k_over_m = (ctx.params.k_over_m ?? 0) * airScale * massScale;
      extra.push(floorEvent(0));
      tEnd = clamp(Math.sqrt((2 * h) / g) * 1.6 + 0.3, 0.6, 12);
      break;
    }
    case "ramp": {
      const alpha = get("angle_deg") * DEG;
      params.alpha = alpha;
      geometry.alpha = alpha;
      const sMax = ctx.fixed.s_max ?? 1.2;
      geometry.sMax = sMax;
      extra.push({
        kind: "stop",
        fn: (s) => sMax - s[0],
        apply: (s) => ({ state: [sMax, s[1]], stop: true }),
      });
      const shape = (values.shape as RampShape | undefined) ?? rampShapeOf(ctx);
      if (shape === "block") {
        model = getModel("R1") ?? base;
        params.mu_k =
          values.mu_k !== undefined ? get("mu_k") : (ctx.params.mu_k ?? (ctx.model_ref === "R0" ? 0 : 0.3));
      } else {
        model = getModel(ctx.params.c_rr !== undefined ? "R3" : "R2") ?? base;
        params.beta = SHAPE_BETA[shape];
        params.c_rr = ctx.params.c_rr ?? 0;
      }
      const a = Math.max(0.05, g * Math.sin(alpha) * 0.3);
      tEnd = clamp(Math.sqrt((2 * sMax) / a) * 1.4, 1.5, 20);
      break;
    }
    case "bounce": {
      const h = get("height_m");
      init.y0 = h;
      init.v0 = 0;
      const e = get("e");
      if (model.id === "B1") {
        const v = Math.sqrt(2 * g * h);
        params.e0 = e + (params.e1 ?? 0) * v;
      } else {
        params.e = e;
      }
      if (base.paramNames.includes("k_over_m")) params.k_over_m = (ctx.params.k_over_m ?? 0) * airScale;
      tEnd = clamp(Math.sqrt((2 * h) / g) * (1 + 2 * e / Math.max(0.05, 1 - e)) + 0.5, 1.5, 20);
      break;
    }
    case "spin_down": {
      const w0 = get("spin_rev_s") * 2 * Math.PI;
      init.omega0 = w0;
      init.phi0 = 0;
      const gRef = ctx.fixed.g ?? 9.81;
      if (base.paramNames.includes("c")) params.c = (ctx.params.c ?? 0) * (g / gRef);
      if (base.paramNames.includes("k")) params.k = (ctx.params.k ?? 0) * airScale;
      geometry.markerRadius = ctx.fixed.marker_radius_m ?? ctx.object.radius_m;
      const stop = spinStopTime(model.id as "S0" | "S1" | "S2" | "S3", params, w0);
      tEnd = Number.isFinite(stop) ? clamp(stop * 1.15 + 1, 3, 90) : clamp(4 / Math.max(params.b, 0.01), 5, 90);
      break;
    }
  }
  return { model, params, init, tEnd, extraEvents: extra, geometry };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

export function runScenario(sc: Scenario, dt?: number): SimResult {
  return integrate(sc.model, sc.params, sc.init, {
    tEnd: sc.tEnd,
    dt,
    extraEvents: sc.extraEvents,
  });
}

/* ------------------------------------------------------------------------------------ */
/* Readouts                                                                              */
/* ------------------------------------------------------------------------------------ */

export interface Readout {
  id: string;
  label: string;
  value: number | null;
  unit: string;
  decimals: number;
}

/** Mean time between upward zero crossings of a series, by linear interpolation. */
export function measuredPeriod(t: readonly number[], y: readonly number[]): number | null {
  const crossings: number[] = [];
  for (let i = 1; i < y.length; i++) {
    if (y[i - 1] < 0 && y[i] >= 0) {
      const f = -y[i - 1] / (y[i] - y[i - 1]);
      crossings.push(t[i - 1] + f * (t[i] - t[i - 1]));
    }
  }
  if (crossings.length < 2) return null;
  return (crossings[crossings.length - 1] - crossings[0]) / (crossings.length - 1);
}

export function computeReadouts(sc: Scenario, res: SimResult): Readout[] {
  const m = sc.model;
  const p = sc.params;
  switch (m.system) {
    case "pendulum": {
      const theta = res.obs[0];
      const period = measuredPeriod(res.t, theta);
      const amp0 = Math.abs(sc.init.theta0 ?? 0);
      // Time for the swing to fall to half its starting amplitude.
      let half: number | null = null;
      let peak = amp0;
      for (let i = 1; i < theta.length - 1; i++) {
        if (Math.abs(theta[i]) >= Math.abs(theta[i - 1]) && Math.abs(theta[i]) > Math.abs(theta[i + 1])) {
          peak = Math.abs(theta[i]);
          if (peak <= amp0 / 2) {
            half = res.t[i];
            break;
          }
        }
      }
      const lastT = res.t[res.t.length - 1];
      let finalPeak = amp0;
      for (let i = theta.length - 2; i > 0; i--) {
        if (Math.abs(theta[i]) >= Math.abs(theta[i - 1]) && Math.abs(theta[i]) > Math.abs(theta[i + 1])) {
          finalPeak = Math.abs(theta[i]);
          break;
        }
      }
      return [
        { id: "period", label: "Period", value: period, unit: "s", decimals: 3 },
        { id: "half_life", label: "Swing halves in", value: half, unit: "s", decimals: 1 },
        {
          id: "final_amp",
          label: `Swing after ${lastT.toFixed(0)} s`,
          value: finalPeak / DEG,
          unit: "°",
          decimals: 1,
        },
      ];
    }
    case "free_fall": {
      const floor = res.events.find((e) => e.kind === "floor");
      const terminal =
        m.id === "FF1" && p.b_over_m > 1e-9
          ? p.g / p.b_over_m
          : m.id === "FF2" && p.k_over_m > 1e-9
            ? Math.sqrt(p.g / p.k_over_m)
            : null;
      return [
        { id: "fall_time", label: "Time to hit the floor", value: floor?.t ?? null, unit: "s", decimals: 3 },
        {
          id: "impact_speed",
          label: "Impact speed",
          value: floor ? Math.abs(floor.state[1]) : null,
          unit: "m/s",
          decimals: 2,
        },
        { id: "terminal", label: "Terminal velocity", value: terminal, unit: "m/s", decimals: 1 },
      ];
    }
    case "ramp": {
      const stop = res.stoppedAt;
      const last = res.states[res.states.length - 1];
      const reached = stop !== null && last[0] >= (sc.geometry.sMax ?? Infinity) - 1e-6;
      const a = rampAccelOf(sc);
      return [
        { id: "accel", label: "Acceleration", value: a, unit: "m/s²", decimals: 2 },
        { id: "time", label: "Time to reach the bottom", value: reached ? stop : null, unit: "s", decimals: 2 },
        { id: "speed", label: "Speed at the bottom", value: reached ? last[1] : null, unit: "m/s", decimals: 2 },
      ];
    }
    case "bounce": {
      const impacts = res.events.filter((e) => e.kind === "impact");
      // Apex height after each impact from the rebound speed: v^2 / 2g (drag ignored).
      const apex = (k: number) => {
        const ev = impacts[k];
        if (!ev) return null;
        const v = ev.state[1];
        return (v * v) / (2 * p.g);
      };
      const stop = res.stoppedAt;
      return [
        { id: "first", label: "First impact", value: impacts[0]?.t ?? null, unit: "s", decimals: 3 },
        { id: "h1", label: "First rebound height", value: apex(0), unit: "m", decimals: 3 },
        { id: "h3", label: "Third rebound height", value: apex(2), unit: "m", decimals: 3 },
        { id: "rest", label: "Comes to rest after", value: stop, unit: "s", decimals: 1 },
      ];
    }
    case "spin_down": {
      const phi = res.states[res.states.length - 1][0];
      const stop = res.stoppedAt;
      let half: number | null = null;
      const w0 = sc.init.omega0 ?? 0;
      for (let i = 0; i < res.states.length; i++) {
        if (res.states[i][1] <= w0 / 2) {
          half = res.t[i];
          break;
        }
      }
      return [
        { id: "stop", label: "Time to stop", value: stop, unit: "s", decimals: 1 },
        { id: "half", label: "Spin halves in", value: half, unit: "s", decimals: 1 },
        { id: "revs", label: "Turns before stopping", value: phi / (2 * Math.PI), unit: "rev", decimals: 0 },
      ];
    }
  }
}

function rampAccelOf(sc: Scenario): number {
  const p = sc.params;
  const sa = Math.sin(p.alpha);
  const ca = Math.cos(p.alpha);
  switch (sc.model.id) {
    case "R0":
      return p.g * sa;
    case "R1":
      return p.g * (sa - p.mu_k * ca);
    case "R2":
      return (p.g * sa) / (1 + p.beta);
    default:
      return (p.g * (sa - (p.c_rr ?? 0) * ca)) / (1 + p.beta);
  }
}

/** The exact undamped period, for the pendulum "T grows as sqrt(L/g)" explanation. */
export function undampedPeriod(sc: Scenario): number | null {
  if (sc.model.system !== "pendulum") return null;
  return exactPeriod(sc.params.g, sc.params.L_eff, sc.init.theta0 ?? 0.1);
}

/* ------------------------------------------------------------------------------------ */
/* Extrapolation guardrail                                                               */
/* ------------------------------------------------------------------------------------ */

const WARNING_LABELS: Record<string, { what: string; unit: string }> = {
  angle_deg: { what: "was measured at", unit: "°" },
  amplitude_deg: { what: "was measured at amplitudes of", unit: "°" },
  height_m: { what: "was measured from heights of", unit: " m" },
  length_cm: { what: "was measured with lengths of", unit: " cm" },
  spin_rev_s: { what: "was measured at spins of", unit: " rev/s" },
  e: { what: "was measured at e =", unit: "" },
};

/** Warn when a slider leaves the range the experiment actually measured. */
export function extrapolationWarnings(ctx: SimContext, values: ControlValues): string[] {
  const out: string[] = [];
  for (const [id, range] of Object.entries(ctx.measured_range)) {
    const v = values[id];
    if (typeof v !== "number") continue;
    if (v < range[0] - 1e-9 || v > range[1] + 1e-9) {
      const label = WARNING_LABELS[id] ?? { what: "was measured at", unit: "" };
      out.push(
        `This is outside what you measured. The fit ${label.what} ${range[0]}–${range[1]}${label.unit}, so treat the answer as a prediction, not a measurement.`,
      );
    }
  }
  return out;
}
