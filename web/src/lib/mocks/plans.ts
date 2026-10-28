import { getModel, integrate } from "@/lib/sim";
import type { Params } from "@/lib/sim";
import type { AnalysisInput, ModelPlan } from "./analysis";
import type { FreeParam } from "./fitting";
import { gaussian, mulberry32 } from "./rng";
import scenarios from "./scenarios.json";

/**
 * The synthetic measurements behind the mock experiments, and the competing models Wonder
 * would propose for them. Data come from the same simulator the what-if uses, plus seeded
 * noise, so every curve on every mock page is internally consistent.
 */

const DEG = Math.PI / 180;

export type ScenarioKey = keyof typeof scenarios;
export const SCENARIOS = scenarios;

function times(fps: number, windowS: number): number[] {
  const n = Math.round(windowS * fps) + 1;
  return Array.from({ length: n }, (_, i) => i / fps);
}

const fp = {
  phi0: (): FreeParam => ({ name: "phi0", init: 0, scale: 0.05, kind: "initial_condition", unit: "rad", label: "φ₀" }),
  omega0Spin: (init: number): FreeParam => ({ name: "omega0", init, scale: 5, lower: 1, kind: "initial_condition", unit: "rad/s", label: "ω₀" }),
  c: (init: number): FreeParam => ({ name: "c", init, scale: 0.2, lower: 1e-4, kind: "physical", unit: "rad/s²", label: "c (rubbing friction)" }),
  b: (init: number): FreeParam => ({ name: "b", init, scale: 0.02, lower: 0, kind: "physical", unit: "1/s", label: "b (viscous friction)" }),
  k: (init: number): FreeParam => ({ name: "k", init, scale: 2e-4, lower: 0, kind: "physical", unit: "s/rad", label: "k (air drag)" }),
  theta0: (init: number): FreeParam => ({ name: "theta0", init, scale: 0.05, kind: "initial_condition", unit: "rad", label: "θ₀" }),
  omega0Pend: (): FreeParam => ({ name: "omega0", init: 0, scale: 0.1, kind: "initial_condition", unit: "rad/s", label: "ω₀" }),
  gPend: (init: number): FreeParam => ({ name: "g", init, scale: 0.4, lower: 5, upper: 15, kind: "physical", unit: "m/s²", label: "g" }),
  gamma: (init: number): FreeParam => ({ name: "gamma", init, scale: 0.02, lower: 0, kind: "physical", unit: "1/s", label: "γ (damping)" }),
  kappa: (init: number): FreeParam => ({ name: "kappa", init, scale: 0.02, lower: 0, kind: "physical", unit: "1/rad", label: "κ (air drag)" }),
  y0: (init: number): FreeParam => ({ name: "y0", init, scale: 0.01, kind: "initial_condition", unit: "m", label: "y₀" }),
  v0: (): FreeParam => ({ name: "v0", init: 0, scale: 0.05, kind: "initial_condition", unit: "m/s", label: "v₀" }),
  gFall: (init: number): FreeParam => ({ name: "g", init, scale: 0.4, lower: 5, upper: 15, kind: "physical", unit: "m/s²", label: "g" }),
  bOverM: (init: number): FreeParam => ({ name: "b_over_m", init, scale: 0.05, lower: 0, kind: "physical", unit: "1/s", label: "b/m (linear drag rate)" }),
  kOverM: (init: number): FreeParam => ({ name: "k_over_m", init, scale: 0.02, lower: 0, kind: "physical", unit: "1/m", label: "k/m (quadratic drag)" }),
};

/* ------------------------------------------------------------------------------------ */
/* Spin-down                                                                             */
/* ------------------------------------------------------------------------------------ */

export function spinInput(): { input: AnalysisInput; truth: number[] } {
  const sc = scenarios.spin;
  const t = times(sc.fps, sc.window_s);
  const model = getModel("S2")!;
  const truthParams: Params = { c: sc.truth.c, b: sc.truth.b };
  const init: Params = { phi0: sc.truth.phi0, omega0: sc.truth.omega0 };
  const truth = t.map((x) => model.closedForm!(x, truthParams, init)[0]);
  const rng = mulberry32(sc.seed);
  const y = truth.map((v) => v + sc.sigma * gaussian(rng));

  const base = { fixed: {}, initBase: {} };
  const plans: ModelPlan[] = [
    {
      ...base,
      ref: "S0",
      prior_rank: 1,
      mechanism:
        "Friction proportional to speed, like a paddle in honey: the spinner loses the same fraction of its spin every second.",
      equations_latex: String.raw`\dot\omega = -b\,\omega \;\Rightarrow\; \omega(t)=\omega_0\,e^{-bt}`,
      refuting_signature:
        "ω(t) must fall by the same fraction each second, so log ω is a straight line. A constant-slope ω(t) that hits zero in finite time refutes it.",
      free: [fp.phi0(), fp.omega0Spin(50), fp.b(0.08)],
    },
    {
      ...base,
      ref: "S1",
      prior_rank: 3,
      mechanism:
        "Rubbing friction in the bearing that does not depend on speed: the spinner loses the same amount of spin every second.",
      equations_latex: String.raw`\dot\omega = -c \;\Rightarrow\; \omega(t)=\omega_0 - c\,t`,
      refuting_signature:
        "ω(t) is a straight line and the spinner stops abruptly at t = ω₀/c. Curvature in ω(t), steeper at first, refutes it.",
      free: [fp.phi0(), fp.omega0Spin(50), fp.c(1.1)],
    },
    {
      ...base,
      ref: "S2",
      prior_rank: 2,
      mechanism: "Both at once: a constant rubbing term plus a term that grows with speed.",
      equations_latex: String.raw`\dot\omega = -c - b\,\omega`,
      refuting_signature:
        "ω(t) is slightly curved, steeper at first. If the fitted b is indistinguishable from zero, the viscous term is not needed.",
      free: [fp.phi0(), fp.omega0Spin(52), fp.c(0.8), fp.b(0.04)],
    },
    {
      ...base,
      ref: "S3",
      prior_rank: 4,
      mechanism: "Rubbing friction plus air drag that grows with the square of the speed.",
      equations_latex: String.raw`\dot\omega = -c - k\,\omega^{2}`,
      refuting_signature:
        "Drag dominates only at high spin and fades quickly, so ω(t) bends the other way: sharply at first, then almost straight.",
      free: [fp.phi0(), fp.omega0Spin(52), fp.c(0.8), fp.k(5e-4)],
    },
  ];
  const range = Math.max(...y) - Math.min(...y);
  return { input: { t, y, sigma: sc.sigma, plans, unit: "rad", label: "Angle φ", range, digits: 3 }, truth };
}

/* ------------------------------------------------------------------------------------ */
/* Pendulum                                                                              */
/* ------------------------------------------------------------------------------------ */

function pendulumPlans(L: number, theta0: number, gInit: number): ModelPlan[] {
  const fixed = { L_eff: L };
  const initBase = {};
  return [
    {
      ref: "P1",
      prior_rank: 1,
      fixed,
      initBase,
      mechanism: "The exact pendulum with no losses: gravity pulling the bob back along a circle.",
      equations_latex: String.raw`\ddot\theta=-\frac{g}{L_\mathrm{eff}}\sin\theta`,
      refuting_signature:
        "The amplitude never shrinks. Swings that get steadily smaller, with a residual that grows like a damped oscillation, refute it.",
      free: [fp.theta0(theta0), fp.omega0Pend(), fp.gPend(gInit)],
    },
    {
      ref: "P2",
      prior_rank: 2,
      fixed,
      initBase,
      mechanism: "The exact pendulum plus drag proportional to speed (air and pivot).",
      equations_latex: String.raw`\ddot\theta=-\frac{g}{L_\mathrm{eff}}\sin\theta-\gamma\,\dot\theta`,
      refuting_signature:
        "The amplitude decays exponentially: every swing is the same fraction smaller than the one before. If γ fits to zero, damping is not needed.",
      free: [fp.theta0(theta0), fp.omega0Pend(), fp.gPend(gInit), fp.gamma(0.02)],
    },
    {
      ref: "P3",
      prior_rank: 4,
      fixed,
      initBase,
      mechanism: "The exact pendulum plus quadratic air drag, which bites hardest at the fastest part of the swing.",
      equations_latex: String.raw`\ddot\theta=-\frac{g}{L_\mathrm{eff}}\sin\theta-\kappa\,|\dot\theta|\,\dot\theta`,
      refuting_signature: "Big swings lose energy faster than small ones; the decay is not exponential.",
      free: [fp.theta0(theta0), fp.omega0Pend(), fp.gPend(gInit), fp.kappa(0.01)],
    },
    {
      ref: "P0",
      prior_rank: 3,
      fixed,
      initBase,
      mechanism:
        "The textbook small-angle pendulum, where the restoring force is proportional to the angle and the period does not depend on the amplitude.",
      equations_latex: String.raw`\ddot\theta=-\frac{g}{L_\mathrm{eff}}\,\theta`,
      refuting_signature:
        "The period is the same at every amplitude. At 38° a real pendulum is about 3% slower, so the phase drifts out of step within a few swings.",
      free: [fp.theta0(theta0), fp.omega0Pend(), fp.gPend(gInit)],
    },
  ];
}

export function pendulumInput(): { input: AnalysisInput; truth: number[] } {
  const sc = scenarios.pendulum;
  const t = times(sc.fps, sc.window_s);
  const model = getModel(sc.truth.model)!;
  const res = integrate(
    model,
    { g: sc.truth.g, L_eff: sc.truth.L_eff, gamma: sc.truth.gamma },
    { theta0: sc.truth.theta0_deg * DEG, omega0: 0 },
    { tEnd: sc.window_s },
  );
  const truth = t.map((x) => res.obs[0][Math.round(x * 240)]);
  const rng = mulberry32(sc.seed);
  const y = truth.map((v) => v + sc.sigma * gaussian(rng));
  const range = Math.max(...y) - Math.min(...y);
  return {
    input: {
      t,
      y,
      sigma: sc.sigma,
      plans: pendulumPlans(sc.truth.L_eff, sc.truth.theta0_deg * DEG, 9.6),
      unit: "rad",
      label: "Angle θ",
      range,
      digits: 3,
    },
    truth,
  };
}

/** The out-of-plane pendulum: the same swing, but the bob drifts toward and away from the camera. */
export function cantTellInput(): { input: AnalysisInput; truth: number[] } {
  const sc = scenarios.cant_tell;
  const t = times(sc.fps, sc.window_s);
  const model = getModel(sc.truth.model)!;
  const res = integrate(
    model,
    { g: sc.truth.g, L_eff: sc.truth.L_eff, gamma: sc.truth.gamma },
    { theta0: sc.truth.theta0_deg * DEG, omega0: 0 },
    { tEnd: sc.window_s },
  );
  const truth = t.map((x) => res.obs[0][Math.round(x * 240)]);
  const rng = mulberry32(sc.seed);
  // The perspective error multiplies the apparent angle by a slowly varying factor.
  const y = truth.map(
    (v, i) =>
      v * (1 + sc.out_of_plane.depth * Math.sin(sc.out_of_plane.omega * t[i] + 0.4)) + sc.sigma * gaussian(rng),
  );
  const range = Math.max(...y) - Math.min(...y);
  const plans = pendulumPlans(sc.truth.L_eff, sc.truth.theta0_deg * DEG, 9.6).filter((p) => p.ref !== "P3");
  return { input: { t, y, sigma: sc.sigma, plans, unit: "rad", label: "Angle θ", range, digits: 3 }, truth };
}

/* ------------------------------------------------------------------------------------ */
/* Free fall                                                                             */
/* ------------------------------------------------------------------------------------ */

export function freeFallInput(): { input: AnalysisInput; truth: number[] } {
  const sc = scenarios.free_fall;
  const t = times(sc.fps, sc.window_s);
  const model = getModel("FF1")!;
  const truth = t.map(
    (x) => model.closedForm!(x, { g: sc.truth.g, b_over_m: sc.truth.b_over_m }, { y0: sc.truth.y0, v0: 0 })[0],
  );
  const rng = mulberry32(sc.seed);
  const y = truth.map((v) => v + sc.sigma * gaussian(rng));
  const fixed = {};
  const plans: ModelPlan[] = [
    {
      ref: "FF0",
      prior_rank: 1,
      fixed,
      initBase: {},
      mechanism: "Gravity alone: a constant downward acceleration g, with no air.",
      equations_latex: String.raw`\ddot y=-g`,
      refuting_signature:
        "y(t) is an exact parabola. Systematic curvature left in the residuals, growing with speed, refutes it.",
      free: [fp.y0(1.19), fp.v0(), fp.gFall(9.5)],
    },
    {
      ref: "FF1",
      prior_rank: 2,
      fixed,
      initBase: {},
      mechanism: "Gravity plus air drag proportional to speed.",
      equations_latex: String.raw`\ddot y=-g-\frac{b}{m}\,\dot y`,
      refuting_signature: "The acceleration shrinks as the ball speeds up. If b/m fits to zero, drag is not needed.",
      free: [fp.y0(1.19), fp.v0(), fp.gFall(9.5), fp.bOverM(0.05)],
    },
    {
      ref: "FF2",
      prior_rank: 3,
      fixed,
      initBase: {},
      mechanism: "Gravity plus quadratic air drag, the usual law for a ball moving through air at speed.",
      equations_latex: String.raw`\ddot y=-g-\frac{k}{m}\,|\dot y|\,\dot y`,
      refuting_signature: "Drag matters only near the end of the fall, where the speed is highest.",
      free: [fp.y0(1.19), fp.v0(), fp.gFall(9.5), fp.kOverM(0.01)],
    },
  ];
  const range = Math.max(...y) - Math.min(...y);
  return { input: { t, y, sigma: sc.sigma, plans, unit: "m", label: "Height y", range, digits: 4 }, truth };
}
