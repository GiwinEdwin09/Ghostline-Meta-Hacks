import type { ModelEvent, Params, SimModel } from "../types";

/**
 * Ramp, with s measured along the incline, downhill positive (build-plan Part 10).
 * State is [s, v].
 *
 *   R0  frictionless slide         s'' = g sin(alpha)
 *   R1  kinetic friction           s'' = g (sin(alpha) - mu_k cos(alpha))
 *   R2  rolling without slipping   s'' = g sin(alpha) / (1 + beta)
 *   R3  R2 + rolling resistance    s'' = g (sin(alpha) - c_rr cos(alpha)) / (1 + beta)
 *
 * beta is the shape factor: 2/5 for a solid sphere, 2/3 for a hollow sphere and 1/2 for a
 * solid cylinder. Friction cannot push the body uphill, so when the net acceleration is
 * negative the body decelerates to rest and stays there (a stop event at v = 0).
 */

export const SHAPE_BETA = {
  block: 0,
  solid_ball: 2 / 5,
  hollow_ball: 2 / 3,
  solid_cylinder: 1 / 2,
} as const;
export type RampShape = keyof typeof SHAPE_BETA;

const base = {
  system: "ramp" as const,
  initialNames: ["s0", "v0"],
  observableNames: ["s"],
  stateNames: ["s", "v"],
  initialState: (init: Params) => [init.s0 ?? 0, init.v0 ?? 0],
  observe: (s: readonly number[]) => [s[0]],
};

const restEvent: ModelEvent = {
  kind: "stop",
  fn: (s) => s[1],
  apply: (s) => ({ state: [s[0], 0], stop: true }),
};

/** Acceleration while moving downhill (negative means friction wins and the body stops). */
function accel(model: "R0" | "R1" | "R2" | "R3", p: Params): number {
  const sa = Math.sin(p.alpha);
  const ca = Math.cos(p.alpha);
  switch (model) {
    case "R0":
      return p.g * sa;
    case "R1":
      return p.g * (sa - p.mu_k * ca);
    case "R2":
      return (p.g * sa) / (1 + p.beta);
    case "R3":
      return (p.g * (sa - p.c_rr * ca)) / (1 + p.beta);
  }
}

function make(
  id: "R0" | "R1" | "R2" | "R3",
  name: string,
  paramNames: string[],
): SimModel {
  return {
    ...base,
    id,
    name,
    paramNames,
    // A body that friction holds (or brings) to rest hits the v = 0 stop event and stays put.
    rhs: (_t, s, p, out) => {
      out[0] = s[1];
      out[1] = accel(id, p);
    },
    closedForm: (t, p, init) => {
      const a = accel(id, p);
      const v0 = init.v0 ?? 0;
      const s0 = init.s0 ?? 0;
      if (a >= 0 || v0 <= 0) {
        return [a >= 0 ? s0 + v0 * t + 0.5 * a * t * t : s0];
      }
      const tStop = -v0 / a;
      const tt = Math.min(t, tStop);
      return [s0 + v0 * tt + 0.5 * a * tt * tt];
    },
    events: [restEvent],
  };
}

export const R0 = make("R0", "Frictionless slide", ["g", "alpha"]);
export const R1 = make("R1", "Kinetic friction", ["g", "alpha", "mu_k"]);
export const R2 = make("R2", "Rolling without slipping", ["g", "alpha", "beta"]);
export const R3 = make("R3", "Rolling with rolling resistance", ["g", "alpha", "beta", "c_rr"]);

/** Constant acceleration down the incline for a given model and parameters (m/s^2). */
export function rampAcceleration(model: "R0" | "R1" | "R2" | "R3", p: Params): number {
  return accel(model, p);
}
