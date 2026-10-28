import type { ModelEvent, Params, SimModel } from "../types";

/**
 * A ball bouncing on a floor at y = 0 (build-plan Part 10). State is [y, v].
 *
 *   B0  constant restitution e
 *   B1  speed-dependent restitution  e = max(0, e0 - e1 |v|)
 *   B2  flights with quadratic drag, plus constant e
 *
 * Each impact reverses the velocity and scales it by e. When the rebound speed falls under
 * REST_SPEED the ball is at rest on the floor (this also ends the infinite chain of ever
 * smaller bounces).
 */

export const REST_SPEED = 0.02; // m/s

function impact(restitution: (vIn: number, p: Params) => number): ModelEvent {
  return {
    kind: "impact",
    fn: (s) => s[0],
    apply: (s, p) => {
      const vIn = Math.abs(s[1]);
      const vOut = restitution(vIn, p) * vIn;
      if (vOut < REST_SPEED) return { state: [0, 0], stop: true };
      return { state: [0, vOut] };
    },
  };
}

const base = {
  system: "bounce" as const,
  initialNames: ["y0", "v0"],
  observableNames: ["y"],
  stateNames: ["y", "v"],
  initialState: (init: Params) => [init.y0 ?? 0, init.v0 ?? 0],
  observe: (s: readonly number[]) => [s[0]],
};

export const B0: SimModel = {
  ...base,
  id: "B0",
  name: "Constant restitution",
  paramNames: ["g", "e"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -p.g;
  },
  events: [impact((_v, p) => p.e)],
};

export const B1: SimModel = {
  ...base,
  id: "B1",
  name: "Speed-dependent restitution",
  paramNames: ["g", "e0", "e1"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -p.g;
  },
  events: [impact((v, p) => Math.max(0, p.e0 - p.e1 * v))],
};

export const B2: SimModel = {
  ...base,
  id: "B2",
  name: "Quadratic drag with constant restitution",
  paramNames: ["g", "e", "k_over_m"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -p.g - p.k_over_m * Math.abs(s[1]) * s[1];
  },
  events: [impact((_v, p) => p.e)],
};

/** Time of the first impact when dropped from rest at height h (no drag). */
export function firstImpactTime(g: number, h: number): number {
  return Math.sqrt((2 * h) / g);
}
