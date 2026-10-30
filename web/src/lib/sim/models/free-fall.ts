import type { ModelEvent, Params, SimModel } from "../types";

/**
 * Free fall, vertical (build-plan Part 10). State is [y, v] with y up.
 *
 *   FF0  constant g                     y'' = -g
 *   FF1  linear drag                    y'' = -g - b y'
 *   FF2  quadratic drag                 y'' = -g - k |y'| y'
 *
 * Names and equations are those of `physics/library/freefall.py`: `b` is the drag rate b/m in
 * 1/s and `k` the drag per unit mass k/m in 1/m. Initial conditions are `y0` and `vy0`. FF1 and
 * FF2 also report a terminal velocity, g/b and sqrt(g/k).
 */

const base = {
  system: "free_fall" as const,
  initialNames: ["y0", "vy0"],
  observableNames: ["y"],
  stateNames: ["y", "v"],
  initialState: (init: Params) => [init.y0 ?? 0, init.vy0 ?? 0],
  observe: (s: readonly number[]) => [s[0]],
};

/** Optional hard floor: stop when y reaches the ground (used by the what-if, not by fits). */
export function floorEvent(level = 0): ModelEvent {
  return {
    kind: "floor",
    fn: (s) => s[0] - level,
    apply: () => ({ state: [level, 0], stop: true }),
  };
}

export const FF0: SimModel = {
  ...base,
  id: "FF0",
  name: "Constant gravity",
  paramNames: ["g"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -p.g;
  },
  closedForm: (t, p, init) => {
    const y0 = init.y0 ?? 0;
    const v0 = init.vy0 ?? 0;
    return [y0 + v0 * t - 0.5 * p.g * t * t];
  },
};

export const FF1: SimModel = {
  ...base,
  id: "FF1",
  name: "Linear drag",
  paramNames: ["g", "b"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -p.g - p.b * s[1];
  },
  closedForm: (t, p, init) => {
    const y0 = init.y0 ?? 0;
    const v0 = init.vy0 ?? 0;
    const lam = p.b;
    if (lam < 1e-12) return [y0 + v0 * t - 0.5 * p.g * t * t];
    const vt = p.g / lam; // terminal speed
    const decay = Math.exp(-lam * t);
    // v(t) = (v0 + vt) e^{-lam t} - vt, integrated.
    return [y0 - vt * t + ((v0 + vt) * (1 - decay)) / lam];
  },
};

export const FF2: SimModel = {
  ...base,
  id: "FF2",
  name: "Quadratic drag",
  paramNames: ["g", "k"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -p.g - p.k * Math.abs(s[1]) * s[1];
  },
  closedForm: (t, p, init) => {
    // Closed form for a drop from rest (v0 = 0), the case the benchmark films.
    const y0 = init.y0 ?? 0;
    const v0 = init.vy0 ?? 0;
    if (Math.abs(v0) > 1e-12 || p.k < 1e-12) {
      return [y0 + v0 * t - 0.5 * p.g * t * t];
    }
    const vt = Math.sqrt(p.g / p.k);
    return [y0 - (vt * vt / p.g) * Math.log(Math.cosh((p.g * t) / vt))];
  },
};

export function terminalVelocity(model: string, p: Params): number | null {
  if (model === "FF1" && p.b > 1e-12) return p.g / p.b;
  if (model === "FF2" && p.k > 1e-12) return Math.sqrt(p.g / p.k);
  return null;
}
