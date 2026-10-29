import type { ModelEvent, Params, SimModel } from "../types";

/**
 * Spin-down of a rotor (build-plan Part 10). State is [phi, omega]; the observable is the
 * unwrapped angle phi. All four models have closed forms and omega clamps at zero.
 *
 *   S0  viscous         omega' = -b omega            (exponential, never quite stops)
 *   S1  Coulomb         omega' = -c                  (linear)
 *   S2  both            omega' = -c - b omega
 *   S3  Coulomb + air   omega' = -c - k omega^2
 *
 * b is in 1/s, c in rad/s^2 and k in s/rad. The spinner never reverses: the clamp ends the
 * simulation at the instant omega reaches zero (a stop event), and phi holds its final value.
 */

const clampEvent: ModelEvent = {
  kind: "stop",
  fn: (s) => s[1],
  apply: (s) => ({ state: [s[0], 0], stop: true }),
};

const base = {
  system: "spin_down" as const,
  initialNames: ["phi0", "omega0"],
  observableNames: ["phi"],
  stateNames: ["phi", "omega"],
  initialState: (init: Params) => [init.phi0 ?? 0, init.omega0 ?? 0],
  observe: (s: readonly number[]) => [s[0]],
  events: [clampEvent],
};

function decel(model: SpinId, w: number, p: Params): number {
  switch (model) {
    case "S0":
      return p.b * w;
    case "S1":
      return p.c;
    case "S2":
      return p.c + p.b * w;
    case "S3":
      return p.c + p.k * w * w;
  }
}

type SpinId = "S0" | "S1" | "S2" | "S3";
type Closed = (t: number, p: Params, w0: number) => { phi: number; omega: number };

function coulomb(t: number, p: Params, w0: number) {
  const tStop = w0 / p.c;
  const tt = Math.min(t, tStop);
  return { omega: Math.max(0, w0 - p.c * t), phi: w0 * tt - 0.5 * p.c * tt * tt };
}

/** Closed forms for omega(t) and the angle turned, from an initial spin w0 > 0. */
const CLOSED: Record<SpinId, Closed> = {
  S0: (t, p, w0) => ({
    omega: w0 * Math.exp(-p.b * t),
    phi: p.b > 1e-12 ? (w0 / p.b) * (1 - Math.exp(-p.b * t)) : w0 * t,
  }),
  S1: coulomb,
  S2: (t, p, w0) => {
    if (p.b < 1e-12) return coulomb(t, p, w0);
    const tStop = Math.log(1 + (p.b * w0) / p.c) / p.b;
    const tt = Math.min(t, tStop);
    const r = p.c / p.b;
    const omega = t >= tStop ? 0 : (w0 + r) * Math.exp(-p.b * t) - r;
    const phi = ((w0 + r) / p.b) * (1 - Math.exp(-p.b * tt)) - r * tt;
    return { omega, phi };
  },
  S3: (t, p, w0) => {
    if (p.k < 1e-12) return coulomb(t, p, w0);
    const a = Math.sqrt(p.c / p.k); // omega scale
    const b = Math.sqrt(p.c * p.k);
    const th0 = Math.atan(w0 / a);
    const tStop = th0 / b;
    const tt = Math.min(t, tStop);
    const omega = t >= tStop ? 0 : a * Math.tan(th0 - b * t);
    // phi = integral of a tan(th0 - b t) dt = (a/b) ln(cos(th0 - b t) / cos(th0))
    const phi = (a / b) * Math.log(Math.cos(th0 - b * tt) / Math.cos(th0));
    return { omega, phi };
  },
};

function make(id: SpinId, name: string, paramNames: string[]): SimModel {
  return {
    ...base,
    id,
    name,
    paramNames,
    // No clamp in the right-hand side: the stop event ends the simulation at omega = 0, and a
    // rotor released at rest triggers it on the first step.
    rhs: (_t, s, p, out) => {
      out[0] = s[1];
      out[1] = -decel(id, s[1], p);
    },
    closedForm: (t, p, init) => [(init.phi0 ?? 0) + CLOSED[id](t, p, init.omega0 ?? 0).phi],
  };
}

export const S0 = make("S0", "Viscous friction", ["b"]);
export const S1 = make("S1", "Coulomb friction", ["c"]);
export const S2 = make("S2", "Coulomb and viscous friction", ["c", "b"]);
export const S3 = make("S3", "Coulomb friction and air drag", ["c", "k"]);

/** Closed-form angular velocity (for tests and plots). */
export function spinOmega(model: SpinId, t: number, p: Params, w0: number): number {
  return CLOSED[model](t, p, w0).omega;
}

/** Time at which the rotor stops, or Infinity for pure viscous friction. */
export function spinStopTime(model: SpinId, p: Params, w0: number): number {
  switch (model) {
    case "S0":
      return Infinity;
    case "S1":
      return w0 / p.c;
    case "S2":
      return p.b < 1e-12 ? w0 / p.c : Math.log(1 + (p.b * w0) / p.c) / p.b;
    case "S3":
      return p.k < 1e-12 ? w0 / p.c : Math.atan(w0 * Math.sqrt(p.k / p.c)) / Math.sqrt(p.c * p.k);
  }
}
