import type { Params, SimModel } from "../types";

/**
 * Pendulum in the angle theta (build-plan Part 10). State is [theta, omega].
 *
 *   P0  small-angle SHM                 theta'' = -(g/L_eff) theta        (closed form)
 *   P1  nonlinear                       theta'' = -(g/L_eff) sin(theta)
 *   P2  P1 + viscous damping            ... - gamma theta'
 *   P3  P1 + quadratic air drag         ... - kappa |theta'| theta'
 *   P4  P1 + Coulomb pivot friction     ... - mu_p tanh(theta'/eps)
 *
 * `gamma` is in 1/s, `kappa` in 1/rad, `mu_p` in rad/s^2 and `eps` (the width of the
 * smoothed Coulomb step) in rad/s.
 */

const base = {
  system: "pendulum" as const,
  initialNames: ["theta0", "omega0"],
  observableNames: ["theta"],
  stateNames: ["theta", "omega"],
  initialState: (init: Params) => [init.theta0 ?? 0, init.omega0 ?? 0],
  observe: (s: readonly number[]) => [s[0]],
};

export const P0: SimModel = {
  ...base,
  id: "P0",
  name: "Small-angle oscillator",
  paramNames: ["g", "L_eff"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -(p.g / p.L_eff) * s[0];
  },
  closedForm: (t, p, init) => {
    const w = Math.sqrt(p.g / p.L_eff);
    const th0 = init.theta0 ?? 0;
    const om0 = init.omega0 ?? 0;
    return [th0 * Math.cos(w * t) + (om0 / w) * Math.sin(w * t)];
  },
};

export const P1: SimModel = {
  ...base,
  id: "P1",
  name: "Nonlinear pendulum",
  paramNames: ["g", "L_eff"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -(p.g / p.L_eff) * Math.sin(s[0]);
  },
};

export const P2: SimModel = {
  ...base,
  id: "P2",
  name: "Nonlinear pendulum with viscous damping",
  paramNames: ["g", "L_eff", "gamma"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -(p.g / p.L_eff) * Math.sin(s[0]) - p.gamma * s[1];
  },
};

export const P3: SimModel = {
  ...base,
  id: "P3",
  name: "Nonlinear pendulum with quadratic air drag",
  paramNames: ["g", "L_eff", "kappa"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -(p.g / p.L_eff) * Math.sin(s[0]) - p.kappa * Math.abs(s[1]) * s[1];
  },
};

export const P4: SimModel = {
  ...base,
  id: "P4",
  name: "Nonlinear pendulum with pivot friction",
  paramNames: ["g", "L_eff", "mu_p", "eps"],
  rhs: (_t, s, p, out) => {
    out[0] = s[1];
    out[1] = -(p.g / p.L_eff) * Math.sin(s[0]) - p.mu_p * Math.tanh(s[1] / (p.eps || 0.02));
  },
};

/** Effective length of a spherical bob of radius r on a string of length L (Part 9). */
export function effectiveLength(L: number, r: number): number {
  return L + (0.4 * r * r) / L;
}

/** Small-angle period 2 pi sqrt(L/g). */
export function smallAnglePeriod(g: number, L: number): number {
  return 2 * Math.PI * Math.sqrt(L / g);
}

/** Exact undamped period 4 sqrt(L/g) K(sin(theta0/2)), via the arithmetic-geometric mean. */
export function exactPeriod(g: number, L: number, theta0: number): number {
  const k = Math.sin(Math.abs(theta0) / 2);
  let a = 1;
  let b = Math.sqrt(1 - k * k);
  for (let i = 0; i < 40 && Math.abs(a - b) > 1e-15; i++) {
    const an = 0.5 * (a + b);
    b = Math.sqrt(a * b);
    a = an;
  }
  const K = Math.PI / (2 * a);
  return 4 * Math.sqrt(L / g) * K;
}
