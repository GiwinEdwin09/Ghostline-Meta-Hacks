/**
 * Types shared by the TypeScript twins of the physics library (build-plan Part 10) and the
 * fixed-step RK4 integrator (Part 18).
 *
 * Conventions: SI units, angles in radians, `y` up. Parameter names follow the symbols in
 * Part 10 (`g`, `L_eff`, `gamma`, `kappa`, `mu_p`, `mu_k`, `beta`, `c_rr`, `e`, `e0`, `e1`,
 * `b`, `c`, `k`, `b_over_m`, `k_over_m`) so the Python library can use the same names.
 */

export type SimSystem = "free_fall" | "pendulum" | "ramp" | "bounce" | "spin_down";

export type Params = Record<string, number>;

export type EventKind = "impact" | "stop" | "floor";

export interface ModelEvent {
  kind: EventKind;
  /** Zero-crossing function. The event fires when it goes from > 0 to <= 0. */
  fn(s: readonly number[], p: Params): number;
  /** Applies the reset map at the event. Return `stop: true` to end the simulation. */
  apply(s: readonly number[], p: Params): { state: number[]; stop?: boolean };
}

export interface SimModel {
  /** Library id, for example "P2". */
  id: string;
  system: SimSystem;
  name: string;
  /** Physical parameters, in the order they are documented. */
  paramNames: string[];
  /** Initial-condition names. */
  initialNames: string[];
  /** What `observe` returns, for example ["theta"]. */
  observableNames: string[];
  /** Names of the state vector entries, for example ["theta", "omega"]. */
  stateNames: string[];
  /** ds/dt written into `out`. Pure: the same inputs always give the same output. */
  rhs(t: number, s: readonly number[], p: Params, out: number[]): void;
  initialState(init: Params): number[];
  observe(s: readonly number[]): number[];
  /** Closed-form observables at time t, when the model has one. */
  closedForm?(t: number, p: Params, init: Params): number[];
  /** Resets and terminal events: bounces, the omega -> 0 clamp, coming to rest. */
  events?: ModelEvent[];
}

export interface SimEventRecord {
  kind: EventKind;
  t: number;
  state: number[];
}

export interface SimResult {
  dt: number;
  /** Uniform sample times 0, dt, 2dt, ... */
  t: number[];
  /** states[i] is the state at t[i]. */
  states: number[][];
  /** obs[k][i] is observable k at t[i]. */
  obs: number[][];
  /** Exact event times, found by bisection inside the step that contains them. */
  events: SimEventRecord[];
  /** Time of the terminal event, or null when the simulation ran to tEnd. */
  stoppedAt: number | null;
}

/**
 * What the what-if simulator needs to know about an experiment. `WhatIfContext` in
 * `lib/types.ts` satisfies this structurally.
 */
export interface SimContext {
  system: SimSystem;
  /** Library id of the winning model, for example "P2". */
  model_ref: string;
  /** Fitted physical parameters by library name. */
  params: Params;
  /** Quantities held fixed in the fit (g, L_eff, alpha, ...). */
  fixed: Params;
  /** Trial 1's initial conditions. */
  initial: Params;
  /** Measured range per control id, for the extrapolation warning. */
  measured_range: Record<string, [number, number]>;
  object: { name: string; radius_m: number; mass_kg: number | null };
  anchor_m: [number, number];
}

export interface IntegrateOptions {
  tEnd: number;
  /** Fixed step. Defaults to 1/240 s (Part 18). */
  dt?: number;
  /** Extra events on top of the model's own, for example a floor for a free fall. */
  extraEvents?: ModelEvent[];
}
