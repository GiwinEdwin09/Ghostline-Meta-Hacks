import type { IntegrateOptions, Params, SimEventRecord, SimModel, SimResult } from "./types";

export const DEFAULT_DT = 1 / 240;

/** One classical Runge-Kutta step of size h from (t, s). */
export function rk4Step(
  model: SimModel,
  t: number,
  s: readonly number[],
  h: number,
  p: Params,
): number[] {
  const n = s.length;
  const k1 = new Array<number>(n).fill(0);
  const k2 = new Array<number>(n).fill(0);
  const k3 = new Array<number>(n).fill(0);
  const k4 = new Array<number>(n).fill(0);
  const tmp = new Array<number>(n).fill(0);

  model.rhs(t, s, p, k1);
  for (let i = 0; i < n; i++) tmp[i] = s[i] + 0.5 * h * k1[i];
  model.rhs(t + 0.5 * h, tmp, p, k2);
  for (let i = 0; i < n; i++) tmp[i] = s[i] + 0.5 * h * k2[i];
  model.rhs(t + 0.5 * h, tmp, p, k3);
  for (let i = 0; i < n; i++) tmp[i] = s[i] + h * k3[i];
  model.rhs(t + h, tmp, p, k4);

  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = s[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
  return out;
}

const MAX_EVENTS_PER_STEP = 64;
const BISECTION_ITERATIONS = 48;

/**
 * Fixed-step RK4 with event handling.
 *
 * The grid is uniform (default dt = 1/240 s), but events are located exactly: when an event
 * function goes from non-negative to non-positive inside a step, the step fraction is found by
 * bisection, the state is advanced to that instant, the model's reset map is applied (a
 * bounce flips and scales the velocity, the spin-down clamp stops the rotor) and the rest of
 * the step is integrated from there. A terminal event holds the final state for the rest of
 * the grid, so samples after a stop stay finite and flat.
 */
export function integrate(
  model: SimModel,
  p: Params,
  init: Params,
  options: IntegrateOptions,
): SimResult {
  const dt = options.dt ?? DEFAULT_DT;
  const events = [...(model.events ?? []), ...(options.extraEvents ?? [])];
  const nSteps = Math.max(1, Math.round(options.tEnd / dt));

  const t: number[] = [0];
  let s = model.initialState(init);
  const states: number[][] = [s.slice()];
  const obs: number[][] = model.observableNames.map(() => []);
  model.observe(s).forEach((v, k) => obs[k].push(v));
  const eventLog: SimEventRecord[] = [];
  let stoppedAt: number | null = null;

  for (let i = 0; i < nSteps; i++) {
    const t0 = i * dt;
    if (stoppedAt === null) {
      let tt = t0;
      let remaining = dt;
      let guard = 0;
      while (remaining > 1e-12 && guard++ < MAX_EVENTS_PER_STEP && stoppedAt === null) {
        const next = rk4Step(model, tt, s, remaining, p);
        let hit: { index: number; fraction: number } | null = null;
        for (let e = 0; e < events.length; e++) {
          const before = events[e].fn(s, p);
          const after = events[e].fn(next, p);
          if (before >= 0 && after <= 0) {
            let lo = 0;
            let hi = 1;
            for (let it = 0; it < BISECTION_ITERATIONS; it++) {
              const mid = 0.5 * (lo + hi);
              const sm = rk4Step(model, tt, s, remaining * mid, p);
              if (events[e].fn(sm, p) > 0) lo = mid;
              else hi = mid;
            }
            // Earliest event inside the step wins.
            if (hit === null || hi < hit.fraction) hit = { index: e, fraction: hi };
          }
        }
        if (hit === null) {
          s = next;
          tt += remaining;
          remaining = 0;
        } else {
          const h1 = remaining * hit.fraction;
          const atEvent = rk4Step(model, tt, s, h1, p);
          const ev = events[hit.index];
          const applied = ev.apply(atEvent, p);
          tt += h1;
          remaining -= h1;
          s = applied.state;
          eventLog.push({ kind: ev.kind, t: tt, state: applied.state.slice() });
          if (applied.stop) stoppedAt = tt;
        }
      }
    }
    t.push((i + 1) * dt);
    states.push(s.slice());
    model.observe(s).forEach((v, k) => obs[k].push(v));
  }

  return { dt, t, states, obs, events: eventLog, stoppedAt };
}

/** Linear interpolation of a uniformly sampled series at time x. */
export function sampleAt(t: readonly number[], y: readonly number[], x: number): number {
  if (t.length === 0) return NaN;
  if (x <= t[0]) return y[0];
  const last = t.length - 1;
  if (x >= t[last]) return y[last];
  const dt = t[1] - t[0];
  const f = x / dt;
  const i = Math.min(last - 1, Math.max(0, Math.floor(f)));
  const w = f - i;
  return y[i] * (1 - w) + y[i + 1] * w;
}
