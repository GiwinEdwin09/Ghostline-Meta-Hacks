import { describe, expect, it } from "vitest";

import {
  buildScenario,
  computeReadouts,
  controlsFor,
  defaultControls,
  extrapolationWarnings,
  instantExplanations,
  integrate,
  measuredPeriod,
  poseAt,
  runScenario,
} from "@/lib/sim";
import { B0, B1 } from "@/lib/sim/models/bounce";
import { FF0, FF1, FF2, floorEvent, terminalVelocity } from "@/lib/sim/models/free-fall";
import { P0, P1, P2, exactPeriod, smallAnglePeriod } from "@/lib/sim/models/pendulum";
import { R0, R1, R2, R3, rampAcceleration } from "@/lib/sim/models/ramp";
import { S0, S1, S2, S3, spinOmega, spinStopTime } from "@/lib/sim/models/spin";
import type { SimContext, SimModel } from "@/lib/sim";

const DEG = Math.PI / 180;

function maxAbsError(
  model: SimModel,
  params: Record<string, number>,
  init: Record<string, number>,
  tEnd: number,
  dt?: number,
): number {
  const res = integrate(model, params, init, { tEnd, dt });
  let worst = 0;
  for (let i = 0; i < res.t.length; i++) {
    const exact = model.closedForm!(res.t[i], params, init)[0];
    worst = Math.max(worst, Math.abs(res.obs[0][i] - exact));
  }
  return worst;
}

describe("RK4 against closed forms", () => {
  it("free fall FF0, FF1 and FF2 match their closed forms", () => {
    expect(maxAbsError(FF0, { g: 9.81 }, { y0: 2, v0: 1.5 }, 0.6)).toBeLessThan(1e-10);
    expect(maxAbsError(FF1, { g: 9.81, b_over_m: 1.7 }, { y0: 2, v0: 1.5 }, 2)).toBeLessThan(1e-9);
    expect(maxAbsError(FF2, { g: 9.81, k_over_m: 0.4 }, { y0: 5, v0: 0 }, 3)).toBeLessThan(1e-9);
  });

  it("FF1 and FF2 approach their terminal velocities", () => {
    const r1 = integrate(FF1, { g: 9.81, b_over_m: 2 }, { y0: 0, v0: 0 }, { tEnd: 8 });
    expect(Math.abs(r1.states.at(-1)![1])).toBeCloseTo(terminalVelocity("FF1", { g: 9.81, b_over_m: 2 })!, 5);
    const r2 = integrate(FF2, { g: 9.81, k_over_m: 0.5 }, { y0: 0, v0: 0 }, { tEnd: 8 });
    expect(Math.abs(r2.states.at(-1)![1])).toBeCloseTo(Math.sqrt(9.81 / 0.5), 6);
  });

  it("the small-angle pendulum P0 matches theta0 cos(wt) + (omega0/w) sin(wt)", () => {
    expect(maxAbsError(P0, { g: 9.81, L_eff: 0.62 }, { theta0: 0.2, omega0: 0.3 }, 10)).toBeLessThan(1e-7);
  });

  it("every ramp model matches its constant-acceleration closed form", () => {
    const base = { g: 9.81, alpha: 0.4 };
    const init = { s0: 0, v0: 0 };
    expect(maxAbsError(R0, base, init, 1.5)).toBeLessThan(1e-10);
    expect(maxAbsError(R1, { ...base, mu_k: 0.2 }, init, 1.5)).toBeLessThan(1e-10);
    expect(maxAbsError(R2, { ...base, beta: 0.4 }, init, 1.5)).toBeLessThan(1e-10);
    expect(maxAbsError(R3, { ...base, beta: 0.4, c_rr: 0.01 }, init, 1.5)).toBeLessThan(1e-10);
    // A solid ball gets 5/7 of the frictionless acceleration (Part 3).
    expect(rampAcceleration("R2", { ...base, beta: 0.4 }) / rampAcceleration("R0", base)).toBeCloseTo(5 / 7, 12);
  });

  it("all four spin-down models match their closed forms, including the stop", () => {
    const init = { phi0: 0.5, omega0: 40 };
    expect(maxAbsError(S0, { b: 0.06 }, init, 30)).toBeLessThan(1e-8);
    expect(maxAbsError(S1, { c: 1.1 }, init, 45)).toBeLessThan(1e-8);
    expect(maxAbsError(S2, { c: 0.9, b: 0.035 }, init, 45)).toBeLessThan(1e-8);
    expect(maxAbsError(S3, { c: 0.9, k: 0.004 }, init, 45)).toBeLessThan(1e-8);
  });

  it("converges at fourth order when the step is halved", () => {
    const p = { g: 9.81, L_eff: 0.62 };
    const init = { theta0: 0.5, omega0: 0 };
    const coarse = maxAbsError(P0, p, init, 5, 1 / 30);
    const fine = maxAbsError(P0, p, init, 5, 1 / 60);
    // A fourth-order method loses a factor of about 16 in error per halving.
    expect(coarse / fine).toBeGreaterThan(12);
    expect(coarse / fine).toBeLessThan(20);
  });
});

describe("pendulum", () => {
  const g = 9.81;
  const L = 0.62;

  it("conserves energy when undamped (P1)", () => {
    const theta0 = 60 * DEG;
    const res = integrate(P1, { g, L_eff: L }, { theta0, omega0: 0 }, { tEnd: 60 });
    const energy = (s: number[]) => 0.5 * L * L * s[1] * s[1] + g * L * (1 - Math.cos(s[0]));
    const e0 = energy(res.states[0]);
    const drift = Math.max(...res.states.map((s) => Math.abs(energy(s) - e0))) / e0;
    expect(drift).toBeLessThan(1e-7);
  });

  it("loses energy monotonically with viscous damping (P2)", () => {
    const res = integrate(P2, { g, L_eff: L, gamma: 0.1 }, { theta0: 0.6, omega0: 0 }, { tEnd: 30 });
    const energy = (s: number[]) => 0.5 * L * L * s[1] * s[1] + g * L * (1 - Math.cos(s[0]));
    let prev = Infinity;
    // Sample at the turning points so the comparison is between peaks.
    for (let i = 1; i < res.states.length - 1; i++) {
      if (Math.abs(res.states[i][1]) < Math.abs(res.states[i - 1][1]) && Math.abs(res.states[i][1]) <= Math.abs(res.states[i + 1][1])) {
        expect(energy(res.states[i])).toBeLessThan(prev + 1e-12);
        prev = energy(res.states[i]);
      }
    }
    expect(energy(res.states.at(-1)!)).toBeLessThan(0.5 * energy(res.states[0]));
  });

  it("the nonlinear period matches the elliptic-integral formula", () => {
    for (const amp of [10, 38, 70]) {
      const res = integrate(P1, { g, L_eff: L }, { theta0: amp * DEG, omega0: 0 }, { tEnd: 30 });
      const T = measuredPeriod(res.t, res.obs[0]);
      // Released from rest at +theta0, the first upward zero crossing happens 3/4 of a period in,
      // and consecutive crossings are one period apart.
      expect(T).not.toBeNull();
      expect(T! / exactPeriod(g, L, amp * DEG)).toBeCloseTo(1, 6);
    }
    // Large swings are slower than the small-angle formula says.
    expect(exactPeriod(g, L, 38 * DEG)).toBeGreaterThan(smallAnglePeriod(g, L) * 1.02);
  });
});

describe("spin-down clamp", () => {
  it("stops exactly at omega = 0 and holds the angle (S1)", () => {
    const res = integrate(S1, { c: 2 }, { phi0: 0, omega0: 10 }, { tEnd: 12 });
    expect(res.stoppedAt).toBeCloseTo(5, 9);
    const last = res.states.at(-1)!;
    expect(last[1]).toBe(0);
    expect(last[0]).toBeCloseTo((10 * 10) / (2 * 2), 9); // omega0^2 / 2c
    // Never negative, never reverses.
    expect(Math.min(...res.states.map((s) => s[1]))).toBeGreaterThanOrEqual(0);
    for (let i = 1; i < res.obs[0].length; i++) expect(res.obs[0][i]).toBeGreaterThanOrEqual(res.obs[0][i - 1]);
    // Everything after the stop is flat.
    const iStop = Math.ceil(5 * 240) + 1;
    expect(res.obs[0][iStop + 100]).toBe(res.obs[0][iStop]);
  });

  it("finds the stop time of S2 and S3 in closed form", () => {
    const p2 = { c: 0.9, b: 0.035 };
    const r2 = integrate(S2, p2, { phi0: 0, omega0: 55 }, { tEnd: 60 });
    expect(r2.stoppedAt).toBeCloseTo(spinStopTime("S2", p2, 55), 8);
    const p3 = { c: 0.9, k: 0.004 };
    const r3 = integrate(S3, p3, { phi0: 0, omega0: 55 }, { tEnd: 60 });
    expect(r3.stoppedAt).toBeCloseTo(spinStopTime("S3", p3, 55), 8);
    expect(spinOmega("S2", 10, p2, 55)).toBeCloseTo(r2.states[Math.round(10 * 240)][1], 6);
  });

  it("pure viscous friction (S0) never reaches zero", () => {
    const res = integrate(S0, { b: 0.2 }, { phi0: 0, omega0: 10 }, { tEnd: 30 });
    expect(res.stoppedAt).toBeNull();
    expect(res.states.at(-1)![1]).toBeGreaterThan(0);
    expect(spinStopTime("S0", { b: 0.2 }, 10)).toBe(Infinity);
  });

  it("a rotor that starts at rest stays at rest", () => {
    const res = integrate(S2, { c: 0.9, b: 0.03 }, { phi0: 1, omega0: 0 }, { tEnd: 2 });
    expect(res.obs[0].every((v) => v === 1)).toBe(true);
  });
});

describe("bounce events", () => {
  it("finds the first impact exactly and flips the velocity by e", () => {
    const g = 9.81;
    const h = 1.2;
    const e = 0.8;
    const res = integrate(B0, { g, e }, { y0: h, v0: 0 }, { tEnd: 3 });
    const impacts = res.events.filter((ev) => ev.kind === "impact");
    expect(impacts[0].t).toBeCloseTo(Math.sqrt((2 * h) / g), 9);
    expect(impacts[0].state[1]).toBeCloseTo(e * g * impacts[0].t, 9);
    // Successive flight times shrink by e: dt_{n+1} / dt_n = e (the audio ground truth in Part 25).
    const gaps = impacts.slice(1).map((ev, i) => ev.t - impacts[i].t);
    expect(gaps[1] / gaps[0]).toBeCloseTo(e, 8);
    // The ball never goes through the floor.
    expect(Math.min(...res.obs[0])).toBeGreaterThanOrEqual(-1e-12);
  });

  it("with e = 1 the ball returns to its release height", () => {
    const res = integrate(B0, { g: 9.81, e: 1 }, { y0: 1, v0: 0 }, { tEnd: 2 * Math.sqrt(2 / 9.81) + 0.01 });
    const tTop = 2 * Math.sqrt(2 / 9.81);
    const i = Math.round(tTop * 240);
    expect(res.obs[0][i]).toBeCloseTo(1, 1);
    expect(Math.max(...res.obs[0])).toBeLessThanOrEqual(1 + 1e-9);
  });

  it("comes to rest instead of bouncing forever, and handles speed-dependent e", () => {
    const res = integrate(B0, { g: 9.81, e: 0.6 }, { y0: 1, v0: 0 }, { tEnd: 10 });
    expect(res.stoppedAt).not.toBeNull();
    expect(res.stoppedAt!).toBeLessThan(10);
    expect(res.states.at(-1)).toEqual([0, 0]);
    const b1 = integrate(B1, { g: 9.81, e0: 0.9, e1: 0.02 }, { y0: 1, v0: 0 }, { tEnd: 8 });
    const first = b1.events.find((ev) => ev.kind === "impact")!;
    const vIn = Math.sqrt(2 * 9.81 * 1);
    expect(first.state[1]).toBeCloseTo((0.9 - 0.02 * vIn) * vIn, 6);
  });
});

describe("free fall floor", () => {
  it("stops at the floor with the impact speed sqrt(2 g h)", () => {
    const res = integrate(FF0, { g: 9.81 }, { y0: 1.2, v0: 0 }, { tEnd: 2, extraEvents: [floorEvent(0)] });
    expect(res.stoppedAt).toBeCloseTo(Math.sqrt((2 * 1.2) / 9.81), 9);
  });
});

/* ------------------------------------------------------------------------------------ */
/* The what-if layer                                                                     */
/* ------------------------------------------------------------------------------------ */

const pendulumCtx: SimContext = {
  system: "pendulum",
  model_ref: "P2",
  params: { g: 9.79, gamma: 0.052 },
  fixed: { L_eff: 0.62 },
  initial: { theta0: 38 * DEG, omega0: 0 },
  measured_range: { amplitude_deg: [30, 40] },
  object: { name: "Steel nut", radius_m: 0.012, mass_kg: 0.03 },
  anchor_m: [0, 0.7],
};

describe("what-if scenarios", () => {
  it("the default sliders reproduce the fitted model exactly", () => {
    const sc = buildScenario(pendulumCtx, defaultControls(pendulumCtx));
    const direct = integrate(P2, { g: 9.79, L_eff: 0.62, gamma: 0.052 }, pendulumCtx.initial, { tEnd: sc.tEnd });
    const res = runScenario(sc);
    expect(res.obs[0]).toEqual(direct.obs[0]);
  });

  it("the Mars pendulum has a period exactly sqrt(9.81 / 3.71) times Earth's", () => {
    const undamped: SimContext = { ...pendulumCtx, model_ref: "P1", params: { g: 9.81 } };
    const earth = buildScenario(undamped, { ...defaultControls(undamped), g: 9.81 });
    const mars = buildScenario(undamped, { ...defaultControls(undamped), g: 3.71 });
    const tE = measuredPeriod(runScenario(earth).t, runScenario(earth).obs[0])!;
    const tM = measuredPeriod(runScenario(mars).t, runScenario(mars).obs[0])!;
    expect(tM / tE).toBeCloseTo(Math.sqrt(9.81 / 3.71), 5);
    expect(Math.sqrt(9.81 / 3.71)).toBeCloseTo(1.6261, 3);
  });

  it("bob mass changes the damping but not the period", () => {
    // A small swing, so the amplitude dependence of the period is negligible.
    const small: SimContext = { ...pendulumCtx, initial: { theta0: 3 * DEG, omega0: 0 } };
    const base = buildScenario(small, defaultControls(small));
    const heavy = buildScenario(small, { ...defaultControls(small), mass: 2 });
    expect(heavy.params.gamma).toBeCloseTo(base.params.gamma / 2, 12);
    const rb = computeReadouts(base, runScenario(base));
    const rh = computeReadouts(heavy, runScenario(heavy));
    expect(rh.find((r) => r.id === "period")!.value!).toBeCloseTo(rb.find((r) => r.id === "period")!.value!, 4);
    // The heavy bob keeps more of its swing after the same time.
    expect(rh.find((r) => r.id === "final_amp")!.value!).toBeGreaterThan(rb.find((r) => r.id === "final_amp")!.value!);
  });

  it("explains a gravity change with the 1/sqrt(g) rule", () => {
    const baseV = defaultControls(pendulumCtx);
    const whatV = { ...baseV, g: 3.71 };
    const bs = buildScenario(pendulumCtx, baseV);
    const ws = buildScenario(pendulumCtx, whatV);
    const text = instantExplanations(
      pendulumCtx,
      baseV,
      whatV,
      bs,
      ws,
      computeReadouts(bs, runScenario(bs)),
      computeReadouts(ws, runScenario(ws)),
    ).join(" ");
    expect(text).toContain("T ∝ 1/√g");
    expect(text).toContain("1.62× longer");
  });

  it("warns when a slider leaves the measured range", () => {
    const inside = extrapolationWarnings(pendulumCtx, { amplitude_deg: 35 });
    const outside = extrapolationWarnings(pendulumCtx, { amplitude_deg: 70 });
    expect(inside).toEqual([]);
    expect(outside).toHaveLength(1);
    expect(outside[0]).toContain("30–40");
  });

  it("offers only the controls the fitted model responds to", () => {
    const ids = controlsFor(pendulumCtx).map((c) => c.id);
    expect(ids).toEqual(["g", "length_cm", "amplitude_deg", "air", "mass"]);
    const undamped = controlsFor({ ...pendulumCtx, model_ref: "P1" }).map((c) => c.id);
    expect(undamped).toEqual(["g", "length_cm", "amplitude_deg", "mass"]);
  });

  it("a ramp: shapes change the acceleration by 1/(1+beta) and friction can stop the block", () => {
    const ctx: SimContext = {
      system: "ramp",
      model_ref: "R1",
      params: { mu_k: 0.25 },
      fixed: { g: 9.81, alpha: 20 * DEG, s_max: 1.2 },
      initial: { s0: 0, v0: 0 },
      measured_range: { angle_deg: [15, 25] },
      object: { name: "Wooden block", radius_m: 0.03, mass_kg: 0.2 },
      anchor_m: [0, 0.5],
    };
    const d = defaultControls(ctx);
    const block = computeReadouts(buildScenario(ctx, d), runScenario(buildScenario(ctx, d)));
    const ball = computeReadouts(
      buildScenario(ctx, { ...d, shape: "solid_ball" }),
      runScenario(buildScenario(ctx, { ...d, shape: "solid_ball" })),
    );
    const aBlock = block.find((r) => r.id === "accel")!.value!;
    expect(aBlock).toBeCloseTo(9.81 * (Math.sin(20 * DEG) - 0.25 * Math.cos(20 * DEG)), 9);
    expect(ball.find((r) => r.id === "accel")!.value!).toBeCloseTo((9.81 * Math.sin(20 * DEG)) / 1.4, 9);
    // At 10 degrees with mu = 0.25 > tan(10 deg) friction wins and the block never moves.
    const flat = buildScenario(ctx, { ...d, angle_deg: 10 });
    const rf = runScenario(flat);
    expect(rf.states.at(-1)![0]).toBeCloseTo(0, 12);
  });

  it("positions the body from the state: bob below the pivot at rest", () => {
    const sc = buildScenario(pendulumCtx, defaultControls(pendulumCtx));
    const rest = poseAt("pendulum", sc, [0, 0], pendulumCtx.anchor_m);
    expect(rest.x).toBeCloseTo(0, 12);
    expect(rest.y).toBeCloseTo(0.7 - 0.62, 12);
  });
});
