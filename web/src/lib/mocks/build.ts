import { getModel, integrate } from "@/lib/sim";
import type { Params } from "@/lib/sim";
import type {
  Conclusion,
  Experiment,
  FitResult,
  GhostRender,
  GhostTrack,
  ModelRecord,
  NotebookEvent,
  Quantity,
  QualityFlag,
  ReviewThread,
  SceneAnalysis,
  Trajectory,
  Trial,
  WhatIfContext,
} from "@/lib/types";
import { fmt, fmtPm, runAnalysis, type AnalysisOutput, type FittedPlan } from "./analysis";
import { fitModel } from "./fitting";
import { buildNotebook } from "./notebook";
import { cantTellInput, freeFallInput, pendulumInput, SCENARIOS, spinInput } from "./plans";
import { gaussian, mulberry32 } from "./rng";

/**
 * The mock experiments: a spin-down whose first guess is refuted, a pendulum with a ghost
 * that drifts and a revised v2, a free-fall drop, and an out-of-plane pendulum that
 * Ghostline says it cannot call. Every number comes from the mock harness in `analysis.ts`.
 */

export interface MockBundle {
  experiment: Experiment;
  events: NotebookEvent[];
  trajectories: Record<string, Trajectory>;
  tracks: Record<string, GhostTrack>;
}

const DEG = Math.PI / 180;
const W = 960;
const H = 540;
const BASE_TIME = Date.parse("2026-10-27T16:40:00Z");
export const iso = (minutes: number) => new Date(BASE_TIME + minutes * 60_000).toISOString();

type Map3 = number[];
function affine(sx: number, sy: number, ox: number, oy: number): Map3 {
  // pixel = (ox + sx * x, oy - sy * y)
  return [sx, 0, ox, 0, -sy, oy, 0, 0, 1];
}
const toPx = (m: Map3, x: number, y: number): [number, number] => [m[0] * x + m[2], m[4] * y + m[5]];

function derivative(t: number[], v: number[], half = 4): number[] {
  return v.map((_, i) => {
    const a = Math.max(0, i - half);
    const b = Math.min(v.length - 1, i + half);
    return (v[b] - v[a]) / (t[b] - t[a]);
  });
}

function thin<T>(arr: T[], maxN: number): T[] {
  if (arr.length <= maxN) return arr;
  const step = arr.length / maxN;
  return Array.from({ length: maxN }, (_, i) => arr[Math.floor(i * step)]);
}

function modelRecords(
  out: AnalysisOutput,
  winnerName: string,
  extras: Record<string, string>,
  unit: string,
): ModelRecord[] {
  const best = out.fitted.find((f) => f.plan.ref === out.winnerRef)!.result.heldout_rmse;
  return out.fitted.map((f) => {
    const status = out.statuses[f.plan.ref];
    const model = getModel(f.plan.ref)!;
    let reason: string | null = null;
    if (status === "rejected") {
      const ratio = f.result.heldout_rmse / best;
      const bounded = f.result.params.find((p) => p.at_bound);
      reason =
        `Held-out RMSE ${fmt(f.result.heldout_rmse)} ${unit} against ${fmt(best)} ${unit} for ${winnerName}: ${ratio >= 10 ? `${ratio.toFixed(0)}` : ratio.toFixed(1)}× worse, and the intervals do not overlap (${fmt(f.result.heldout_ci[0])} to ${fmt(f.result.heldout_ci[1])} ${unit}).` +
        (bounded ? ` ${bounded.label} was pinned at its bound of ${fmt(bounded.value)} ${bounded.unit}, which means the model is bending a parameter to hide a wrong shape.` : "") +
        (extras[f.plan.ref] ? ` ${extras[f.plan.ref]}` : "");
    }
    return {
      id: f.plan.ref,
      name: f.plan.name ?? model.name,
      round: 1,
      source: "library",
      library_ref: f.plan.ref,
      mechanism: f.plan.mechanism,
      equations_latex: f.plan.equations_latex,
      prior_rank: f.plan.prior_rank,
      refuting_signature: f.plan.refuting_signature,
      status,
      rejection_reason: reason,
      fit: f.result as FitResult,
    };
  });
}

/** Refit the winner on all the data and combine the uncertainty budget (Part 12). */
function finalQuantity(
  out: AnalysisOutput,
  winner: FittedPlan,
  paramName: string,
  meta: { name: string; symbol_latex: string; unit: string; scaleExp: number; timeExp: number; offsetExp?: number; note?: string },
  calib: { scaleRel: number; offsetRel: number; timingRel: number },
): Quantity {
  const { t, y } = out.input;
  const spec = { model: getModel(winner.plan.ref)!, fixed: winner.plan.fixed, initBase: winner.plan.initBase, free: winner.plan.free };
  const fin = fitModel(spec, t, y, t.length);
  const value = fin.values[paramName];
  const stat = 1.96 * 3 * (fin.stderr[paramName] ?? 0);
  const sScale = Math.abs(meta.scaleExp) * calib.scaleRel * Math.abs(value) * 1.96;
  const sOffset = Math.abs(meta.offsetExp ?? meta.scaleExp) * calib.offsetRel * Math.abs(value) * 1.96;
  const sTiming = Math.abs(meta.timeExp) * calib.timingRel * Math.abs(value) * 1.96;
  const tiedVals = out.fitted
    .filter((f) => out.tiedRefs.includes(f.plan.ref) && f.result.params.some((p) => p.name === paramName))
    .map((f) => f.result.params.find((p) => p.name === paramName)!.value);
  const sModel = tiedVals.length > 1 ? (Math.max(...tiedVals) - Math.min(...tiedVals)) / 2 : 0;
  const total = Math.sqrt(stat ** 2 + sScale ** 2 + sOffset ** 2 + sTiming ** 2 + sModel ** 2);
  return {
    name: meta.name,
    symbol_latex: meta.symbol_latex,
    value,
    ci_low: value - total,
    ci_high: value + total,
    unit: meta.unit,
    note: meta.note,
    budget: [
      { label: "Statistical (fit)", sigma: stat },
      { label: "Calibration scale", sigma: sScale },
      { label: "Plane offset", sigma: sOffset },
      { label: "Timing", sigma: sTiming },
      { label: "Model choice", sigma: sModel },
    ],
  };
}

function media(file: string): string {
  return `/mock/${file}`;
}

function baseTrial(
  id: string,
  role: "fit" | "prove",
  expId: string,
  o: {
    n: number;
    duration: number;
    fps: number;
    declared: "normal" | "slowmo";
    declaredFps: number;
    video: string;
    poster: string;
    mmPerPx: number;
    sigma: number;
    offsetCm: number;
    worldToPx: Map3 | null;
    flags: QualityFlag[];
    cameraMotion?: number;
    trajectory: boolean;
    paper?: "a4" | "letter";
  },
): Trial {
  return {
    id,
    role,
    status: "tracked",
    width: W,
    height: H,
    n_frames: o.n,
    duration_s: o.duration,
    timing: {
      classification: "cfr",
      detected_fps: o.fps,
      declared_mode: o.declared,
      declared_fps: o.declaredFps,
      time_factor: 1,
      jitter_pct: 0.4,
      warnings: [],
    },
    trim: {
      start_s: 0,
      end_s: o.duration,
      suggested_start_s: 0,
      suggested_end_s: o.duration,
    },
    calibration: {
      paper: o.paper ?? "a4",
      mm_per_px: o.mmPerPx,
      sigma_scale_rel: o.sigma,
      plane_offset_cm: o.offsetCm,
      gravity_source: "data",
      ramp_angle_deg: null,
      camera_motion_px: o.cameraMotion ?? 0.3,
      world_to_px: o.worldToPx,
    },
    quality_flags: o.flags,
    media: {
      playback_url: media(o.video),
      poster_url: media(o.poster),
      calib_overlay_url: media("calib-overlay.svg"),
      trajectory_url: o.trajectory ? `mock:trajectory/${expId}/${id}` : null,
      sam_events_url: null,
    },
    error: null,
  };
}

/* ------------------------------------------------------------------------------------ */
/* Shared pieces                                                                         */
/* ------------------------------------------------------------------------------------ */

const NO_FLAGS: QualityFlag[] = [];

function sceneAnalysis(
  system: SceneAnalysis["system_type"],
  description: string,
  objects: SceneAnalysis["tracked_objects"],
  props: SceneAnalysis["object_properties"],
  out: AnalysisOutput,
  issues: SceneAnalysis["quality_issues"] = [],
): SceneAnalysis {
  return {
    system_type: system,
    description,
    tracked_objects: objects,
    object_properties: props,
    initial_hypotheses: out.fitted
      .slice()
      .sort((a, b) => a.plan.prior_rank - b.plan.prior_rank)
      .slice(0, 3)
      .map((f) => ({
        name: f.plan.name ?? getModel(f.plan.ref)!.name,
        mechanism: f.plan.mechanism,
        observable_signature: f.plan.refuting_signature,
      })),
    quality_issues: issues,
    proceed: true,
    reason_if_not: null,
  };
}

function assembleConclusion(
  out: AnalysisOutput,
  confidence: Conclusion["confidence"],
  reasons: string[],
  quantities: Quantity[],
  narrative: string,
  caveats: string[],
  cant: Conclusion["cant_tell"] = null,
): Conclusion {
  const winner = out.fitted.find((f) => f.plan.ref === out.winnerRef)!;
  const records = out.fitted.filter((f) => out.statuses[f.plan.ref] === "rejected");
  return {
    winner_id: winner.plan.ref,
    quantities,
    rejected: records.map((f) => ({
      id: f.plan.ref,
      reason: `Held-out RMSE ${fmt(f.result.heldout_rmse)} ${out.input.unit}, ${(f.result.heldout_rmse / winner.result.heldout_rmse).toFixed(1)}× the winner's.`,
      evidence: `heldout_rmse=${fmt(f.result.heldout_rmse)}; ci=[${fmt(f.result.heldout_ci[0])}, ${fmt(f.result.heldout_ci[1])}]`,
    })),
    confidence,
    confidence_reasons: reasons,
    caveats,
    narrative_md: narrative,
    cant_tell: cant,
  };
}

function summaryOf(
  id: string,
  title: string,
  system: Experiment["system_type"],
  conclusion: Conclusion,
  winnerName: string,
  review: ReviewThread | null,
  blurb: string,
  thumb: string,
  headline: string,
): Pick<
  Experiment,
  | "id" | "title" | "system_type" | "status" | "created_at" | "is_example" | "confidence"
  | "review_status" | "headline" | "winner_name" | "thumbnail_url" | "blurb"
> {
  return {
    id,
    title,
    system_type: system,
    status: "complete",
    created_at: iso(0),
    is_example: true,
    confidence: conclusion.confidence,
    review_status: review?.status ?? "not_reviewed",
    headline,
    winner_name: winnerName,
    thumbnail_url: thumb,
    blurb,
  };
}

function plusMinus(q: Quantity, _digits = 3): string {
  void _digits;
  return `${fmtPm(q.value, (q.ci_high - q.ci_low) / 2)} ${q.unit}`;
}

/* ------------------------------------------------------------------------------------ */
/* 1. Spin-down: the hero story                                                          */
/* ------------------------------------------------------------------------------------ */

function buildSpin(): MockBundle {
  const sc = SCENARIOS.spin;
  const { input } = spinInput();
  const out = runAnalysis(input);
  const winner = out.fitted.find((f) => f.plan.ref === out.winnerRef)!;
  const s0 = out.fitted.find((f) => f.plan.ref === "S0")!;
  const s1 = out.fitted.find((f) => f.plan.ref === "S1")!;
  const wName = "Coulomb and viscous friction";
  const models = modelRecords(
    out,
    wName,
    {
      S0: "An exponential never reaches zero, but the real spinner stopped at about 32.7 s: the model kept spinning through the whole last third of the clip.",
      S1: "A constant deceleration cannot bend: the real ω(t) is steeper at the start and shallower at the end.",
      S3: "Quadratic drag removes far too much spin early and then almost none.",
    },
    "rad",
  );

  const calib = { scaleRel: 0.004, offsetRel: 0.0, timingRel: 0.0004 };
  const quantities = [
    finalQuantity(out, winner, "c", { name: "Rubbing friction (per unit inertia)", symbol_latex: "c", unit: "rad/s²", scaleExp: 0, timeExp: 2, note: "Depends only on timing: angles need no scale." }, calib),
    finalQuantity(out, winner, "b", { name: "Viscous friction rate", symbol_latex: "b", unit: "1/s", scaleExp: 0, timeExp: 1 }, calib),
  ];
  const c = quantities[0];
  const b = quantities[1];

  const conclusion = assembleConclusion(
    out,
    "high",
    [
      "S2 beat the runner-up on all three splits (50%, 65% and 80%).",
      "No systematic pattern is left in its residuals.",
      "No quality flags: tripod, level, sheet fully visible, no aliasing (largest step 41° per frame).",
      "The reviewer challenged once, Wonder answered with numbers, and the reviewer then accepted.",
    ],
    [c, b],
    `The spinner slows down by a **steady rubbing term** (${plusMinus(c, 3)}) plus a **small speed-dependent term** (${plusMinus(b, 2)}). My first guess, pure viscous friction, was **refuted**: it predicted a spinner that never stops, and this one stopped at ${fmt(out.input.t[out.input.t.length - 1] - 1.3, 3)} s. The prediction for the last third of the clip missed by ${fmt(s0.result.heldout_rmse, 2)} rad (RMSE), against ${fmt(winner.result.heldout_rmse, 2)} rad for the winner.`,
    [
      "The rubbing and viscous terms are per unit of inertia; a spinner's mass distribution cannot be seen on video, so absolute torques are not claimed.",
      "A viscous term and a quadratic air-drag term cannot be separated by this clip alone; S3 is rejected on shape, not on mechanism.",
    ],
  );

  const review: ReviewThread = {
    status: "accepted",
    rounds: [
      {
        round: 1,
        verdict: "challenge",
        summary:
          "The headline is plausible, but the second parameter is carrying more weight than the evidence shows. I want to see that b is needed.",
        issues: [
          {
            id: "R1-1",
            severity: "major",
            claim: "S2 beats S1 because of the viscous term b.",
            evidence: `S1 held-out RMSE ${fmt(s1.result.heldout_rmse, 2)} rad, S2 ${fmt(winner.result.heldout_rmse, 2)} rad. b = ${fmt(b.value, 2)} 1/s is small; the intervals depend on one 65% split.`,
            requested_action: "refit",
            detail: "Refit both models with the split at 50% and at 80%, and report whether b stays away from zero.",
          },
          {
            id: "R1-2",
            severity: "minor",
            claim: "The narrative says the spinner 'stopped'.",
            evidence: "The last 1.3 s of the clip are flat, but the angle estimate is only exact to ±0.015 rad.",
            requested_action: "add_caveat",
            detail: "Say that the stop time is read from where ω reaches zero in the fit, not observed directly.",
          },
        ],
        response_md: `**Refit at 50% and 80%.** S2 wins both: held-out RMSE ${fmt(winner.splitRmse["50"], 2)} rad at 50% and ${fmt(winner.splitRmse["80"], 2)} rad at 80%, against ${fmt(s1.splitRmse["50"], 2)} and ${fmt(s1.splitRmse["80"], 2)} rad for S1. In all three fits b stays at ${fmt(b.value, 2)} 1/s with an interval that excludes zero.\n\n**Caveat added:** the stop time is read from the fit's ω(t) reaching zero.`,
      },
      {
        round: 2,
        verdict: "accept",
        summary:
          "The refits answer the objection: the viscous term is needed on every split and the advantage is far larger than the interval. Accepted.",
        issues: [],
        response_md: null,
      },
    ],
  };

  const traj = spinTrajectory(out, input.t, input.y);
  const trial = baseTrial("spin-t1", "fit", sc.id, {
    n: input.t.length,
    duration: sc.window_s,
    fps: 60,
    declared: "normal",
    declaredFps: 60,
    video: "spin-t1.mp4",
    poster: "spin-poster.jpg",
    mmPerPx: 0.52,
    sigma: 0.004,
    offsetCm: 0,
    worldToPx: affine(3000, 3000, 480, 270),
    flags: NO_FLAGS,
    trajectory: true,
  });
  const scene = sceneAnalysis(
    "spin_down",
    "A fidget spinner lying flat on a table, filmed from above, with a dot sticker on one arm. It is flicked once and spins down to rest. An A4 sheet lies flat beside it.",
    [
      { key: "marker", sam_prompt: "blue dot sticker", role: "marker", why: "Its angle around the hub gives the spin angle." },
      { key: "hub", sam_prompt: "black spinner hub", role: "pivot", why: "The center of rotation." },
    ],
    { shape: "three-arm spinner", material_guess: "plastic", approx_size_cm: 7 },
    out,
  );

  const whatif: WhatIfContext = {
    system: "spin_down",
    model_ref: "S2",
    params: { c: c.value, b: b.value },
    fixed: { g: 9.8, marker_radius_m: sc.marker_radius_m },
    initial: { phi0: 0, omega0: sc.truth.omega0 },
    measured_range: { spin_rev_s: [8, 9.5] },
    object: { name: "Fidget spinner", radius_m: sc.marker_radius_m, mass_kg: null },
    still_url: media("spin-still.jpg"),
    sim_supported: true,
    anchor_m: [0, 0],
  };

  const experiment: Experiment = {
    ...summaryOf(sc.id, "Fidget spinner spin-down", "spin_down", conclusion, `S2: ${wName}`, review,
      "Wonder guessed exponential decay. The held-out frames said otherwise.", media("spin-poster.jpg"),
      `Rubbing friction c = ${plusMinus(c, 3)}`),
    description: "Flicked a fidget spinner on a table and filmed it from above.",
    declared_capture: { mode: "normal", fps: 60 },
    setup: { plane_offset_cm: 0, paper: "a4", pendulum_length_cm: null },
    scene,
    trials: [trial],
    models,
    conclusion,
    review,
    ghosts: [],
    whatif,
    export_formats: ["json"],
    quality_flags: NO_FLAGS,
    job: null,
    error: null,
    last_seq: 0,
  };
  const events = buildNotebook(experiment, out, { favorite: "S0", rerunnable: true });
  experiment.last_seq = events.length;
  return { experiment, events, trajectories: { [trial.id]: traj }, tracks: {} };
}

function spinTrajectory(out: AnalysisOutput, t: number[], phi: number[]): Trajectory {
  const sc = SCENARIOS.spin;
  const m = affine(3000, 3000, 480, 270);
  return {
    t,
    x: null,
    y: null,
    theta: null,
    phi,
    omega: derivative(t, phi, 8),
    px: phi.map((p) => toPx(m, sc.marker_radius_m * Math.cos(p), sc.marker_radius_m * Math.sin(p))),
    split_t: out.splitT,
    noise_sigma: sc.sigma,
    main: "phi",
    main_unit: "rad",
    events: [
      { kind: "release", t: 0 },
      { kind: "stop", t: 32.7 },
    ],
  };
}

/* ------------------------------------------------------------------------------------ */
/* 2. Pendulum with a ghost                                                              */
/* ------------------------------------------------------------------------------------ */

function simulateTheta(modelRef: string, params: Params, theta0: number, omega0: number, t: number[]): number[] {
  const model = getModel(modelRef)!;
  const res = integrate(model, params, { theta0, omega0 }, { tEnd: t[t.length - 1] + 0.01 });
  return t.map((x) => res.obs[0][Math.round(x * 240)]);
}

/** Build one ghost's track: fit initial conditions on the launch window, then predict. */
function pendulumGhost(
  version: 1 | 2,
  actual: { t: number[]; y: number[]; truthTheta: number[] },
  frozen: { ref: string; params: Params },
  L: number,
  pivotPx: [number, number],
  scalePx: number,
): { track: GhostTrack; ghost: Omit<GhostRender, "id" | "version" | "trial_id" | "video_url" | "slow_video_url" | "track_url" | "poster_url" | "revision" | "model_id" | "model_name"> } {
  const { t, y } = actual;
  const sigma = 0.001;
  const launch = Math.max(Math.round(0.1 * t.length), Math.round(0.25 * 60));
  const model = getModel(frozen.ref)!;
  const fit = fitModel(
    {
      model,
      fixed: { ...frozen.params, L_eff: L },
      initBase: {},
      free: [
        { name: "theta0", init: y[0], scale: 0.05, kind: "initial_condition", unit: "rad", label: "θ₀" },
        { name: "omega0", init: 0, scale: 0.1, kind: "initial_condition", unit: "rad/s", label: "ω₀" },
      ],
    },
    t,
    y,
    launch,
  );
  const pred = fit.predict(t);
  const errMm = pred.map((p, i) => Math.abs(p - y[i]) * L * 1000);
  const radiusMm = SCENARIOS.pendulum.bob_radius_m * 1000;
  const tau = Math.max(3 * sigma * L * 1000, radiusMm / 2);
  let onset: number | null = null;
  let run = 0;
  for (let i = launch; i < errMm.length; i++) {
    run = errMm[i] > tau ? run + 1 : 0;
    if (run >= 5) {
      onset = t[i - 4];
      break;
    }
  }
  const px = (theta: number): [number, number] => [
    pivotPx[0] + scalePx * L * Math.sin(theta),
    pivotPx[1] + scalePx * L * Math.cos(theta),
  ];
  const track: GhostTrack = {
    t: thin(t, 400),
    error_mm: thin(errMm, 400),
    predicted_px: thin(pred.map(px), 400),
    actual_px: thin(y.map(px), 400),
    tau_mm: tau,
    onset_s: onset,
    launch_window_s: t[launch],
  };
  void version;
  return {
    track,
    ghost: {
      divergence_onset_s: onset,
      max_error_mm: Math.max(...errMm.slice(launch)),
      threshold_mm: tau,
      predicted_ahead_s: t[t.length - 1] - t[launch],
    },
  };
}

function buildPendulum(): MockBundle {
  const sc = SCENARIOS.pendulum;
  const { input } = pendulumInput();
  const out = runAnalysis(input);
  const winner = out.fitted.find((f) => f.plan.ref === out.winnerRef)!;
  const p2 = out.fitted.find((f) => f.plan.ref === "P2")!;
  const p0 = out.fitted.find((f) => f.plan.ref === "P0")!;
  const L = sc.truth.L_eff;
  const wName = "Nonlinear pendulum";
  const models = modelRecords(
    out,
    wName,
    {
      P0: "At this amplitude the small-angle period is wrong by several percent, so the phase of the predicted swing walks away from the real one.",
    },
    "rad",
  );
  const calib = { scaleRel: 0.004, offsetRel: 0.0025, timingRel: 0.001 };
  const g = finalQuantity(
    out,
    winner,
    "g",
    { name: "Gravitational acceleration", symbol_latex: "g", unit: "m/s²", scaleExp: 1, timeExp: 2, note: "Scales with the length L_eff measured from the video." },
    calib,
  );
  const lq: Quantity = {
    name: "Effective length from the video",
    symbol_latex: "L_\\mathrm{eff}",
    value: L,
    ci_low: L * (1 - 0.0045),
    ci_high: L * (1 + 0.0045),
    unit: "m",
    budget: null,
  };
  const conclusion = assembleConclusion(
    out,
    "medium",
    [
      "The winner is the simplest of three models the data cannot tell apart (P1, P2 and P3 have overlapping held-out intervals), chosen by parsimony.",
      "P1 had the lowest held-out error on 0 of 3 splits, so the evidence for it is weak.",
      "The reviewer accepted with caveats.",
    ],
    [g, lq],
    `The swing is the **exact nonlinear pendulum**, g = ${plusMinus(g, 3)}. The small-angle formula is **refuted** at this 50° amplitude: its held-out error is ${(p0.result.heldout_rmse / winner.result.heldout_rmse).toFixed(1)}× larger, and it only fits by pushing g down to its bound of ${fmt(p0.result.params.find((p) => p.name === "g")!.value, 3)} m/s². Over only ${sc.window_s} s the damping is too small to see, so the data **cannot distinguish** the undamped model from the damped ones.`,
    [
      "In 5.5 s of swinging, any damping smaller than about 0.02 /s is invisible. Film a longer clip to measure it.",
      "g is quoted for the video-measured length L_eff; with a tape-measured L the value moves by the tape's uncertainty.",
    ],
  );
  const review: ReviewThread = {
    status: "accepted_with_caveats",
    rounds: [
      {
        round: 1,
        verdict: "accept_with_caveats",
        summary:
          "The nonlinear model is the right physics for a 50° swing and the small-angle alternative is properly rejected. The claim is honest that damping is unresolved.",
        issues: [
          {
            id: "R1-1",
            severity: "minor",
            claim: "The conclusion prefers P1 without damping.",
            evidence: `P2 and P3 have the lower held-out error (${fmt(p2.result.heldout_rmse, 2)} rad against ${fmt(winner.result.heldout_rmse, 2)} rad) but the intervals overlap.`,
            requested_action: "add_caveat",
            detail: "State that the preference is by parsimony and that a longer clip would resolve damping.",
          },
        ],
        response_md: "Caveat added to the conclusion: the preference for P1 is by parsimony, not by evidence.",
      },
    ],
  };

  // Trial 2 for the ghost.
  const t2 = Array.from({ length: Math.round(sc.trial2.window_s * 60) + 1 }, (_, i) => i / 60);
  const truth2 = simulateTheta("P2", { g: sc.truth.g, L_eff: L, gamma: sc.truth.gamma }, sc.trial2.theta0_deg * DEG, 0, t2);
  const rng = mulberry32(sc.trial2.seed);
  const noise2 = 0.001;
  const y2 = truth2.map((v) => v + noise2 * gaussian(rng));
  const pivotPx: [number, number] = [480, 70];
  const scalePx = 600;
  const finalP1 = fitModel(
    { model: getModel("P1")!, fixed: { L_eff: L }, initBase: {}, free: winner.plan.free },
    input.t, input.y, input.t.length,
  ).values;
  const finalP2 = fitModel(
    { model: getModel("P2")!, fixed: { L_eff: L }, initBase: {}, free: p2.plan.free },
    input.t, input.y, input.t.length,
  ).values;
  const v1 = pendulumGhost(1, { t: t2, y: y2, truthTheta: truth2 }, { ref: "P1", params: { g: finalP1.g } }, L, pivotPx, scalePx);
  const v2 = pendulumGhost(2, { t: t2, y: y2, truthTheta: truth2 }, { ref: "P2", params: { g: finalP2.g, gamma: finalP2.gamma } }, L, pivotPx, scalePx);

  const trial1 = baseTrial("pend-t1", "fit", sc.id, {
    n: input.t.length, duration: sc.window_s + 1, fps: 60, declared: "normal", declaredFps: 60,
    video: "pendulum-t1.mp4", poster: "pendulum-poster.jpg", mmPerPx: 1.67, sigma: 0.004, offsetCm: 2,
    worldToPx: affine(scalePx, scalePx, 480, 70 + scalePx * 0.7), flags: NO_FLAGS, trajectory: true,
  });
  const trial2 = baseTrial("pend-t2", "prove", sc.id, {
    n: t2.length, duration: sc.trial2.window_s, fps: 60, declared: "normal", declaredFps: 60,
    video: "pendulum-ghost-v1.mp4", poster: "pendulum-poster.jpg", mmPerPx: 1.67, sigma: 0.004, offsetCm: 2,
    worldToPx: affine(scalePx, scalePx, 480, 70 + scalePx * 0.7), flags: NO_FLAGS, trajectory: false,
  });

  const adopted = v2.ghost.max_error_mm < 0.7 * v1.ghost.max_error_mm;
  const ghosts: GhostRender[] = [
    {
      id: "gh-pend-1", version: 1, trial_id: trial2.id, model_id: "P1", model_name: "P1: Nonlinear pendulum",
      video_url: media("pendulum-ghost-v1.mp4"), slow_video_url: null, track_url: "mock:ghost/ex-pendulum/v1",
      poster_url: media("pendulum-poster.jpg"), revision: null, ...v1.ghost,
    },
    {
      id: "gh-pend-2", version: 2, trial_id: trial2.id, model_id: "P2", model_name: "P2: Nonlinear pendulum with viscous damping",
      video_url: media("pendulum-ghost-v2.mp4"), slow_video_url: null, track_url: "mock:ghost/ex-pendulum/v2",
      poster_url: media("pendulum-poster.jpg"),
      revision: {
        diagnosis: "The real swing shrinks while the ghost keeps its full amplitude: the ghost leads by more and more late in the run. That is the signature of missing damping.",
        adopted,
        note: adopted
          ? `P2 (γ = ${fmt(finalP2.gamma, 2)} 1/s, fitted on trial 1 only) cuts the maximum error from ${fmt(v1.ghost.max_error_mm, 3)} mm to ${fmt(v2.ghost.max_error_mm, 3)} mm. Adopted.`
          : "The revision did not predict trial 2 better beyond noise, so v1 stays.",
      },
      ...v2.ghost,
    },
  ];

  const whatif: WhatIfContext = {
    system: "pendulum",
    model_ref: "P1",
    params: { g: finalP1.g },
    fixed: { L_eff: L },
    initial: { theta0: finalP1.theta0, omega0: finalP1.omega0 },
    measured_range: { amplitude_deg: [48, 52], length_cm: [60, 64] },
    object: { name: "Steel nut", radius_m: sc.bob_radius_m, mass_kg: 0.031 },
    still_url: media("pendulum-still.jpg"),
    sim_supported: true,
    anchor_m: [0, 0.7],
  };

  const experiment: Experiment = {
    ...summaryOf(sc.id, "Pendulum at 50°", "pendulum", conclusion, `P1: ${wName}`, review,
      "A big swing breaks the textbook formula. Then the ghost predicts a second trial.", media("pendulum-poster.jpg"),
      `g = ${plusMinus(g, 3)}`),
    description: "A steel nut on 62 cm of string, released from about 50 degrees.",
    declared_capture: { mode: "normal", fps: 60 },
    setup: { plane_offset_cm: 2, paper: "a4", pendulum_length_cm: 62.1 },
    scene: sceneAnalysis(
      "pendulum",
      "A heavy nut hangs from a string and swings in a plane parallel to the wall. An A4 sheet is taped to the wall behind it.",
      [
        { key: "bob", sam_prompt: "steel nut", role: "moving_body", why: "The swinging bob." },
        { key: "pivot", sam_prompt: "black clamp", role: "pivot", why: "The fixed end of the string." },
      ],
      { shape: "hex nut on a string", material_guess: "steel", approx_size_cm: 2.4 },
      out,
    ),
    trials: [trial1, trial2],
    models,
    conclusion,
    review,
    ghosts,
    whatif,
    export_formats: [],
    quality_flags: NO_FLAGS,
    job: null,
    error: null,
    last_seq: 0,
  };
  const events = buildNotebook(experiment, out, { favorite: "P1", rerunnable: true });
  experiment.last_seq = events.length;
  return {
    experiment,
    events,
    trajectories: { [trial1.id]: pendulumTrajectory(out, input.t, input.y, L, scalePx) },
    tracks: { "ex-pendulum/v1": v1.track, "ex-pendulum/v2": v2.track },
  };
}

function pendulumTrajectory(out: AnalysisOutput, t: number[], theta: number[], L: number, s: number): Trajectory {
  const m = affine(s, s, 480, 70 + s * 0.7);
  const x = theta.map((th) => L * Math.sin(th));
  const y = theta.map((th) => 0.7 - L * Math.cos(th));
  return {
    t,
    x,
    y,
    theta,
    phi: null,
    omega: derivative(t, theta, 4),
    px: theta.map((th) => toPx(m, L * Math.sin(th), 0.7 - L * Math.cos(th))),
    split_t: out.splitT,
    noise_sigma: out.input.sigma,
    main: "theta",
    main_unit: "rad",
    events: [{ kind: "release", t: 0 }],
  };
}

/* ------------------------------------------------------------------------------------ */
/* 3. Free fall                                                                          */
/* ------------------------------------------------------------------------------------ */

function buildFreeFall(): MockBundle {
  const sc = SCENARIOS.free_fall;
  const { input } = freeFallInput();
  const out = runAnalysis(input);
  const winner = out.fitted.find((f) => f.plan.ref === out.winnerRef)!;
  const wName = "Constant gravity";
  const models = modelRecords(out, wName, {}, "m");
  const calib = { scaleRel: 0.0035, offsetRel: 0.0025, timingRel: 0.0008 };
  const g = finalQuantity(out, winner, "g", { name: "Gravitational acceleration", symbol_latex: "g", unit: "m/s²", scaleExp: 1, timeExp: 2 }, calib);
  const conclusion = assembleConclusion(
    out,
    "medium",
    [
      "FF0 is the simplest of three models whose held-out intervals overlap; the drag terms are consistent with zero.",
      "The reviewer accepted with caveats.",
    ],
    [g],
    `A dense ball falls with **constant acceleration**: g = ${plusMinus(g, 3)}. Drag is too small to see in a half-second drop, so adding it does not help; the data **cannot distinguish** the three models and the simplest is reported.`,
    ["A foam ball, which is slowed by the air, would tell the drag models apart. Try one."],
  );
  const review: ReviewThread = {
    status: "accepted_with_caveats",
    rounds: [{
      round: 1, verdict: "accept_with_caveats",
      summary: "Constant g is the right reading of a half-second drop. The uncertainty budget is dominated by the sheet's scale, as it should be.",
      issues: [{
        id: "R1-1", severity: "minor", claim: "g is quoted to a precision finer than the scale uncertainty alone.",
        evidence: `Scale uncertainty is ${(calib.scaleRel * 100).toFixed(2)}% and the plane offset ${(calib.offsetRel * 100).toFixed(2)}%.`,
        requested_action: "add_caveat", detail: "Keep the calibration terms in the headline interval.",
      }],
      response_md: "The headline interval already includes the calibration terms; the budget table lists them.",
    }],
  };

  const scale = 350;
  const m = affine(scale, scale, 640, 500);
  const t = input.t;
  const y = input.y;
  const trial = baseTrial("fall-t1", "fit", sc.id, {
    n: t.length, duration: 0.6, fps: 240, declared: "slowmo", declaredFps: 240,
    video: "drop-t1.mp4", poster: "drop-poster.jpg", mmPerPx: 2.86, sigma: 0.0035, offsetCm: 2.5,
    worldToPx: m, flags: NO_FLAGS, trajectory: true,
  });
  const trial2 = baseTrial("fall-t2", "prove", sc.id, {
    n: t.length, duration: 0.6, fps: 240, declared: "slowmo", declaredFps: 240,
    video: "drop-ghost.mp4", poster: "drop-poster.jpg", mmPerPx: 2.86, sigma: 0.0035, offsetCm: 2.5,
    worldToPx: m, flags: NO_FLAGS, trajectory: false,
  });
  // Ghost for trial 2: drop from 1.15 m, FF0 frozen at trial 1's g, initial conditions from the launch window.
  const model = getModel("FF1")!;
  const rng = mulberry32(sc.trial2.seed);
  const y2 = t.map((x) => model.closedForm!(x, { g: sc.truth.g, b_over_m: sc.truth.b_over_m }, { y0: sc.trial2.y0, v0: 0 })[0] + 0.0015 * gaussian(rng));
  const finalFF0 = fitModel({ model: getModel("FF0")!, fixed: {}, initBase: {}, free: winner.plan.free }, t, y, t.length).values;
  const launch = Math.max(Math.round(0.1 * t.length), Math.round(0.25 * 240));
  const gf = fitModel(
    { model: getModel("FF0")!, fixed: { g: finalFF0.g }, initBase: {}, free: [
      { name: "y0", init: 1.14, scale: 0.01, kind: "initial_condition", unit: "m", label: "y₀" },
      { name: "v0", init: 0, scale: 0.05, kind: "initial_condition", unit: "m/s", label: "v₀" },
    ] }, t, y2, launch);
  const pred = gf.predict(t);
  const errMm = pred.map((p, i) => Math.abs(p - y2[i]) * 1000);
  const tau = Math.max(3 * 1.5, (sc.ball_radius_m * 1000) / 2);
  const track: GhostTrack = {
    t, error_mm: errMm,
    predicted_px: pred.map((v) => toPx(m, 0, v)),
    actual_px: y2.map((v) => toPx(m, 0, v)),
    tau_mm: tau, onset_s: null, launch_window_s: t[launch],
  };
  const ghosts: GhostRender[] = [{
    id: "gh-fall-1", version: 1, trial_id: trial2.id, model_id: "FF0", model_name: "FF0: Constant gravity",
    video_url: media("drop-ghost.mp4"), slow_video_url: media("drop-ghost-slow.mp4"), track_url: "mock:ghost/ex-free-fall/v1",
    poster_url: media("drop-poster.jpg"), divergence_onset_s: null, max_error_mm: Math.max(...errMm.slice(launch)),
    threshold_mm: tau, predicted_ahead_s: t[t.length - 1] - t[launch], revision: null,
  }];
  const whatif: WhatIfContext = {
    system: "free_fall", model_ref: "FF0", params: { g: finalFF0.g }, fixed: {},
    initial: { y0: finalFF0.y0, v0: finalFF0.v0 }, measured_range: { height_m: [1.1, 1.3] },
    object: { name: "Steel ball", radius_m: sc.ball_radius_m, mass_kg: 0.067 }, still_url: media("drop-still.jpg"),
    sim_supported: true, anchor_m: [0, 0],
  };
  const experiment: Experiment = {
    ...summaryOf(sc.id, "Steel ball drop", "free_fall", conclusion, `FF0: ${wName}`, review,
      "A 1.2 m drop at 240 fps. The simplest law wins, and says why it can't say more.", media("drop-poster.jpg"),
      `g = ${plusMinus(g, 3)}`),
    description: "A steel ball dropped 1.2 m beside a wall, filmed at 240 fps.",
    declared_capture: { mode: "slowmo", fps: 240 },
    setup: { plane_offset_cm: 2.5, paper: "a4", pendulum_length_cm: null },
    scene: sceneAnalysis("free_fall", "A steel ball is released from rest beside a wall with an A4 sheet on it, and falls about 1.2 m.",
      [{ key: "ball", sam_prompt: "steel ball", role: "moving_body", why: "The falling object." }],
      { shape: "sphere", material_guess: "steel", approx_size_cm: 4 }, out),
    trials: [trial, trial2], models, conclusion, review, ghosts, whatif, export_formats: [],
    quality_flags: NO_FLAGS, job: null, error: null, last_seq: 0,
  };
  const events = buildNotebook(experiment, out, { favorite: "FF0", rerunnable: true });
  experiment.last_seq = events.length;
  const traj: Trajectory = {
    t, x: null, y, theta: null, phi: null, omega: derivative(t, y, 3),
    px: y.map((v) => toPx(m, 0, v)), split_t: out.splitT, noise_sigma: sc.sigma, main: "y", main_unit: "m",
    events: [{ kind: "release", t: 0 }],
  };
  return { experiment, events, trajectories: { [trial.id]: traj }, tracks: { "ex-free-fall/v1": track } };
}

/* ------------------------------------------------------------------------------------ */
/* 4. "Can't tell": an out-of-plane pendulum                                             */
/* ------------------------------------------------------------------------------------ */

function buildCantTell(): MockBundle {
  const sc = SCENARIOS.cant_tell;
  const { input } = cantTellInput();
  const out = runAnalysis(input);
  const winner = out.fitted.find((f) => f.plan.ref === out.winnerRef)!;
  const models = modelRecords(out, "Nonlinear pendulum with viscous damping", {}, "rad");
  const flags: QualityFlag[] = [
    { id: "oop", severity: "major", title: "Possible out-of-plane motion",
      detail: "The bob's mask area varies by 11% and rises and falls with the swing (|r| = 0.71). The bob is drifting toward and away from the camera." },
    { id: "resid", severity: "minor", title: "Systematic residual",
      detail: "A low-order curve explains more than 30% of what every model leaves behind, which means missing physics (or geometry)." },
  ];
  const calib = { scaleRel: 0.006, offsetRel: 0.02, timingRel: 0.001 };
  const g = finalQuantity(out, winner, "g", { name: "Gravitational acceleration", symbol_latex: "g", unit: "m/s²", scaleExp: 1, timeExp: 2, offsetExp: 1 }, calib);
  // The out-of-plane term is not in any model; widen the interval with it.
  g.budget = [...(g.budget ?? []), { label: "Out-of-plane motion (ΔZ/Z ≈ 5.5%)", sigma: 0.055 * g.value }];
  const wide = Math.sqrt((g.budget ?? []).reduce((s, b) => s + b.sigma ** 2, 0));
  g.ci_low = g.value - wide;
  g.ci_high = g.value + wide;
  const gam = winner.result.params.find((p) => p.name === "gamma")!;
  const conclusion = assembleConclusion(
    out,
    "low",
    [
      "Major quality flag: possible out-of-plane motion.",
      "Every model leaves a systematic pattern in its residuals.",
      "The reviewer's challenge stands after two rounds, so the verdict is contested.",
    ],
    [g],
    `The best fit is a damped pendulum with g = ${plusMinus(g, 2)}, but I do **not trust it**. The damping it needs (γ = ${fmt(gam.value, 2)} 1/s) is far larger than the pivot or air could give, and the video shows the bob moving toward and away from the camera. The fit is probably using damping to imitate a perspective effect.`,
    ["Treat g as unmeasured for this clip."],
    {
      reason: "The bob does not stay in the plane of the sheet, so the pixel-to-meter scale changes during the swing, and no model here can account for that.",
      fixes: [
        "Hang the pendulum from two strings (a V) so it can only swing in one plane.",
        "Film from further back and use the 2x lens, which reduces the perspective error.",
        "Start the swing with a push along the wall, not toward the camera.",
      ],
    },
  );
  const review: ReviewThread = {
    status: "contested",
    rounds: [
      { round: 1, verdict: "challenge", summary: "The claim rests on a damping coefficient that the physics cannot supply.",
        issues: [{ id: "R1-1", severity: "major", claim: "A damped pendulum explains the data.",
          evidence: `γ = ${fmt(gam.value, 2)} 1/s would halve the swing in ${fmt(Math.LN2 / (gam.value / 2), 2)} s, and the mask area flag says the bob changes distance.`,
          requested_action: "lower_confidence", detail: "Lower confidence and say what would settle it." }],
        response_md: "Confidence lowered to **low** and the out-of-plane flag put at the top of the result. I still report P2 as the best of the models tried, as a description and not a law." },
      { round: 2, verdict: "challenge", summary: "Better, but the headline still reports a g value. The residual structure means no model here is right.",
        issues: [{ id: "R2-1", severity: "major", claim: "g = " + fmt(g.value, 2) + " is reported as a result.",
          evidence: "The out-of-plane term (±5.5% in scale) is larger than the statistical interval.", requested_action: "add_caveat", detail: "Show g only with the full budget." }],
        response_md: null },
    ],
  };
  const m = affine(600, 600, 480, 70 + 600 * 0.62);
  const t = input.t;
  const traj: Trajectory = {
    t, x: null, y: null, theta: input.y, phi: null, omega: derivative(t, input.y, 4),
    px: input.y.map((th) => toPx(m, sc.truth.L_eff * Math.sin(th), 0.62 - sc.truth.L_eff * Math.cos(th))),
    split_t: out.splitT, noise_sigma: sc.sigma, main: "theta", main_unit: "rad", events: [{ kind: "release", t: 0 }],
  };
  const trial = baseTrial("cant-t1", "fit", sc.id, {
    n: t.length, duration: sc.window_s + 1, fps: 60, declared: "normal", declaredFps: 60, video: "pendulum-t1.mp4",
    poster: "pendulum-poster.jpg", mmPerPx: 1.67, sigma: 0.006, offsetCm: 2, worldToPx: m, flags, trajectory: true,
  });
  const experiment: Experiment = {
    ...summaryOf(sc.id, "Pendulum that wandered off the wall", "pendulum", conclusion, "P2: damped pendulum (not trusted)", review,
      "The bob drifts toward the camera. Ghostline says it can't tell, and what to change.", media("pendulum-poster.jpg"),
      "Can't tell yet: out-of-plane motion"),
    description: "A swing that started with a push toward the camera.",
    declared_capture: { mode: "normal", fps: 60 },
    setup: { plane_offset_cm: 2, paper: "a4", pendulum_length_cm: 55 },
    scene: sceneAnalysis("pendulum", "A nut on a string, swinging, but the bob also moves toward and away from the camera.",
      [{ key: "bob", sam_prompt: "steel nut", role: "moving_body", why: "The swinging bob." }],
      { shape: "hex nut", material_guess: "steel", approx_size_cm: 2.4 }, out,
      [{ issue: "Motion toward the camera", severity: "high", note: "The bob's apparent size changes during the swing." }]),
    trials: [trial], models, conclusion, review, ghosts: [], whatif: null, export_formats: [],
    quality_flags: flags, job: null, error: null, last_seq: 0,
  };
  const events = buildNotebook(experiment, out, { favorite: "P1", rerunnable: false });
  experiment.last_seq = events.length;
  return { experiment, events, trajectories: { [trial.id]: traj }, tracks: {} };
}

/* ------------------------------------------------------------------------------------ */

const builders: Record<string, () => MockBundle> = {
  "ex-spin-down": buildSpin,
  "ex-pendulum": buildPendulum,
  "ex-free-fall": buildFreeFall,
  "ex-cant-tell": buildCantTell,
};
export const EXAMPLE_ORDER = ["ex-spin-down", "ex-pendulum", "ex-free-fall", "ex-cant-tell"];

const cache = new Map<string, MockBundle>();

export function getBundle(id: string): MockBundle | undefined {
  const hit = cache.get(id);
  if (hit) return hit;
  const b = builders[id];
  if (!b) return undefined;
  const made = b();
  cache.set(id, made);
  return made;
}
