import type { ApiBackend } from "@/lib/api";
import { ApiError } from "@/lib/errors";
import { getModel, integrate } from "@/lib/sim";
import {
  PIPELINE_PHASES,
  type AdminOverview,
  type BenchmarkRow,
  type BenchmarkTable,
  type CreateExperimentRequest,
  type EventsResponse,
  type Experiment,
  type ExperimentSummary,
  type GhostTrack,
  type Job,
  type NotebookEvent,
  type Trajectory,
  type Trial,
} from "@/lib/types";
import type { MockBundle } from "./build";
import summaries from "./generated/index.json";

/**
 * The mock back end, switched on with NEXT_PUBLIC_USE_MOCKS=1 (see `lib/api.ts`).
 *
 * The four example experiments are pre-generated JSON (`generated/`, produced by the mock
 * harness in `build.ts`). "Run your own" is simulated too: creating an experiment, a
 * chunked upload that really slices and counts bytes, an ingest that takes a couple of
 * seconds, and an analysis that walks the pipeline phases for about 25 seconds and then
 * hands back the spin-down example under the new experiment's name.
 */

const loaders: Record<string, () => Promise<{ default: unknown }>> = {
  "ex-spin-down": () => import("./generated/ex-spin-down.json"),
  "ex-pendulum": () => import("./generated/ex-pendulum.json"),
  "ex-free-fall": () => import("./generated/ex-free-fall.json"),
  "ex-cant-tell": () => import("./generated/ex-cant-tell.json"),
};

async function bundle(id: string): Promise<MockBundle | null> {
  const load = loaders[id];
  if (!load) return null;
  return (await load()).default as unknown as MockBundle;
}

const delay = (ms = 120) => new Promise<void>((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------------------------ */
/* Experiments made through the wizard                                                   */
/* ------------------------------------------------------------------------------------ */

interface Draft {
  id: string;
  req: CreateExperimentRequest;
  createdAt: number;
  uploadedAt?: number;
  analyzeAt?: number;
}

const ANALYSIS_SECONDS = 26;
const INGEST_SECONDS = 2.5;
const STORE_KEY = "ghostline.mock.drafts";
const drafts = new Map<string, Draft>();
let restored = false;

function restore() {
  if (restored) return;
  restored = true;
  try {
    const raw = window.sessionStorage.getItem(STORE_KEY);
    if (raw) for (const d of JSON.parse(raw) as Draft[]) drafts.set(d.id, d);
  } catch {
    /* sessionStorage unavailable: drafts live for this page view */
  }
}

function persist() {
  try {
    window.sessionStorage.setItem(STORE_KEY, JSON.stringify([...drafts.values()]));
  } catch {
    /* ignore */
  }
}

function draftTrial(d: Draft, base: Trial, ready: boolean): Trial {
  return {
    ...base,
    id: `${d.id}-t1`,
    role: "fit",
    status: ready ? "ready" : "ingesting",
    trim: { start_s: 0.4, end_s: 33.4, suggested_start_s: 0.4, suggested_end_s: 33.4 },
    media: { ...base.media, trajectory_url: null },
  };
}

function emptyExperiment(d: Draft, spin: Experiment): Experiment {
  return {
    ...spin,
    id: d.id,
    title: d.req.title,
    is_example: false,
    status: "draft",
    created_at: new Date(d.createdAt).toISOString(),
    description: d.req.description,
    declared_capture: d.req.declared_capture,
    setup: { plane_offset_cm: d.req.plane_offset_cm, paper: d.req.paper, pendulum_length_cm: d.req.pendulum_length_cm },
    scene: null,
    trials: [],
    models: [],
    conclusion: null,
    review: null,
    ghosts: [],
    whatif: null,
    export_formats: [],
    quality_flags: [],
    confidence: null,
    review_status: null,
    headline: null,
    winner_name: null,
    blurb: null,
    job: null,
    error: null,
    last_seq: 0,
  };
}

interface DraftView {
  experiment: Experiment;
  events: NotebookEvent[];
}

async function viewDraft(d: Draft): Promise<DraftView> {
  const spin = (await bundle("ex-spin-down"))!;
  const now = Date.now();
  const exp = emptyExperiment(d, spin.experiment);
  if (!d.uploadedAt) return { experiment: exp, events: [] };
  const sinceUpload = (now - d.uploadedAt) / 1000;
  if (sinceUpload < INGEST_SECONDS) {
    exp.status = "ingesting";
    exp.trials = [draftTrial(d, spin.experiment.trials[0], false)];
    exp.job = job(d.id, "ingest_trial", "running", "normalize", sinceUpload / INGEST_SECONDS, d.uploadedAt);
    return { experiment: exp, events: [] };
  }
  exp.status = "ready";
  exp.trials = [draftTrial(d, spin.experiment.trials[0], true)];
  if (!d.analyzeAt) return { experiment: exp, events: [] };

  const progress = Math.min(1, (now - d.analyzeAt) / 1000 / ANALYSIS_SECONDS);
  const all = spin.events.map((e) => ({ ...e, experiment_id: d.id }));
  if (progress >= 1) {
    return {
      experiment: {
        ...spin.experiment,
        id: d.id,
        title: d.req.title,
        is_example: false,
        created_at: exp.created_at,
        declared_capture: d.req.declared_capture,
        trials: [draftTrial(d, spin.experiment.trials[0], true)].map((t) => ({ ...t, status: "tracked" as const, media: spin.experiment.trials[0].media })),
      },
      events: all,
    };
  }
  const shown = all.slice(0, Math.max(1, Math.floor(progress * all.length)));
  const phaseIndex = Math.min(PIPELINE_PHASES.length - 1, Math.floor(progress * PIPELINE_PHASES.length));
  exp.status = "analyzing";
  exp.job = job(d.id, "analyze_experiment", "running", PIPELINE_PHASES[phaseIndex], progress, d.analyzeAt);
  exp.last_seq = shown.length;
  return { experiment: exp, events: shown };
}

function job(
  expId: string,
  kind: Job["kind"],
  status: Job["status"],
  phase: string,
  progress: number,
  startedAt: number,
): Job {
  return {
    id: `job-${expId}-${kind}`,
    experiment_id: expId,
    kind,
    status,
    phase: phase as Job["phase"],
    progress,
    params: {},
    attempt: 1,
    retry_of: null,
    created_at: new Date(startedAt).toISOString(),
    started_at: new Date(startedAt).toISOString(),
    heartbeat_at: new Date().toISOString(),
    finished_at: null,
    error: null,
    error_code: null,
    retryable: true,
  };
}

/* ------------------------------------------------------------------------------------ */
/* Benchmark (demo numbers: labelled as such in the UI)                                  */
/* ------------------------------------------------------------------------------------ */

function row(
  clip: string,
  system: BenchmarkRow["system"],
  object: string,
  quantity: string,
  unit: string,
  truth: number,
  sigma: number,
  method: string,
  measured: number | null,
  half: number | null,
  selected: string | null,
  expected: string | null,
  confidence: BenchmarkRow["confidence"],
  experimentId: string | null = null,
  extra: Partial<BenchmarkRow> = {},
): BenchmarkRow {
  const z = measured === null || half === null ? null : (measured - truth) / Math.sqrt((half / 1.96) ** 2 + sigma ** 2);
  return {
    clip_id: clip, experiment_id: experimentId, system, object, quantity, unit, truth, truth_sigma: sigma,
    truth_method: method, measured, ci_low: measured === null || half === null ? null : measured - half,
    ci_high: measured === null || half === null ? null : measured + half, z,
    covered: z === null ? null : Math.abs((measured as number) - truth) <= (half as number) + 2 * sigma,
    model_selected: selected, model_expected: expected, model_correct: selected === null ? null : selected === expected,
    confidence, is_failure_case: false, failure_expected: null, failure_flagged: null, note: null, ...extra,
  };
}

function benchmark(): BenchmarkTable {
  const g = 'Local g from latitude';
  const rows: BenchmarkRow[] = [
    row("ff-dense-1", "free_fall", "steel ball", "g", "m/s²", 9.802, 0.002, g, 9.75, 0.15, "FF0", "FF0", "medium", "ex-free-fall"),
    row("ff-dense-2", "free_fall", "golf ball", "g", "m/s²", 9.802, 0.002, g, 9.83, 0.14, "FF0", "FF0", "medium"),
    row("ff-dense-3", "free_fall", "steel ball", "g", "m/s²", 9.802, 0.002, g, 9.88, 0.16, "FF0", "FF0", "medium"),
    row("ff-foam-1", "free_fall", "foam ball", "g", "m/s²", 9.802, 0.002, g, 9.71, 0.3, "FF2", "FF2", "medium"),
    row("pend-small-1", "pendulum", "nut, 8°", "g", "m/s²", 9.802, 0.002, g, 9.79, 0.12, "P1", "P1", "medium"),
    row("pend-small-2", "pendulum", "nut, 10°", "g", "m/s²", 9.802, 0.002, g, 9.86, 0.13, "P1", "P1", "medium"),
    row("pend-large-1", "pendulum", "nut, 50°", "g", "m/s²", 9.802, 0.002, g, 9.8, 0.1, "P1", "P1", "medium", "ex-pendulum"),
    row("pend-large-2", "pendulum", "nut, 45°", "g", "m/s²", 9.802, 0.002, g, 9.74, 0.11, "P2", "P2", "high"),
    row("ramp-block-1", "ramp", "wooden block", "μ_k", "", 0.27, 0.02, "Tilt until constant speed, 5 repeats", 0.26, 0.04, "R1", "R1", "high"),
    row("ramp-block-2", "ramp", "wooden block", "μ_k", "", 0.27, 0.02, "Tilt until constant speed, 5 repeats", 0.29, 0.05, "R1", "R1", "medium"),
    row("ramp-ball-1", "ramp", "solid ball", "β", "", 0.4, 0.001, "Theory: solid sphere", 0.41, 0.03, "R2", "R2", "high"),
    row("bounce-1", "bounce", "rubber ball", "e", "", 0.81, 0.01, "Audio: Δt ratio, 5 drops", 0.8, 0.03, "B0", "B0", "high"),
    row("bounce-2", "bounce", "rubber ball", "e", "", 0.81, 0.01, "Audio: Δt ratio, 5 drops", 0.83, 0.03, "B0", "B0", "high"),
    row("spin-1", "spin_down", "fidget spinner", "c", "rad/s²", 0.9, 0.05, "Stopwatch over 3 spins", 0.9, 0.06, "S2", "S2", "high", "ex-spin-down"),
    row("fail-handheld", "pendulum", "nut (handheld camera)", "g", "m/s²", 9.802, 0.002, g, null, null, null, null, "low", null, {
      is_failure_case: true, failure_expected: "Camera motion over 3 px", failure_flagged: true, note: "Stopped before fitting and asked for a tripod.",
    }),
    row("fail-retimed", "free_fall", "ball (slow-mo re-timed 8×)", "g", "m/s²", 9.802, 0.002, g, null, null, null, null, "low", null, {
      is_failure_case: true, failure_expected: "Clip looks re-timed", failure_flagged: true, note: "Physics check found g ≈ 9.81/64 and warned.",
    }),
    row("fail-oop", "pendulum", "nut (out of plane)", "g", "m/s²", 9.802, 0.002, g, 9.2, 1.4, "P2", "P1", "low", "ex-cant-tell", {
      is_failure_case: true, failure_expected: "Out-of-plane motion", failure_flagged: true, note: "Low confidence, contested review.",
    }),
  ];
  const scored = rows.filter((r) => !r.is_failure_case && r.covered !== null);
  const failures = rows.filter((r) => r.is_failure_case);
  return {
    rows,
    summary: {
      n_clips: rows.length,
      coverage: scored.filter((r) => r.covered).length / scored.length,
      model_selection_accuracy: scored.filter((r) => r.model_correct).length / scored.length,
      failure_cases_flagged: `${failures.filter((r) => r.failure_flagged).length} of ${failures.length}`,
      generated_at: "2026-11-02T20:00:00Z",
    },
  };
}

/* ------------------------------------------------------------------------------------ */

const admin = (): AdminOverview => ({
  budget: { daily_usd: 20, spent_today_usd: 6.4, per_experiment_usd: 3 },
  cache: { hit_rate: 0.82, calls_24h: 311 },
  jobs: [],
  experiments: [
    { id: "ex-spin-down", title: "Fidget spinner spin-down", cost_usd: 1.12, status: "complete", llm_mode: "replay" },
    { id: "ex-pendulum", title: "Pendulum at 50°", cost_usd: 0.94, status: "complete", llm_mode: "replay" },
    { id: "ex-free-fall", title: "Steel ball drop", cost_usd: 0.71, status: "complete", llm_mode: "replay" },
    { id: "ex-cant-tell", title: "Pendulum that wandered off the wall", cost_usd: 1.4, status: "complete", llm_mode: "replay" },
  ],
});

/* ------------------------------------------------------------------------------------ */
/* The backend                                                                           */
/* ------------------------------------------------------------------------------------ */

const uploads = new Map<string, { experimentId: string; role: "fit" | "prove" }>();

export const mockBackend: ApiBackend = {
  async health() {
    return { status: "ok", version: "mock", llm_mode: "replay", database: "ok", volume: "ok", meta_api: "skipped" };
  },

  async listExamples() {
    await delay(150);
    return summaries as unknown as ExperimentSummary[];
  },

  async getExperiment(id) {
    await delay();
    const b = await bundle(id);
    if (b) return b.experiment;
    restore();
    const d = drafts.get(id);
    if (!d) throw new ApiError("Experiment not found", 404);
    return (await viewDraft(d)).experiment;
  },

  async getEvents(id, after): Promise<EventsResponse> {
    await delay(60);
    const b = await bundle(id);
    if (b) {
      return {
        events: b.events.filter((e) => e.seq > after),
        job: null,
        status: b.experiment.status,
        last_seq: b.events.length,
      };
    }
    restore();
    const d = drafts.get(id);
    if (!d) throw new ApiError("Experiment not found", 404);
    const v = await viewDraft(d);
    return {
      events: v.events.filter((e) => e.seq > after),
      job: v.experiment.job,
      status: v.experiment.status,
      last_seq: v.events.length,
    };
  },

  async createExperiment(req) {
    await delay(200);
    restore();
    const id = `exp-${Math.random().toString(36).slice(2, 8)}`;
    drafts.set(id, { id, req, createdAt: Date.now() });
    persist();
    return { id };
  },

  async setTrim() {
    await delay(100);
  },

  async analyze(experimentId) {
    await delay(150);
    restore();
    const d = drafts.get(experimentId);
    if (!d) throw new ApiError("Demo data: examples are already analyzed. Start a new experiment to see the pipeline run.", 409);
    d.analyzeAt = Date.now();
    persist();
    return job(d.id, "analyze_experiment", "queued", "normalize", 0, d.analyzeAt);
  },

  async prove() {
    await delay(150);
    throw new ApiError(
      "Demo data: this example has no second clip. Open the pendulum or the steel-ball example to see a ghost.",
      422,
    );
  },

  async revise() {
    await delay(150);
    throw new ApiError("Demo data: the pendulum example already includes the diagnosis and ghost v2.", 409);
  },

  async rerunModel(experimentId, modelId, req) {
    await delay(300);
    return {
      ...job(experimentId, "rerun", "succeeded", "fitting", 1, Date.now()),
      id: `job-rerun-${experimentId}-${modelId}`,
      error: req.split_pct < 20 ? "Split too small" : null,
    };
  },

  async simulate(experimentId, req) {
    await delay(900);
    const b = await bundle(experimentId);
    const w = b?.experiment.whatif;
    const model = w ? getModel(w.model_ref) : undefined;
    if (!w || !model) throw new ApiError("Nothing to simulate for this experiment.", 404);
    const res = integrate(model, req.params ?? {}, req.initial ?? {}, { tEnd: req.t_end, dt: req.dt });
    return { model_id: w.model_ref, t: res.t, series: {}, derived: {}, observables: res.obs };
  },

  async explain(_id, req) {
    await delay(1200);
    return {
      cached: false,
      text: `${req.summary} The numbers above come from the same equation you measured, so the change you made follows directly from the physics rather than from a new fit.`,
    };
  },

  async getExport(experimentId, format, massG) {
    await delay(200);
    const b = await bundle(experimentId);
    if (!b || format !== "json" || b.experiment.export_formats.length === 0) {
      throw new ApiError("This experiment has no exportable parameters in that format.", 404);
    }
    const exp = b.experiment;
    const params = exp.conclusion!.quantities.map((q) => ({
      name: q.symbol_latex,
      value: q.value,
      ci95: [q.ci_low, q.ci_high],
      unit: q.unit,
      definition: q.name,
      from_model: exp.conclusion!.winner_id,
    }));
    const text = JSON.stringify(
      {
        schema_version: "ghostline.sim/1",
        experiment_url: `https://ghostline.example/e/${exp.id}`,
        object: exp.whatif?.object.name ?? "object",
        surface: null,
        confidence: exp.conclusion!.confidence,
        mass_kg: massG ? massG / 1000 : null,
        parameters: params,
        randomization: Object.fromEntries(params.map((p) => [p.name, p.ci95])),
        valid_for: { omega0_rad_s: [50, 60], note: "Per unit of inertia; outside these ranges is extrapolation." },
        replay_check: null,
      },
      null,
      2,
    );
    return { filename: `ghostline-${exp.id}.json`, content_type: "application/json", text };
  },

  async verifyExport() {
    await delay(200);
    throw new ApiError("Demo data: the PyBullet replay check is not available in mock mode.", 422);
  },

  async deleteExperiment(id) {
    await delay(150);
    restore();
    if (!drafts.delete(id)) throw new ApiError("Examples can't be deleted.", 403);
    persist();
  },

  async getBenchmark() {
    await delay(150);
    return benchmark();
  },

  async getAdminOverview() {
    await delay(150);
    return admin();
  },

  uploads: {
    async start(req) {
      await delay(150);
      const id = `up-${req.experiment_id}-${req.role}`;
      uploads.set(id, { experimentId: req.experiment_id, role: req.role });
      return { upload_id: id, chunk_size: 8 * 1024 * 1024, received_chunks: [] };
    },
    sendChunk(_id, _index, body, onProgress, signal) {
      // ~10 MiB/s, in 50 ms ticks, so the progress bar moves like a decent Wi-Fi connection.
      return new Promise<void>((resolve, reject) => {
        let sent = 0;
        const step = Math.max(1, Math.round((10 * 1024 * 1024) / 20));
        const timer = setInterval(() => {
          sent = Math.min(body.size, sent + step);
          onProgress(sent);
          if (sent >= body.size) {
            clearInterval(timer);
            signal.removeEventListener("abort", onAbort);
            resolve();
          }
        }, 50);
        const onAbort = () => {
          clearInterval(timer);
          reject(new DOMException("Aborted", "AbortError"));
        };
        signal.addEventListener("abort", onAbort, { once: true });
      });
    },
    async complete(uploadId) {
      await delay(200);
      restore();
      const u = uploads.get(uploadId);
      const id = u?.experimentId ?? uploadId.replace(/^up-/, "").replace(/-(fit|prove)$/, "");
      const d = drafts.get(id);
      if (d && (u?.role ?? "fit") === "fit") {
        d.uploadedAt = Date.now();
        persist();
      }
      return { trial_id: `${id}-t1` };
    },
  },

  async getTrajectory(url): Promise<Trajectory> {
    await delay(100);
    const [, expId, trialId] = url.split("/");
    const b = await bundle(expId);
    const t = b?.trajectories[trialId];
    if (!t) throw new ApiError("Trajectory not found", 404);
    return t;
  },

  async getGhostTrack(url): Promise<GhostTrack> {
    await delay(100);
    const key = url.replace(/^mock:ghost\//, "");
    const expId = key.split("/")[0];
    const b = await bundle(expId);
    const t = b?.tracks[key];
    if (!t) throw new ApiError("Ghost track not found", 404);
    return t;
  },

  async getSamEvents() {
    return [];
  },
};
