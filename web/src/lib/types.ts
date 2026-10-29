/**
 * Provisional front-end types for the Ghostline API.
 *
 * These follow build-plan Appendix C (schemas) and Part 22 (data model) and are what the
 * UI, the typed client in `api.ts` and the mocks all share. `api-types.ts` is generated
 * from the FastAPI OpenAPI schema and must not be edited by hand. When the back end
 * lands, reconcile by re-exporting the generated schema types from here, for example
 *
 *     export type Experiment = components["schemas"]["ExperimentDetail"];
 *
 * and fixing whatever the compiler flags. Nothing outside this file and `api.ts` should
 * need to change, because every other module imports its types from `@/lib/types`.
 *
 * Media fields (`*_url`) hold either an absolute URL, a site-absolute path ("/mock/x.mp4")
 * or a path relative to `GET /api/media/` ("{exp}/trials/{trial}/playback_720.mp4").
 * `resolveMedia()` in `api.ts` turns any of them into something a <video> can load.
 */

export type SystemType =
  | "free_fall"
  | "projectile"
  | "pendulum"
  | "ramp"
  | "bounce"
  | "spin_down"
  | "oscillator"
  | "other";

export type Confidence = "high" | "medium" | "low";

export type ReviewStatus =
  | "accepted"
  | "accepted_with_caveats"
  | "contested"
  | "not_reviewed"
  | "pending";

export type ExperimentStatus =
  | "draft"
  | "processing" // a trial is being ingested
  | "ready" // trial ingested, waiting for the user to press Analyze
  | "running" // analyze_experiment is running
  | "complete"
  | "failed"
  | "blocked"; // Wonder said proceed=false

export type CaptureMode = "normal" | "slowmo";
export type PaperSize = "a4" | "letter";
export type TrialRole = "fit" | "prove";

/* ------------------------------------------------------------------------------------ */
/* Jobs and pipeline phases                                                              */
/* ------------------------------------------------------------------------------------ */

/** The seven stepper phases of Part 0, in order. */
export const PIPELINE_PHASES = [
  "normalize",
  "watch",
  "track",
  "calibrate",
  "hypothesize",
  "fit",
  "review",
] as const;
export type PipelinePhase = (typeof PIPELINE_PHASES)[number];

/** The phases of a "prove it" job. */
export const PROVE_PHASES = ["normalize", "track", "calibrate", "predict", "render"] as const;
export type ProvePhase = (typeof PROVE_PHASES)[number];

export type JobKind =
  | "ingest_trial"
  | "analyze_experiment"
  | "prove_trial"
  | "revise"
  | "export_verify"
  | "rerun";

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Job {
  id: string;
  kind: JobKind;
  status: JobStatus;
  /** A free-form phase string; `phaseToStep()` maps both our names and the orchestrator's. */
  phase: string;
  /** 0 to 1 */
  progress: number;
  started_at: string;
  heartbeat_at: string | null;
  error: string | null;
}

/* ------------------------------------------------------------------------------------ */
/* Scene, trials, calibration                                                            */
/* ------------------------------------------------------------------------------------ */

export interface TrackedObject {
  key: string;
  sam_prompt: string;
  role: "moving_body" | "marker" | "pivot" | "surface";
  why: string;
}

export interface InitialHypothesis {
  name: string;
  mechanism: string;
  observable_signature: string;
}

export interface QualityFlag {
  id: string;
  severity: "info" | "minor" | "major";
  title: string;
  detail: string;
}

export interface SceneAnalysis {
  system_type: SystemType;
  description: string;
  tracked_objects: TrackedObject[];
  object_properties: { shape?: string; material_guess?: string; approx_size_cm?: number };
  initial_hypotheses: InitialHypothesis[];
  quality_issues: { issue: string; severity: "low" | "medium" | "high"; note: string }[];
  proceed: boolean;
  reason_if_not: string | null;
}

export interface TimingInfo {
  classification: "cfr" | "vfr_jitter" | "mixed" | "retimed";
  detected_fps: number;
  declared_mode: CaptureMode;
  declared_fps: number;
  /** Real seconds per file second (Part 5). 1 for an original file. */
  time_factor: number;
  jitter_pct: number;
  warnings: string[];
}

export interface CalibrationSummary {
  paper: PaperSize;
  mm_per_px: number;
  /** Relative 1-sigma uncertainty of the scale, from the Monte Carlo budget (Part 8). */
  sigma_scale_rel: number;
  plane_offset_cm: number;
  gravity_source: "data" | "plumb" | "camera_level";
  ramp_angle_deg: number | null;
  /** Largest shift of the sheet corners across the clip, in px. */
  camera_motion_px: number;
  /**
   * Row-major 3x3 homography from physics-frame world coordinates (meters, x right, y up)
   * to image pixels, with the plane-offset correction already folded in. The what-if
   * simulator uses it to place a simulated object on the user's own still frame.
   */
  world_to_px: number[] | null;
}

export interface TrialMedia {
  playback_url: string;
  poster_url: string | null;
  calib_overlay_url: string | null;
  /** JSON, `Trajectory` below. */
  trajectory_url: string | null;
  /** JSONL of raw SAM 3.1 Responses events, replayed into @meta-sam/parser (Part 24). */
  sam_events_url: string | null;
}

export interface Trial {
  id: string;
  role: TrialRole;
  status: "uploading" | "ingesting" | "ready" | "tracking" | "tracked" | "failed";
  width: number;
  height: number;
  n_frames: number;
  duration_s: number;
  timing: TimingInfo | null;
  trim: {
    start_s: number;
    end_s: number;
    suggested_start_s: number;
    suggested_end_s: number;
  };
  calibration: CalibrationSummary | null;
  quality_flags: QualityFlag[];
  media: TrialMedia;
  error: string | null;
}

/** `trajectory_url` payload. Positions in meters, angles in radians, time in real seconds. */
export interface Trajectory {
  t: number[];
  x: (number | null)[] | null;
  y: (number | null)[] | null;
  theta: (number | null)[] | null;
  phi: (number | null)[] | null;
  /** Display-only Savitzky-Golay derivative of the main observable. */
  omega: (number | null)[] | null;
  /** Pixel centroid per frame, for the trail overlay on the video. */
  px: ([number, number] | null)[];
  /** End of the training window (first 65% of the motion), in seconds. */
  split_t: number;
  noise_sigma: number;
  /** Which observable to feature in the main chart. */
  main: "x" | "y" | "theta" | "phi";
  main_unit: string;
  events: { kind: "release" | "impact" | "stop"; t: number }[];
}

/* ------------------------------------------------------------------------------------ */
/* Models, fits, conclusion                                                              */
/* ------------------------------------------------------------------------------------ */

export interface ParamEstimate {
  name: string;
  label: string;
  value: number;
  ci_low: number;
  ci_high: number;
  unit: string;
  kind: "physical" | "initial_condition" | "nuisance";
  /** True when the estimate sits on one of its bounds, which the reviewer checks. */
  at_bound?: boolean;
}

export interface FitDiagnostics {
  reduced_chi2: number;
  lag1_autocorr: number;
  durbin_watson: number;
  runs_p: number;
  systematic_residual: boolean;
}

export interface FitSeries {
  t: number[];
  observed: number[];
  predicted: number[];
  /** Training window ends here; the rest is held out. */
  split_t: number;
  unit: string;
  label: string;
}

export interface FitResult {
  id: string;
  params: ParamEstimate[];
  train_rmse: number;
  heldout_rmse: number;
  heldout_ci: [number, number];
  rmse_unit: string;
  /** Held-out RMSE as a fraction of the motion's range. */
  nrmse: number;
  /** First time after the split that the error exceeds 3 sigma of the noise, if ever. */
  time_to_divergence_s: number | null;
  aic: number;
  bic: number;
  diagnostics: FitDiagnostics;
  /** Matplotlib PNGs from the harness. When absent, `series` is drawn client-side. */
  plot_urls: { fit: string | null; residuals: string | null };
  series: FitSeries | null;
  /** Splits (of 50%, 65%, 80%) that this model won against the runner-up. */
  splits_won: number;
}

export type ModelStatus =
  | "proposed"
  | "winner"
  | "tied"
  | "rejected"
  | "invalid";

export interface ModelRecord {
  id: string;
  name: string;
  round: number;
  source: "library" | "wonder";
  /** Library id such as "P2"; maps to a TypeScript twin in `lib/sim` for the what-if. */
  library_ref: string | null;
  mechanism: string;
  equations_latex: string;
  /** 1 is Wonder's favorite, named before any fit. */
  prior_rank: number;
  refuting_signature: string;
  status: ModelStatus;
  rejection_reason: string | null;
  fit: FitResult | null;
}

export interface Quantity {
  name: string;
  /** Symbol in LaTeX, for display. */
  symbol_latex: string;
  value: number;
  ci_low: number;
  ci_high: number;
  unit: string;
  /** Expandable table of where the uncertainty comes from (Part 15). */
  budget: { label: string; sigma: number }[] | null;
  note?: string;
}

export interface RejectedModel {
  id: string;
  reason: string;
  evidence: string;
}

export interface Conclusion {
  winner_id: string;
  quantities: Quantity[];
  rejected: RejectedModel[];
  confidence: Confidence;
  /** Why the rules in Part 15 landed on this level. */
  confidence_reasons: string[];
  caveats: string[];
  narrative_md: string;
  /** Present when confidence is low: the "Ghostline can't tell yet" card. */
  cant_tell: { reason: string; fixes: string[] } | null;
}

/* ------------------------------------------------------------------------------------ */
/* Review                                                                                */
/* ------------------------------------------------------------------------------------ */

export interface ReviewIssue {
  id: string;
  severity: "minor" | "major";
  claim: string;
  evidence: string;
  requested_action: "run_model" | "refit" | "lower_confidence" | "add_caveat" | "none";
  detail: string;
}

export interface ReviewRound {
  round: number;
  verdict: "accept" | "accept_with_caveats" | "challenge";
  issues: ReviewIssue[];
  summary: string;
  /** Wonder's answer to this round, in Markdown. */
  response_md: string | null;
}

export interface ReviewThread {
  status: ReviewStatus;
  rounds: ReviewRound[];
}

/* ------------------------------------------------------------------------------------ */
/* Ghost (prove it)                                                                      */
/* ------------------------------------------------------------------------------------ */

export interface GhostRender {
  id: string;
  version: number;
  trial_id: string;
  model_id: string;
  model_name: string;
  video_url: string;
  /** Slow-motion clips also get a 30 fps slow-playback version. */
  slow_video_url: string | null;
  track_url: string;
  poster_url: string | null;
  divergence_onset_s: number | null;
  max_error_mm: number;
  threshold_mm: number;
  /** How far ahead of the launch window the ghost predicted, in seconds. */
  predicted_ahead_s: number;
  /** Set when v2 was tested against v1 (Part 17). */
  revision: {
    diagnosis: string;
    adopted: boolean;
    note: string;
  } | null;
}

/** `track_url` payload (ghost_track.json). */
export interface GhostTrack {
  t: number[];
  error_mm: number[];
  predicted_px: [number, number][];
  actual_px: [number, number][];
  tau_mm: number;
  onset_s: number | null;
  launch_window_s: number;
}

/* ------------------------------------------------------------------------------------ */
/* What-if                                                                               */
/* ------------------------------------------------------------------------------------ */

/** Systems that have a TypeScript twin in `lib/sim`. */
export type SimulableSystem = "free_fall" | "pendulum" | "ramp" | "bounce" | "spin_down";

export interface WhatIfContext {
  system: SimulableSystem;
  /** Library id of the winning model, e.g. "P2". */
  model_ref: string;
  /** Fitted parameters by library name (see `lib/sim/models`). */
  params: Record<string, number>;
  /** Quantities held fixed in the fit (g, L_eff, alpha, ...). */
  fixed: Record<string, number>;
  /** Trial 1's initial conditions (theta0, omega0, y0, v0, ...). */
  initial: Record<string, number>;
  /** Measured ranges per control, for the extrapolation warning. */
  measured_range: Record<string, [number, number]>;
  object: { name: string; radius_m: number; mass_kg: number | null };
  still_url: string | null;
  /** False for Wonder-written models, which are simulated by POST /simulate. */
  sim_supported: boolean;
  /**
   * Where the motion is anchored in world meters (x right, y up): the pendulum pivot, the
   * spinner hub, the drop point, or the top of the ramp's slope.
   */
  anchor_m: [number, number];
}

/* ------------------------------------------------------------------------------------ */
/* Notebook                                                                              */
/* ------------------------------------------------------------------------------------ */

export type EventActor = "wonder" | "spark_open" | "harness" | "sam" | "system" | "user";
export type EventKind =
  | "observation"
  | "hypothesis"
  | "prediction"
  | "fit"
  | "rejection"
  | "acceptance"
  | "review"
  | "response"
  | "revision"
  | "ghost"
  | "whatif"
  | "warning";

export interface NotebookEvent {
  id: number;
  experiment_id: string;
  seq: number;
  ts: string;
  phase: string;
  actor: EventActor;
  kind: EventKind;
  title: string;
  body_md: string;
  /** Raw details: parameters, code, JSON. `model_id` marks a fit that can be re-run. */
  data: Record<string, unknown>;
  attachments: { kind: "image" | "video" | "file"; url: string; caption?: string }[];
  refs: number[];
}

/* ------------------------------------------------------------------------------------ */
/* Experiments                                                                           */
/* ------------------------------------------------------------------------------------ */

export interface ExperimentSummary {
  id: string;
  title: string;
  system_type: SystemType;
  status: ExperimentStatus;
  created_at: string;
  is_example: boolean;
  confidence: Confidence | null;
  review_status: ReviewStatus | null;
  /** One line such as "g = 9.79 ± 0.07 m/s²". */
  headline: string | null;
  winner_name: string | null;
  thumbnail_url: string | null;
  blurb: string | null;
}

export interface Experiment extends ExperimentSummary {
  description: string;
  declared_capture: { mode: CaptureMode; fps: number };
  setup: {
    plane_offset_cm: number;
    paper: PaperSize;
    pendulum_length_cm: number | null;
  };
  scene: SceneAnalysis | null;
  trials: Trial[];
  models: ModelRecord[];
  conclusion: Conclusion | null;
  review: ReviewThread | null;
  ghosts: GhostRender[];
  whatif: WhatIfContext | null;
  /** Formats `GET /export` can produce for the winner. Empty or absent hides the button. */
  export_formats: ("json" | "pybullet")[];
  quality_flags: QualityFlag[];
  job: Job | null;
  error: string | null;
  /** Highest notebook seq at the time of the response. */
  last_seq: number;
  /** Admin only. */
  cost_usd?: number;
}

/* ------------------------------------------------------------------------------------ */
/* Requests and responses                                                                */
/* ------------------------------------------------------------------------------------ */

export interface CreateExperimentRequest {
  title: string;
  system_hint: SystemType | "unknown";
  description: string;
  declared_capture: { mode: CaptureMode; fps: number };
  plane_offset_cm: number;
  paper: PaperSize;
  pendulum_length_cm: number | null;
}

export interface UploadStartRequest {
  experiment_id: string;
  role: TrialRole;
  filename: string;
  size: number;
}

export interface UploadSession {
  upload_id: string;
  /** The server's chunk size; the client uses 8 MiB unless the server says otherwise. */
  chunk_size: number;
  /** Chunk numbers the server already holds (for resuming). */
  received_chunks: number[];
}

export interface EventsResponse {
  events: NotebookEvent[];
  job: Job | null;
  status: ExperimentStatus;
  last_seq: number;
}

export interface RerunRequest {
  split_pct: number;
  initial_guesses?: Record<string, number>;
}

export interface SimulateRequest {
  params: Record<string, number>;
  initial: Record<string, number>;
  t_end: number;
  dt: number;
}

export interface SimulateResponse {
  t: number[];
  /** One array per observable. */
  observables: number[][];
}

export interface ExplainRequest {
  summary: string;
  numbers: Record<string, number | string>;
}

export interface ExportFile {
  filename: string;
  content_type: string;
  text: string;
}

export interface ReplayCheck {
  simulator: string;
  quantity: string;
  simulated: number;
  measured: number;
  ci95: [number, number];
  unit: string;
  passed: boolean;
}

export interface BenchmarkRow {
  clip_id: string;
  experiment_id: string | null;
  system: SystemType;
  object: string;
  quantity: string;
  unit: string;
  truth: number;
  truth_sigma: number;
  truth_method: string;
  measured: number | null;
  ci_low: number | null;
  ci_high: number | null;
  z: number | null;
  covered: boolean | null;
  model_selected: string | null;
  model_expected: string | null;
  model_correct: boolean | null;
  confidence: Confidence | null;
  /** For the deliberately bad clips: did Ghostline flag the right problem? */
  is_failure_case: boolean;
  failure_expected: string | null;
  failure_flagged: boolean | null;
  note: string | null;
}

export interface BenchmarkTable {
  rows: BenchmarkRow[];
  summary: {
    n_clips: number;
    coverage: number;
    model_selection_accuracy: number;
    failure_cases_flagged: string;
    generated_at: string;
  };
}

export interface AdminOverview {
  budget: { daily_usd: number; spent_today_usd: number; per_experiment_usd: number };
  cache: { hit_rate: number; calls_24h: number };
  jobs: (Job & { experiment_id: string; experiment_title: string })[];
  experiments: { id: string; title: string; cost_usd: number; status: ExperimentStatus; llm_mode: string }[];
}
