/**
 * Front-end types for the Ghostline API.
 *
 * Everything the API returns is a re-export of a schema generated from the FastAPI app
 * (`api-types.ts`, `npm run gen:api`), so a field renamed on the server fails the build here.
 * The response models live in `backend/ghostline/api/views.py`. What remains hand-written is
 * what only the UI knows: the stepper phases, the shapes of the forms it sends, and the file
 * the export dialog saves.
 *
 * Media fields (`*_url`) hold a path relative to `GET /api/media/` ("{exp}/trials/{trial}/x.mp4"),
 * an "api/..." path for the one computed file (the trajectory JSON), an absolute URL, or a
 * site-absolute path ("/mock/x.mp4"). `resolveMedia()` in `api.ts` turns any of them into
 * something a <video> can load.
 */
import type { components } from "@/lib/api-types";
import type { ChunkTransport } from "@/lib/upload/chunked";

type S = components["schemas"];

export type SystemType = S["SceneView"]["system_type"];
export type Confidence = S["ConclusionView"]["confidence"];
export type ReviewStatus = NonNullable<S["ExperimentSummaryView"]["review_status"]>;
/** "blocked" is a failed analysis whose reason was that Wonder found nothing to measure. */
export type ExperimentStatus = S["ExperimentSummaryView"]["status"];
export type CaptureMode = S["DeclaredCaptureView"]["mode"];
export type PaperSize = S["SetupView"]["paper"];
export type TrialRole = S["TrialView"]["role"];

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

export type Job = S["Job"];
export type JobKind = Job["kind"];
export type JobStatus = Job["status"];

/* ------------------------------------------------------------------------------------ */
/* Scene, trials, calibration                                                            */
/* ------------------------------------------------------------------------------------ */

export type TrackedObject = S["TrackedObjectView"];
export type SceneAnalysis = S["SceneView"];
export type QualityFlag = S["QualityFlagView"];
export type TimingInfo = S["TimingView"];
/** `world_to_px` is a row-major 3x3 from physics-frame metres (x right, y up) to image pixels. */
export type CalibrationSummary = S["CalibrationView"];
export type TrialMedia = S["TrialMediaView"];
export type Trial = S["TrialView"];
/** `trajectory_url` payload. Positions in metres, angles in radians, time in real seconds. */
export type Trajectory = S["TrajectoryView"];

/* ------------------------------------------------------------------------------------ */
/* Models, fits, conclusion                                                              */
/* ------------------------------------------------------------------------------------ */

export type ParamEstimate = S["ParamEstimateView"];
export type FitDiagnostics = S["FitDiagnosticsView"];
export type FitSeries = S["FitSeriesView"];
export type FitResult = S["FitView"];
export type ModelRecord = S["ModelView"];
export type ModelStatus = ModelRecord["status"];
export type Quantity = S["QuantityView"];
export type RejectedModel = S["RejectedView"];
export type Conclusion = S["ConclusionView"];

/* ------------------------------------------------------------------------------------ */
/* Review                                                                                */
/* ------------------------------------------------------------------------------------ */

export type ReviewIssue = S["ReviewIssueView"];
export type ReviewRound = S["ReviewRoundView"];
export type ReviewThread = S["ReviewThreadView"];

/* ------------------------------------------------------------------------------------ */
/* Ghost (prove it)                                                                      */
/* ------------------------------------------------------------------------------------ */

export type GhostRender = S["GhostView"];
/** `track_url` payload (ghost/v{n}_track.json), written by the stage that renders the ghost. */
export type GhostTrack = S["GhostTrackView"];

/* ------------------------------------------------------------------------------------ */
/* What-if                                                                               */
/* ------------------------------------------------------------------------------------ */

/** Systems that have a TypeScript twin in `lib/sim`. */
export type SimulableSystem = S["WhatIfView"]["system"];
export type WhatIfContext = S["WhatIfView"];

/* ------------------------------------------------------------------------------------ */
/* Notebook                                                                              */
/* ------------------------------------------------------------------------------------ */

export type NotebookEvent = S["EventView"];
export type EventActor = NotebookEvent["actor"];
export type EventKind = NotebookEvent["kind"];

/* ------------------------------------------------------------------------------------ */
/* Experiments                                                                           */
/* ------------------------------------------------------------------------------------ */

export type ExperimentSummary = S["ExperimentSummaryView"];
export type Experiment = S["ExperimentView"];

/* ------------------------------------------------------------------------------------ */
/* Requests and responses                                                                */
/* ------------------------------------------------------------------------------------ */

/** What the wizard collects. `api.ts` turns it into the nested body `POST /experiments` takes. */
export interface CreateExperimentRequest {
  title: string;
  system_hint: SystemType | "unknown";
  description: string;
  declared_capture: { mode: CaptureMode; fps: number };
  plane_offset_cm: number;
  paper: PaperSize;
  pendulum_length_cm: number | null;
}

export type UploadStartRequest = S["UploadCreate"];
export type UploadSession = Pick<S["UploadInfo"], "upload_id" | "chunk_size" | "received_chunks">;
export type EventsResponse = S["EventsView"];
export type RerunRequest = S["RerunRequest"];
export type SimulateRequest = S["SimulateBody"];
export type SimulateResponse = S["SimulateResponse"];
export type ExplainRequest = S["ExplainRequest"];
export type ExplainResponse = S["ExplainResponse"];

export interface ExportFile {
  filename: string;
  content_type: string;
  text: string;
}

/** The `replay_check` of an exported simulator file (Part 19). */
export interface ReplayCheck {
  simulator: string;
  quantity: string;
  simulated: number;
  measured: number;
  ci95: [number, number];
  unit: string;
  passed: boolean;
}

export type BenchmarkRow = S["BenchmarkRowView"];
export type BenchmarkTable = S["BenchmarkTableView"];
export type AdminOverview = S["AdminOverview"];

export type { ChunkTransport };
