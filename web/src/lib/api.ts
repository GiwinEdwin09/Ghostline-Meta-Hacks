import type { components } from "@/lib/api-types";
import { ApiError, NetworkError } from "@/lib/errors";
import { currentPasscode, promptForPasscode } from "@/lib/passcode";
import type { ChunkTransport } from "@/lib/upload/chunked";
import type {
  AdminOverview,
  BenchmarkTable,
  CreateExperimentRequest,
  EventsResponse,
  Experiment,
  ExperimentSummary,
  ExplainRequest,
  ExportFile,
  GhostTrack,
  Job,
  RerunRequest,
  SimulateRequest,
  SimulateResponse,
  Trajectory,
} from "@/lib/types";

/**
 * The typed API layer.
 *
 * Everything the UI knows about the back end goes through `api` below, which implements the
 * `ApiBackend` interface. With `NEXT_PUBLIC_USE_MOCKS=1` the interface is served by
 * `lib/mocks` (loaded lazily, so it is not in the production bundle when the flag is off);
 * otherwise by the HTTP implementation in this file. Endpoints follow build-plan Part 21.
 *
 * Request and response types are the provisional ones in `lib/types.ts`. Once the back end's
 * OpenAPI schema is generated into `api-types.ts`, swap them there.
 */

export type Health = components["schemas"]["Health"];

const DEFAULT_API_BASE = "http://localhost:8000";

export const USE_MOCKS = process.env.NEXT_PUBLIC_USE_MOCKS === "1";

export function apiBase(): string {
  return (process.env.NEXT_PUBLIC_API_BASE ?? DEFAULT_API_BASE).replace(/\/+$/, "");
}

export function apiUrl(
  path: string,
  base: string = process.env.NEXT_PUBLIC_API_BASE ?? DEFAULT_API_BASE,
): string {
  return `${base.replace(/\/+$/, "")}/api/${path.replace(/^\/+/, "")}`;
}

/**
 * Turn a media field into a URL a <video>, <img> or fetch can load: absolute URLs and
 * site paths are used as they are, anything else is a path under `GET /api/media/`.
 */
export function resolveMedia(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  if (/^(https?:|data:|blob:|\/)/.test(url)) return url;
  return apiUrl(`media/${url}`);
}

/* ------------------------------------------------------------------------------------ */
/* The interface                                                                         */
/* ------------------------------------------------------------------------------------ */

export interface ApiBackend {
  health(): Promise<Health>;
  listExamples(): Promise<ExperimentSummary[]>;
  getExperiment(id: string): Promise<Experiment>;
  getEvents(id: string, after: number): Promise<EventsResponse>;
  createExperiment(req: CreateExperimentRequest): Promise<{ id: string }>;
  setTrim(trialId: string, trim: { start_s: number; end_s: number }): Promise<void>;
  analyze(experimentId: string): Promise<Job | null>;
  prove(experimentId: string, trialId: string): Promise<Job | null>;
  revise(experimentId: string): Promise<Job | null>;
  rerunModel(experimentId: string, modelId: string, req: RerunRequest): Promise<Job | null>;
  simulate(experimentId: string, req: SimulateRequest): Promise<SimulateResponse>;
  explain(experimentId: string, req: ExplainRequest): Promise<{ text: string }>;
  getExport(experimentId: string, format: "json" | "pybullet", massG: number | null): Promise<ExportFile>;
  verifyExport(experimentId: string): Promise<Job | null>;
  deleteExperiment(experimentId: string): Promise<void>;
  getBenchmark(): Promise<BenchmarkTable>;
  getAdminOverview(): Promise<AdminOverview>;
  /** Upload plumbing (build-plan Part 4); the chunk loop itself is `lib/upload/chunked.ts`. */
  uploads: ChunkTransport;
  getTrajectory(url: string): Promise<Trajectory>;
  getGhostTrack(url: string): Promise<GhostTrack>;
  /** Raw SAM 3.1 event stream (JSONL), one object per line. */
  getSamEvents(url: string): Promise<unknown[]>;
}

/* ------------------------------------------------------------------------------------ */
/* HTTP implementation                                                                   */
/* ------------------------------------------------------------------------------------ */

interface RequestOptions {
  query?: Record<string, string | number | undefined>;
  body?: unknown;
  /** Write routes carry the X-Ghostline-Key passcode (Part 21). */
  write?: boolean;
  signal?: AbortSignal;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const url = apiUrl(path);
  if (!query) return url;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) if (v !== undefined) qs.set(k, String(v));
  const s = qs.toString();
  return s ? `${url}?${s}` : url;
}

async function toApiError(res: Response): Promise<ApiError> {
  let detail: unknown;
  let message = `${res.status} ${res.statusText}`.trim();
  try {
    const data = await res.clone().json();
    detail = data;
    const d = (data as { detail?: unknown }).detail;
    if (typeof d === "string") message = d;
    else if (Array.isArray(d) && d.length > 0 && typeof d[0]?.msg === "string") message = d[0].msg;
  } catch {
    /* not JSON */
  }
  return new ApiError(message, res.status, detail);
}

async function send(method: string, path: string, opts: RequestOptions, passcode: string | null): Promise<Response> {
  const headers: Record<string, string> = {};
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (opts.write && passcode) headers["X-Ghostline-Key"] = passcode;
  try {
    return await fetch(buildUrl(path, opts.query), {
      method,
      headers,
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      signal: opts.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new NetworkError();
  }
}

async function request<T>(method: string, path: string, opts: RequestOptions = {}): Promise<T> {
  let passcode = opts.write ? currentPasscode() : null;
  if (opts.write && !passcode) passcode = await promptForPasscode("missing");
  let res = await send(method, path, opts, passcode);
  if (res.status === 401 && opts.write) {
    const fresh = await promptForPasscode("rejected");
    if (fresh) res = await send(method, path, opts, fresh);
  }
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

async function fetchJson<T>(url: string): Promise<T> {
  const target = resolveMedia(url);
  if (!target) throw new ApiError("Missing file", 404);
  let res: Response;
  try {
    res = await fetch(target);
  } catch {
    throw new NetworkError();
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

/** PUT one raw chunk with XMLHttpRequest, because fetch cannot report upload progress. */
function putChunk(
  uploadId: string,
  index: number,
  body: Blob,
  onProgress: (loaded: number) => void,
  signal: AbortSignal,
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", apiUrl(`uploads/${encodeURIComponent(uploadId)}/chunks/${index}`));
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    const key = currentPasscode();
    if (key) xhr.setRequestHeader("X-Ghostline-Key", key);
    // Modal web requests time out at 150 s (build-plan Part 4); stay under it.
    xhr.timeout = 140_000;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new ApiError(`Chunk ${index} failed`, xhr.status));
    };
    xhr.onerror = () => reject(new NetworkError());
    xhr.ontimeout = () => reject(new NetworkError("The upload timed out."));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    const onAbort = () => xhr.abort();
    signal.addEventListener("abort", onAbort, { once: true });
    xhr.onloadend = () => signal.removeEventListener("abort", onAbort);
    xhr.send(body);
  });
}

function filenameFrom(res: Response, fallback: string): string {
  const cd = res.headers.get("Content-Disposition") ?? "";
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  return m ? decodeURIComponent(m[1]) : fallback;
}

const httpBackend: ApiBackend = {
  health: () => request<Health>("GET", "health"),
  listExamples: () => request<ExperimentSummary[]>("GET", "experiments", { query: { examples: "true" } }),
  getExperiment: (id) => request<Experiment>("GET", `experiments/${encodeURIComponent(id)}`),
  getEvents: (id, after) =>
    request<EventsResponse>("GET", `experiments/${encodeURIComponent(id)}/events`, { query: { after } }),
  createExperiment: (req) => request<{ id: string }>("POST", "experiments", { body: req, write: true }),
  setTrim: (trialId, trim) =>
    request<void>("POST", `trials/${encodeURIComponent(trialId)}/trim`, { body: trim, write: true }),
  analyze: (id) => request<Job | null>("POST", `experiments/${encodeURIComponent(id)}/analyze`, { write: true }),
  prove: (id, trialId) =>
    request<Job | null>("POST", `experiments/${encodeURIComponent(id)}/prove`, {
      body: { trial_id: trialId },
      write: true,
    }),
  revise: (id) => request<Job | null>("POST", `experiments/${encodeURIComponent(id)}/revise`, { write: true }),
  rerunModel: (id, modelId, req) =>
    request<Job | null>(
      "POST",
      `experiments/${encodeURIComponent(id)}/models/${encodeURIComponent(modelId)}/rerun`,
      { body: req, write: true },
    ),
  simulate: (id, req) =>
    request<SimulateResponse>("POST", `experiments/${encodeURIComponent(id)}/simulate`, { body: req }),
  explain: (id, req) =>
    request<{ text: string }>("POST", `experiments/${encodeURIComponent(id)}/explain`, { body: req, write: true }),
  getExport: async (id, format, massG) => {
    const res = await send(
      "GET",
      `experiments/${encodeURIComponent(id)}/export`,
      { query: { format, mass_g: massG ?? undefined } },
      null,
    );
    if (!res.ok) throw await toApiError(res);
    const ext = format === "json" ? "json" : "py";
    return {
      filename: filenameFrom(res, `ghostline-${id}.${ext}`),
      content_type: res.headers.get("Content-Type") ?? "text/plain",
      text: await res.text(),
    };
  },
  verifyExport: (id) =>
    request<Job | null>("POST", `experiments/${encodeURIComponent(id)}/export/verify`, { write: true }),
  deleteExperiment: (id) => request<void>("DELETE", `experiments/${encodeURIComponent(id)}`, { write: true }),
  getBenchmark: () => request<BenchmarkTable>("GET", "benchmark"),
  getAdminOverview: () => request<AdminOverview>("GET", "admin/overview", { write: true }),
  uploads: {
    start: (req) => request("POST", "uploads", { body: req, write: true }),
    sendChunk: putChunk,
    complete: (uploadId) =>
      request("POST", `uploads/${encodeURIComponent(uploadId)}/complete`, { write: true }),
  },
  getTrajectory: (url) => fetchJson<Trajectory>(url),
  getGhostTrack: (url) => fetchJson<GhostTrack>(url),
  getSamEvents: async (url) => {
    const target = resolveMedia(url);
    if (!target) return [];
    const res = await fetch(target);
    if (!res.ok) throw await toApiError(res);
    const text = await res.text();
    return text
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => JSON.parse(line) as unknown);
  },
};

/* ------------------------------------------------------------------------------------ */
/* Backend selection                                                                     */
/* ------------------------------------------------------------------------------------ */

let backendPromise: Promise<ApiBackend> | null = null;

function backend(): Promise<ApiBackend> {
  if (!backendPromise) {
    backendPromise = USE_MOCKS
      ? import("@/lib/mocks").then((m) => m.mockBackend)
      : Promise.resolve(httpBackend);
  }
  return backendPromise;
}

type Fn<K extends keyof ApiBackend> = ApiBackend[K] extends (...args: infer A) => infer R
  ? (...args: A) => R
  : never;

function call<K extends Exclude<keyof ApiBackend, "uploads">>(key: K): Fn<K> {
  return ((...args: unknown[]) =>
    backend().then((b) => (b[key] as (...a: unknown[]) => Promise<unknown>)(...args))) as unknown as Fn<K>;
}

/** The API the UI uses. Methods resolve against the mock or the HTTP back end. */
export const api: ApiBackend = {
  health: call("health"),
  listExamples: call("listExamples"),
  getExperiment: call("getExperiment"),
  getEvents: call("getEvents"),
  createExperiment: call("createExperiment"),
  setTrim: call("setTrim"),
  analyze: call("analyze"),
  prove: call("prove"),
  revise: call("revise"),
  rerunModel: call("rerunModel"),
  simulate: call("simulate"),
  explain: call("explain"),
  getExport: call("getExport"),
  verifyExport: call("verifyExport"),
  deleteExperiment: call("deleteExperiment"),
  getBenchmark: call("getBenchmark"),
  getAdminOverview: call("getAdminOverview"),
  getTrajectory: call("getTrajectory"),
  getGhostTrack: call("getGhostTrack"),
  getSamEvents: call("getSamEvents"),
  uploads: {
    start: (req) => backend().then((b) => b.uploads.start(req)),
    sendChunk: (id, index, body, onProgress, signal) =>
      backend().then((b) => b.uploads.sendChunk(id, index, body, onProgress, signal)),
    complete: (id) => backend().then((b) => b.uploads.complete(id)),
  },
};

/** Back-end health, for the landing page's status line. */
export async function getHealth(): Promise<Health> {
  if (USE_MOCKS) return api.health();
  const response = await fetch(apiUrl("health"));
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}
