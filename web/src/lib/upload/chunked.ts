import type { TrialRole, UploadSession, UploadStartRequest } from "@/lib/types";

/**
 * Chunked, resumable upload (build-plan Part 4).
 *
 * 1. `start` -> upload_id.
 * 2. `sendChunk` for each 8 MiB `Blob.slice`, three at a time. A failed chunk is retried up to
 *    five times with exponential backoff. The server treats a repeated chunk as a no-op, so a
 *    retry or a resume never corrupts anything.
 * 3. `complete` -> trial_id.
 *
 * The upload id and the chunk numbers that went through are kept in sessionStorage, inside
 * try/catch, so a reload (or a phone that suspended the tab) can pick up where it stopped
 * once the user chooses the same file again. When the device is offline the retries are
 * parked until the browser reports it is back, rather than burning through the five tries
 * during a few seconds in airplane mode.
 */

export const CHUNK_SIZE = 8 * 1024 * 1024;
export const PARALLEL_CHUNKS = 3;
export const MAX_RETRIES = 5;
/** The Meta Files API limit (build-plan Part 4). */
export const MAX_FILE_BYTES = 1024 * 1024 * 1024;

export interface ChunkTransport {
  start(req: UploadStartRequest): Promise<UploadSession>;
  sendChunk(
    uploadId: string,
    index: number,
    body: Blob,
    onProgress: (loadedBytes: number) => void,
    signal: AbortSignal,
  ): Promise<void>;
  complete(uploadId: string): Promise<{ trial_id: string }>;
}

export interface ResumeRecord {
  uploadId: string;
  experimentId: string;
  role: TrialRole;
  filename: string;
  size: number;
  chunkSize: number;
  received: number[];
}

export interface ResumeStore {
  load(experimentId: string, role: TrialRole): ResumeRecord | null;
  save(record: ResumeRecord): void;
  clear(experimentId: string, role: TrialRole): void;
}

export interface UploadProgress {
  sentBytes: number;
  totalBytes: number;
  chunksDone: number;
  chunkCount: number;
  /** "retrying" while a chunk is waiting out its backoff, "offline" while parked. */
  state: "uploading" | "retrying" | "offline" | "finishing";
  /** Which attempt the slowest chunk is on (1 = first try). */
  attempt: number;
  resumed: boolean;
}

export class UploadError extends Error {
  constructor(
    message: string,
    readonly kind: "too_large" | "empty" | "chunk_failed" | "aborted",
    readonly chunkIndex?: number,
    readonly cause?: unknown,
  ) {
    super(message);
    this.name = "UploadError";
  }
}

export interface UploadOptions {
  transport: ChunkTransport;
  experimentId: string;
  role: TrialRole;
  onProgress?: (p: UploadProgress) => void;
  signal?: AbortSignal;
  store?: ResumeStore;
  chunkSize?: number;
  parallel?: number;
  maxRetries?: number;
  /** Injected for tests. */
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  random?: () => number;
  isOffline?: () => boolean;
  waitForOnline?: (signal: AbortSignal) => Promise<void>;
}

interface StatusError {
  status?: number;
}

/** Errors that will fail the same way if sent again. */
export function isPermanent(err: unknown): boolean {
  const status = (err as StatusError | null)?.status;
  if (typeof status !== "number") return false;
  if (status === 408 || status === 429) return false;
  return status >= 400 && status < 500;
}

/** Backoff before retry number `retry` (1-based): 0.5 s, 1 s, 2 s, 4 s, 8 s, with jitter. */
export function backoffMs(retry: number, random: () => number = Math.random): number {
  const base = Math.min(8000, 500 * 2 ** (retry - 1));
  return Math.round(base * (0.85 + 0.3 * random()));
}

export function chunkCount(size: number, chunkSize = CHUNK_SIZE): number {
  return Math.ceil(size / chunkSize);
}

function defaultSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new UploadError("Upload cancelled", "aborted"));
    const id = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(id);
      reject(new UploadError("Upload cancelled", "aborted"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function defaultIsOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

function defaultWaitForOnline(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new UploadError("Upload cancelled", "aborted"));
    const done = () => {
      window.removeEventListener("online", onOnline);
      signal.removeEventListener("abort", onAbort);
    };
    const onOnline = () => {
      done();
      resolve();
    };
    const onAbort = () => {
      done();
      reject(new UploadError("Upload cancelled", "aborted"));
    };
    window.addEventListener("online", onOnline);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export const sessionResumeStore: ResumeStore = {
  load(experimentId, role) {
    try {
      const raw = window.sessionStorage.getItem(storageKey(experimentId, role));
      return raw ? (JSON.parse(raw) as ResumeRecord) : null;
    } catch {
      return null;
    }
  },
  save(record) {
    try {
      window.sessionStorage.setItem(storageKey(record.experimentId, record.role), JSON.stringify(record));
    } catch {
      /* private mode or storage full: resuming simply will not be offered */
    }
  },
  clear(experimentId, role) {
    try {
      window.sessionStorage.removeItem(storageKey(experimentId, role));
    } catch {
      /* ignore */
    }
  },
};

function storageKey(experimentId: string, role: TrialRole): string {
  return `ghostline.upload.${experimentId}.${role}`;
}

/** What the UI can show before the user picks the file again after a reload. */
export function pendingUpload(
  experimentId: string,
  role: TrialRole,
  store: ResumeStore = sessionResumeStore,
): ResumeRecord | null {
  const rec = store.load(experimentId, role);
  if (!rec || rec.received.length === 0) return null;
  return rec;
}

export async function uploadFile(
  file: Blob & { name: string },
  opts: UploadOptions,
): Promise<{ trial_id: string; resumed: boolean }> {
  const chunkSize = opts.chunkSize ?? CHUNK_SIZE;
  const parallel = opts.parallel ?? PARALLEL_CHUNKS;
  const maxRetries = opts.maxRetries ?? MAX_RETRIES;
  const sleep = opts.sleep ?? defaultSleep;
  const random = opts.random ?? Math.random;
  const isOffline = opts.isOffline ?? defaultIsOffline;
  const waitForOnline = opts.waitForOnline ?? defaultWaitForOnline;
  const store = opts.store ?? sessionResumeStore;
  const { transport, experimentId, role } = opts;

  if (file.size === 0) throw new UploadError("That file is empty.", "empty");
  if (file.size > MAX_FILE_BYTES) {
    throw new UploadError("That clip is over 1 GiB. Trim it in Photos or film a shorter one.", "too_large");
  }

  const controller = new AbortController();
  const abort = () => controller.abort();
  if (opts.signal) {
    if (opts.signal.aborted) controller.abort();
    else opts.signal.addEventListener("abort", abort, { once: true });
  }
  const signal = controller.signal;

  const total = chunkCount(file.size, chunkSize);
  const received = new Set<number>();
  const inflight = new Map<number, number>();
  let uploadId: string;
  let resumed = false;

  const saved = store.load(experimentId, role);
  if (
    saved &&
    saved.filename === file.name &&
    saved.size === file.size &&
    saved.chunkSize === chunkSize &&
    saved.received.length > 0
  ) {
    uploadId = saved.uploadId;
    saved.received.forEach((n) => received.add(n));
    resumed = true;
  } else {
    store.clear(experimentId, role);
    const session = await transport.start({
      experiment_id: experimentId,
      role,
      filename: file.name,
      size: file.size,
    });
    uploadId = session.upload_id;
    session.received_chunks.forEach((n) => received.add(n));
    resumed = session.received_chunks.length > 0;
  }

  const persist = () =>
    store.save({
      uploadId,
      experimentId,
      role,
      filename: file.name,
      size: file.size,
      chunkSize,
      received: [...received].sort((a, b) => a - b),
    });
  persist();

  let state: UploadProgress["state"] = "uploading";
  let attempt = 1;
  const sizeOf = (n: number) => Math.min(chunkSize, file.size - n * chunkSize);
  const emit = () => {
    let sent = 0;
    received.forEach((n) => (sent += sizeOf(n)));
    inflight.forEach((loaded) => (sent += loaded));
    opts.onProgress?.({
      sentBytes: Math.min(sent, file.size),
      totalBytes: file.size,
      chunksDone: received.size,
      chunkCount: total,
      state,
      attempt,
      resumed,
    });
  };
  emit();

  const queue: number[] = [];
  for (let n = 0; n < total; n++) if (!received.has(n)) queue.push(n);

  async function sendWithRetry(index: number): Promise<void> {
    const body = file.slice(index * chunkSize, index * chunkSize + sizeOf(index));
    let tries = 0;
    for (;;) {
      if (signal.aborted) throw new UploadError("Upload cancelled", "aborted");
      try {
        inflight.set(index, 0);
        await transport.sendChunk(
          uploadId,
          index,
          body,
          (loaded) => {
            inflight.set(index, loaded);
            emit();
          },
          signal,
        );
        inflight.delete(index);
        received.add(index);
        persist();
        state = "uploading";
        attempt = 1;
        emit();
        return;
      } catch (err) {
        inflight.delete(index);
        if (signal.aborted) throw new UploadError("Upload cancelled", "aborted", index, err);
        if (isPermanent(err)) {
          throw new UploadError(`The server rejected part ${index + 1} of the clip.`, "chunk_failed", index, err);
        }
        if (isOffline()) {
          // Parked, not failed: wait for the network without using up a retry.
          state = "offline";
          emit();
          await waitForOnline(signal);
          state = "retrying";
          emit();
          continue;
        }
        tries += 1;
        if (tries > maxRetries) {
          throw new UploadError(
            `Part ${index + 1} of the clip would not upload after ${maxRetries} retries.`,
            "chunk_failed",
            index,
            err,
          );
        }
        state = "retrying";
        attempt = tries + 1;
        emit();
        await sleep(backoffMs(tries, random), signal);
      }
    }
  }

  async function worker(): Promise<void> {
    for (;;) {
      if (signal.aborted) return;
      const index = queue.shift();
      if (index === undefined) return;
      await sendWithRetry(index);
    }
  }

  try {
    const workers = Array.from({ length: Math.min(parallel, Math.max(queue.length, 1)) }, () => worker());
    // If one worker fails, stop the others instead of letting them keep sending. The first
    // rejection is the real failure; the aborts it causes in its siblings come later.
    await Promise.all(
      workers.map((w) =>
        w.catch((err) => {
          controller.abort();
          throw err;
        }),
      ),
    );
  } finally {
    opts.signal?.removeEventListener("abort", abort);
  }
  // A cancel that arrived while the last chunks were in flight still means "do not finish".
  if (signal.aborted) throw new UploadError("Upload cancelled", "aborted");

  state = "finishing";
  emit();
  const { trial_id } = await transport.complete(uploadId);
  store.clear(experimentId, role);
  return { trial_id, resumed };
}

/** An in-memory ResumeStore, for tests. */
export function memoryResumeStore(): ResumeStore & { data: Map<string, ResumeRecord> } {
  const data = new Map<string, ResumeRecord>();
  return {
    data,
    load: (e, r) => data.get(storageKey(e, r)) ?? null,
    save: (rec) => void data.set(storageKey(rec.experimentId, rec.role), { ...rec, received: [...rec.received] }),
    clear: (e, r) => void data.delete(storageKey(e, r)),
  };
}
