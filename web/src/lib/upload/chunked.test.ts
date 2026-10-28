import { describe, expect, it } from "vitest";

import {
  CHUNK_SIZE,
  MAX_FILE_BYTES,
  UploadError,
  backoffMs,
  chunkCount,
  isPermanent,
  memoryResumeStore,
  uploadFile,
  type ChunkTransport,
  type UploadProgress,
} from "@/lib/upload/chunked";

const MiB = 1024 * 1024;

function fakeFile(size: number, name = "pendulum.mov"): Blob & { name: string } {
  // A real Blob with the right size; contents do not matter.
  return Object.assign(new Blob([new Uint8Array(size)]), { name });
}

class StatusError extends Error {
  constructor(readonly status: number) {
    super(`HTTP ${status}`);
  }
}

interface Calls {
  started: number;
  sent: number[];
  sizes: Map<number, number>;
  maxConcurrent: number;
  completed: string[];
}

function transportWith(
  behavior: (index: number, attempt: number) => Promise<void> | void = () => {},
): { transport: ChunkTransport; calls: Calls } {
  const calls: Calls = { started: 0, sent: [], sizes: new Map(), maxConcurrent: 0, completed: [] };
  const attempts = new Map<number, number>();
  let concurrent = 0;
  const transport: ChunkTransport = {
    async start() {
      calls.started += 1;
      return { upload_id: "up_1", chunk_size: CHUNK_SIZE, received_chunks: [] };
    },
    async sendChunk(_id, index, body, onProgress) {
      concurrent += 1;
      calls.maxConcurrent = Math.max(calls.maxConcurrent, concurrent);
      try {
        const attempt = (attempts.get(index) ?? 0) + 1;
        attempts.set(index, attempt);
        await new Promise((r) => setTimeout(r, 2)); // let other workers start
        await behavior(index, attempt);
        onProgress(body.size);
        calls.sent.push(index);
        calls.sizes.set(index, body.size);
      } finally {
        concurrent -= 1;
      }
    },
    async complete(id) {
      calls.completed.push(id);
      return { trial_id: "trial_9" };
    },
  };
  return { transport, calls };
}

const instant = async () => {};
const noJitter = () => 0.5; // backoffMs multiplies by 0.85 + 0.3 * 0.5 = 1

describe("chunk arithmetic", () => {
  it("splits a 20 MiB file into 8, 8 and 4 MiB chunks", async () => {
    expect(chunkCount(20 * MiB)).toBe(3);
    const { transport, calls } = transportWith();
    await uploadFile(fakeFile(20 * MiB), {
      transport,
      experimentId: "e1",
      role: "fit",
      store: memoryResumeStore(),
      sleep: instant,
    });
    expect(calls.sizes.get(0)).toBe(8 * MiB);
    expect(calls.sizes.get(1)).toBe(8 * MiB);
    expect(calls.sizes.get(2)).toBe(4 * MiB);
    expect([...calls.sent].sort()).toEqual([0, 1, 2]);
    expect(calls.completed).toEqual(["up_1"]);
  });

  it("an exact multiple has no empty trailing chunk", () => {
    expect(chunkCount(16 * MiB)).toBe(2);
    expect(chunkCount(16 * MiB + 1)).toBe(3);
  });
});

describe("parallelism", () => {
  it("sends three chunks at a time, never more", async () => {
    const { transport, calls } = transportWith();
    await uploadFile(fakeFile(60 * MiB), {
      transport,
      experimentId: "e1",
      role: "fit",
      store: memoryResumeStore(),
      sleep: instant,
    });
    expect(calls.sent).toHaveLength(8);
    expect(calls.maxConcurrent).toBe(3);
  });

  it("reports monotonically growing progress that ends at the file size", async () => {
    const { transport } = transportWith();
    const seen: UploadProgress[] = [];
    await uploadFile(fakeFile(24 * MiB), {
      transport,
      experimentId: "e1",
      role: "fit",
      store: memoryResumeStore(),
      sleep: instant,
      onProgress: (p) => seen.push(p),
    });
    const bytes = seen.map((p) => p.sentBytes);
    for (let i = 1; i < bytes.length; i++) expect(bytes[i]).toBeGreaterThanOrEqual(bytes[i - 1]);
    expect(bytes.at(-1)).toBe(24 * MiB);
    expect(seen.at(-1)!.state).toBe("finishing");
  });
});

describe("retries", () => {
  it("backs off 0.5 s, 1 s, 2 s, 4 s, 8 s", () => {
    expect([1, 2, 3, 4, 5].map((n) => backoffMs(n, noJitter))).toEqual([500, 1000, 2000, 4000, 8000]);
    expect(backoffMs(9, noJitter)).toBe(8000);
  });

  it("retries a flaky chunk with growing delays and then succeeds", async () => {
    const delays: number[] = [];
    const { transport, calls } = transportWith((index, attempt) => {
      if (index === 1 && attempt <= 3) throw new StatusError(503);
    });
    await uploadFile(fakeFile(20 * MiB), {
      transport,
      experimentId: "e1",
      role: "fit",
      store: memoryResumeStore(),
      random: noJitter,
      sleep: async (ms) => void delays.push(ms),
    });
    expect(delays).toEqual([500, 1000, 2000]);
    expect(calls.sent).toContain(1);
  });

  it("gives up after five retries (six attempts) and says which part failed", async () => {
    let attempts = 0;
    const { transport } = transportWith((index) => {
      if (index === 0) {
        attempts += 1;
        throw new StatusError(502);
      }
    });
    const store = memoryResumeStore();
    await expect(
      uploadFile(fakeFile(20 * MiB), {
        transport,
        experimentId: "e1",
        role: "fit",
        store,
        sleep: instant,
        random: noJitter,
      }),
    ).rejects.toMatchObject({ kind: "chunk_failed", chunkIndex: 0 });
    expect(attempts).toBe(6);
    // The upload is still resumable.
    expect(store.load("e1", "fit")).not.toBeNull();
  });

  it("does not retry errors that would fail the same way again", async () => {
    let attempts = 0;
    const { transport } = transportWith(() => {
      attempts += 1;
      throw new StatusError(413);
    });
    await expect(
      uploadFile(fakeFile(9 * MiB), {
        transport,
        experimentId: "e1",
        role: "fit",
        store: memoryResumeStore(),
        sleep: instant,
        parallel: 1,
      }),
    ).rejects.toBeInstanceOf(UploadError);
    expect(attempts).toBe(1);
    expect(isPermanent(new StatusError(413))).toBe(true);
    expect(isPermanent(new StatusError(429))).toBe(false);
    expect(isPermanent(new StatusError(408))).toBe(false);
    expect(isPermanent(new StatusError(500))).toBe(false);
    expect(isPermanent(new Error("network"))).toBe(false);
  });

  it("waits for the network while offline without using up retries", async () => {
    let offline = true;
    let failures = 0;
    const { transport, calls } = transportWith((index) => {
      if (index === 0 && failures < 8) {
        failures += 1;
        throw new Error("network down");
      }
    });
    let waits = 0;
    await uploadFile(fakeFile(9 * MiB), {
      transport,
      experimentId: "e1",
      role: "fit",
      store: memoryResumeStore(),
      parallel: 1,
      sleep: instant,
      isOffline: () => offline,
      waitForOnline: async () => {
        waits += 1;
        // Back online after the 8th wait, but the first online try still fails a few times.
        if (waits >= 8) offline = false;
      },
    });
    expect(waits).toBeGreaterThanOrEqual(8);
    // 8 failures is more than the 5 retries allow, yet the upload finished.
    expect(calls.sent).toContain(0);
  });
});

describe("resume", () => {
  it("only sends the chunks the stored record is missing", async () => {
    const store = memoryResumeStore();
    store.save({
      uploadId: "up_old",
      experimentId: "e1",
      role: "fit",
      filename: "pendulum.mov",
      size: 30 * MiB,
      chunkSize: CHUNK_SIZE,
      received: [0, 1],
    });
    const { transport, calls } = transportWith();
    const out = await uploadFile(fakeFile(30 * MiB), { transport, experimentId: "e1", role: "fit", store, sleep: instant });
    expect(calls.started).toBe(0); // no new session
    expect(calls.sent).toEqual([2, 3]);
    expect(calls.completed).toEqual(["up_old"]);
    expect(out.resumed).toBe(true);
    expect(store.load("e1", "fit")).toBeNull(); // cleared once complete
  });

  it("starts over when the user picks a different file", async () => {
    const store = memoryResumeStore();
    store.save({
      uploadId: "up_old",
      experimentId: "e1",
      role: "fit",
      filename: "other.mov",
      size: 30 * MiB,
      chunkSize: CHUNK_SIZE,
      received: [0, 1],
    });
    const { transport, calls } = transportWith();
    const out = await uploadFile(fakeFile(30 * MiB, "pendulum.mov"), {
      transport,
      experimentId: "e1",
      role: "fit",
      store,
      sleep: instant,
    });
    expect(calls.started).toBe(1);
    expect(calls.sent).toHaveLength(4);
    expect(out.resumed).toBe(false);
  });

  it("an aborted upload keeps its record so it can resume", async () => {
    const store = memoryResumeStore();
    const controller = new AbortController();
    const { transport, calls } = transportWith((index) => {
      if (index === 2) controller.abort();
    });
    await expect(
      uploadFile(fakeFile(60 * MiB), {
        transport,
        experimentId: "e1",
        role: "fit",
        store,
        sleep: instant,
        parallel: 1,
        signal: controller.signal,
      }),
    ).rejects.toMatchObject({ kind: "aborted" });
    const rec = store.load("e1", "fit")!;
    expect(rec.received).toEqual(expect.arrayContaining([0, 1]));
    expect(rec.received).not.toContain(3);
    expect(calls.completed).toEqual([]);
  });
});

describe("limits", () => {
  it("rejects files over 1 GiB and empty files before any request", async () => {
    const { transport, calls } = transportWith();
    const huge = { size: MAX_FILE_BYTES + 1, name: "big.mov", slice: () => new Blob([]) } as unknown as Blob & { name: string };
    await expect(
      uploadFile(huge, { transport, experimentId: "e1", role: "fit", store: memoryResumeStore() }),
    ).rejects.toMatchObject({ kind: "too_large" });
    await expect(
      uploadFile(fakeFile(0), { transport, experimentId: "e1", role: "fit", store: memoryResumeStore() }),
    ).rejects.toMatchObject({ kind: "empty" });
    expect(calls.started).toBe(0);
  });
});
