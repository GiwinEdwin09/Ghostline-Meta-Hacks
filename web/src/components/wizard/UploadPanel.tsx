"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { Button, ErrorState, Notice, buttonClass, cn } from "@/components/ui";
import { api } from "@/lib/api";
import { describeError } from "@/lib/errors";
import { fmtBytes, fmtDuration } from "@/lib/format";
import {
  MAX_FILE_BYTES,
  UploadError,
  pendingUpload,
  uploadFile,
  type UploadProgress,
} from "@/lib/upload/chunked";
import type { TrialRole } from "@/lib/types";

/**
 * File picker plus chunked, resumable upload (build-plan Part 4). The input has no `capture`
 * attribute, so on a phone it offers the camera roll as well as the camera, and a slow-mo
 * clip can be picked from Photos.
 */
export function UploadPanel({
  role,
  ensureExperiment,
  existingExperimentId,
  onUploaded,
}: {
  role: TrialRole;
  /** Creates the experiment on first use and returns its id. */
  ensureExperiment: () => Promise<string>;
  existingExperimentId: string | null;
  onUploaded: (experimentId: string, trialId: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [phase, setPhase] = useState<"idle" | "uploading" | "stopped" | "done">("idle");
  const [speed, setSpeed] = useState<number | null>(null);
  const [dragging, setDragging] = useState(false);
  const [notVideo, setNotVideo] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const samples = useRef<{ t: number; bytes: number }[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const pending = useMemo(
    () => (existingExperimentId ? pendingUpload(existingExperimentId, role) : null),
    [existingExperimentId, role],
  );

  useEffect(() => () => controller.current?.abort(), []);

  // A file dropped next to the drop zone would otherwise make the browser open it and leave the page.
  useEffect(() => {
    const block = (e: DragEvent) => {
      if (e.dataTransfer?.types.includes("Files")) e.preventDefault();
    };
    window.addEventListener("dragover", block);
    window.addEventListener("drop", block);
    return () => {
      window.removeEventListener("dragover", block);
      window.removeEventListener("drop", block);
    };
  }, []);

  const uploading = phase === "uploading";
  const pick = (f: File | null) => {
    if (f && f.type && !f.type.startsWith("video/")) {
      setNotVideo(f.name);
      return;
    }
    setNotVideo(null);
    setFile(f);
    setProgress(null);
    setError(null);
    setPhase("idle");
  };

  const tooBig = file !== null && file.size > MAX_FILE_BYTES;
  // 8 Mbit/s is a conservative cellular upload speed for the estimate before we have a real one.
  const estimate = file ? file.size / (1_000_000 / 8) : null;
  const matchesPending = pending !== null && file !== null && pending.filename === file.name && pending.size === file.size;

  const start = async () => {
    if (!file) return;
    setError(null);
    setPhase("uploading");
    samples.current = [];
    const ctrl = new AbortController();
    controller.current = ctrl;
    try {
      const experimentId = await ensureExperiment();
      const out = await uploadFile(file, {
        transport: api.uploads,
        experimentId,
        role,
        signal: ctrl.signal,
        onProgress: (p) => {
          setProgress(p);
          const now = performance.now();
          const s = samples.current;
          s.push({ t: now, bytes: p.sentBytes });
          while (s.length > 2 && now - s[0].t > 6000) s.shift();
          if (s.length > 1 && now - s[0].t > 800) {
            setSpeed(Math.max(0, (p.sentBytes - s[0].bytes) / ((now - s[0].t) / 1000)));
          }
        },
      });
      setPhase("done");
      onUploaded(experimentId, out.trial_id);
    } catch (err) {
      if (err instanceof UploadError && err.kind === "aborted") {
        setPhase("stopped");
        return;
      }
      setError(err);
      setPhase("stopped");
    }
  };

  const pct = progress ? Math.round((progress.sentBytes / progress.totalBytes) * 100) : 0;
  const remaining = progress && speed && speed > 0 ? (progress.totalBytes - progress.sentBytes) / speed : null;

  return (
    <div className="space-y-4">
      {pending && !file ? (
        <Notice tone="cyan" title="You have an unfinished upload">
          {pending.filename} ({fmtBytes(pending.size)}), {pending.received.length} part
          {pending.received.length === 1 ? "" : "s"} already sent. Choose the same file again and it will continue from where it stopped.
        </Notice>
      ) : null}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!uploading) setDragging(true);
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!uploading) pick(e.dataTransfer.files?.[0] ?? null);
        }}
        className={cn(
          "relative overflow-hidden rounded-md border border-dashed px-5 py-8 text-center transition-colors",
          dragging ? "border-ghost bg-ghost/[0.06]" : "border-line-strong bg-surface/60",
        )}
      >
        <div aria-hidden className="graph-paper pointer-events-none absolute inset-0 opacity-60 [mask-image:radial-gradient(ellipse_at_center,black,transparent_70%)]" />
        <div className="relative">
          <svg viewBox="0 0 24 24" className="mx-auto mb-3 size-8 text-muted" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <rect x="3" y="6" width="13" height="12" rx="1.5" />
            <path d="m16 10 5-3v10l-5-3" />
          </svg>
          <input
            ref={inputRef}
            id={`file-${role}`}
            type="file"
            accept="video/*"
            className="peer sr-only"
            disabled={uploading}
            onChange={(e) => pick(e.target.files?.[0] ?? null)}
          />
          <label
            htmlFor={`file-${role}`}
            className={cn(
              buttonClass("secondary", "min-h-12 px-5"),
              "cursor-pointer peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-ghost peer-disabled:cursor-not-allowed peer-disabled:opacity-45",
            )}
          >
            {file ? "Choose a different video" : "Choose a video"}
          </label>
          <p className="mt-3 text-xs text-muted">
            From your camera roll or your camera<span className="hidden sm:inline">, or drop a file here</span>. Up to 1 GiB.
          </p>
          {file ? (
            <div className="mx-auto mt-5 flex max-w-md items-center gap-3 rounded-md border border-line bg-background/80 px-3.5 py-2.5 text-left text-sm">
              <span aria-hidden className="size-2 shrink-0 rounded-full bg-ghost" />
              <div className="min-w-0">
                <p className="break-all font-medium">{file.name}</p>
                <p className="font-mono text-xs text-muted">
                  {fmtBytes(file.size)}
                  {estimate && phase === "idle" ? ` · about ${fmtDuration(estimate)} on a typical cellular upload` : ""}
                </p>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      {notVideo ? (
        <Notice tone="red" title="That file isn't a video">
          {notVideo} can&apos;t be analyzed. Choose a video clip instead.
        </Notice>
      ) : null}

      {tooBig ? (
        <Notice tone="red" title="That clip is over 1 GiB">
          Trim it in Photos, or film a shorter one. Ghostline needs only the part where the object moves.
        </Notice>
      ) : null}

      {file && phase === "idle" ? (
        <p className="text-xs text-muted">Use Wi-Fi if you can. If the connection drops, the upload pauses and picks up where it left off.</p>
      ) : null}

      {progress && phase !== "idle" ? (
        <div aria-live="polite">
          <div
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
            aria-label="Upload progress"
            className="h-2 overflow-hidden rounded-full bg-surface-2"
          >
            <div className={cn("h-full rounded-full transition-[width] duration-200", phase === "stopped" ? "bg-amber-400" : "bg-ghost")} style={{ width: `${pct}%` }} />
          </div>
          <p className="mt-2 font-mono text-xs tabular-nums text-fg-2">
            {pct}% · {fmtBytes(progress.sentBytes)} of {fmtBytes(progress.totalBytes)}
            {phase === "uploading" && speed ? ` · ${fmtBytes(speed)}/s` : ""}
            {phase === "uploading" && remaining ? ` · about ${fmtDuration(remaining)} left` : ""}
          </p>
          {phase === "uploading" && progress.state === "retrying" ? (
            <p className="mt-1 text-sm text-amber-300">A part didn&apos;t go through. Trying again (attempt {progress.attempt}).</p>
          ) : null}
          {phase === "uploading" && progress.state === "offline" ? (
            <p className="mt-1 text-sm text-amber-300">You&apos;re offline. Waiting for the connection; your progress is saved.</p>
          ) : null}
          {phase === "uploading" && progress.state === "finishing" ? <p className="mt-1 text-sm text-muted">Putting the parts together…</p> : null}
          {progress.resumed && phase === "uploading" ? <p className="mt-1 text-xs text-muted">Resumed from where it stopped.</p> : null}
        </div>
      ) : null}

      {error ? (
        <ErrorState
          error={error}
          title={error instanceof UploadError ? error.message : describeError(error).title}
          onRetry={file ? start : undefined}
        />
      ) : null}

      <div className="flex flex-wrap gap-2">
        {phase === "uploading" ? (
          <Button tone="ghost" onClick={() => controller.current?.abort()}>
            Pause
          </Button>
        ) : (
          <Button tone="primary" disabled={!file || tooBig || phase === "done"} onClick={start}>
            {phase === "stopped" || matchesPending ? "Resume upload" : "Upload"}
          </Button>
        )}
      </div>
    </div>
  );
}
