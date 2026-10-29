"use client";

import { useState } from "react";

import { GhostPlayer } from "@/components/GhostPlayer";
import { PipelineStepper } from "@/components/PipelineStepper";
import { Button, Card, ErrorState, Notice, SectionTitle, Skeleton } from "@/components/ui";
import { UploadPanel } from "@/components/wizard/UploadPanel";
import { api } from "@/lib/api";
import { isJobActive } from "@/lib/hooks";
import type { Experiment, Job } from "@/lib/types";

/**
 * Prove it (build-plan Part 16): upload a second trial, let Ghostline predict it from the
 * frozen law, and watch the ghost. After a drift, "Why did it drift?" asks Wonder for a
 * diagnosis and a revised model (Part 17).
 */
export function ProveTab({ exp, job, onChanged }: { exp: Experiment; job: Job | null; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const proveTrial = exp.trials.find((t) => t.role === "prove");
  const first = exp.trials.find((t) => t.role === "fit") ?? exp.trials[0];
  const proving = job?.kind === "prove_trial" && isJobActive(job);
  const revising = job?.kind === "revise" && isJobActive(job);

  const call = async (fn: () => Promise<Job | null>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  if (exp.ghosts.length > 0) {
    return (
      <div className="space-y-4">
        <p className="text-sm text-zinc-300">
          A second trial, with the law frozen from trial 1. Only the starting position and speed were fitted, to the launch window; everything after it is a pure prediction.
        </p>
        <GhostPlayer
          ghosts={exp.ghosts}
          aspect={(first?.width ?? 16) / (first?.height ?? 9)}
          onRevise={() => call(() => api.revise(exp.id))}
          revising={busy || revising}
          reviseError={error}
        />
      </div>
    );
  }

  if (proving && job) return <PipelineStepper job={job} mode="prove" />;

  const ready = proveTrial && (proveTrial.status === "ready" || proveTrial.status === "tracked");
  return (
    <div className="space-y-5">
      <Card>
        <SectionTitle>Make a prediction, then check it</SectionTitle>
        <p className="text-sm text-zinc-300">
          Film a second trial with the same setup and any starting conditions: a different swing height, a harder flick. Ghostline freezes the law it found, fits only where the object
          starts, and draws a ghost where the law says it will go. Then it measures how far reality drifts from the ghost.
        </p>
      </Card>
      {proveTrial && !ready ? (
        <div aria-live="polite" className="space-y-3">
          <p className="text-sm text-muted">Preparing your second clip…</p>
          <Skeleton className="aspect-video w-full" />
        </div>
      ) : ready ? (
        <Card className="space-y-3">
          <p className="text-sm">Your second clip is ready.</p>
          <Button tone="primary" disabled={busy} onClick={() => call(() => api.prove(exp.id, proveTrial.id))}>
            {busy ? "Starting…" : "Predict it"}
          </Button>
        </Card>
      ) : (
        <UploadPanel
          role="prove"
          existingExperimentId={exp.id}
          ensureExperiment={async () => exp.id}
          onUploaded={() => onChanged()}
        />
      )}
      {error ? <ErrorState error={error} /> : null}
      {exp.is_example ? <Notice tone="cyan">This is an example, so it may not accept a new trial. Open the pendulum or the steel-ball example to see a finished ghost.</Notice> : null}
    </div>
  );
}
