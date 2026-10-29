"use client";

import { Button, Card, cn } from "@/components/ui";
import { fmtDuration } from "@/lib/format";
import { useElapsedSeconds } from "@/lib/hooks";
import { PIPELINE_PHASES, PROVE_PHASES, type Job } from "@/lib/types";
import { phaseToStep } from "@/lib/phases";

const LABELS: Record<string, string> = {
  normalize: "Normalizing",
  watch: "Watching",
  track: "Tracking",
  calibrate: "Calibrating",
  hypothesize: "Hypothesizing",
  fit: "Fitting",
  review: "Peer review",
  predict: "Predicting",
  render: "Rendering the ghost",
};

const HINTS: Record<string, string> = {
  normalize: "Fixing orientation, color and timestamps",
  watch: "Wonder decides what to measure",
  track: "SAM 3.1 follows the object frame by frame",
  calibrate: "The A4 sheet turns pixels into meters",
  hypothesize: "Competing explanations, named before any fit",
  fit: "Scoring each on frames it never saw",
  review: "A second model challenges the claim",
  predict: "Freezing the law and forecasting trial 2",
  render: "Drawing the ghost over your video",
};

export function PipelineStepper({
  job,
  mode = "analyze",
  onRetry,
}: {
  job: Job;
  mode?: "analyze" | "prove" | "ingest";
  onRetry?: () => void;
}) {
  const phases: readonly string[] = mode === "prove" ? PROVE_PHASES : PIPELINE_PHASES;
  const active = job.status === "queued" || job.status === "running";
  const elapsed = useElapsedSeconds(job.started_at, active);
  const current = phaseToStep(job.phase, phases);
  const failed = job.status === "failed";

  return (
    <Card className="border-ghost/25">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="font-medium">
          {failed ? "This run stopped" : mode === "prove" ? "Predicting your second trial" : "Analyzing your clip"}
        </p>
        <p className="text-sm tabular-nums text-muted">{active ? fmtDuration(elapsed) : null}</p>
      </div>
      {active ? <p className="mt-1 text-sm text-muted">You can leave this page. The analysis keeps running and will be here when you come back.</p> : null}

      <ol className="mt-4 grid gap-1.5" aria-label="Pipeline progress">
        {phases.map((p, i) => {
          const state = failed && i === current ? "failed" : i < current ? "done" : i === current ? "current" : "todo";
          return (
            <li
              key={p}
              aria-current={state === "current" ? "step" : undefined}
              className={cn(
                "flex min-h-11 items-center gap-3 rounded-lg px-3 py-1.5",
                state === "current" && "bg-ghost/10",
                state === "failed" && "bg-red-500/10",
              )}
            >
              <span
                aria-hidden
                className={cn(
                  "flex size-6 shrink-0 items-center justify-center rounded-full border text-xs",
                  state === "done" && "border-emerald-400/60 bg-emerald-400/15 text-emerald-300",
                  state === "current" && "border-ghost bg-ghost/20 text-ghost",
                  state === "failed" && "border-red-400 bg-red-400/20 text-red-300",
                  state === "todo" && "border-line-strong text-muted",
                )}
              >
                {state === "done" ? "✓" : state === "failed" ? "!" : state === "current" ? <span className="size-2 rounded-full bg-ghost" style={{ animation: "ghost-pulse 1.4s ease-in-out infinite" }} /> : i + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span className={cn("block text-sm font-medium", state === "todo" && "text-muted")}>{LABELS[p]}</span>
                {state === "current" ? <span className="block text-xs text-muted">{HINTS[p]}</span> : null}
              </span>
            </li>
          );
        })}
      </ol>

      {failed ? (
        <div className="mt-4 rounded-lg border border-red-400/30 bg-red-400/5 p-3 text-sm" role="alert">
          <p className="font-medium text-red-200">{job.error ?? "The worker stopped responding."}</p>
          <p className="mt-1 text-zinc-300">Nothing was lost: finished steps are cached, so a retry picks up where it stopped.</p>
          {onRetry ? (
            <Button tone="secondary" className="mt-3" onClick={onRetry}>
              Retry
            </Button>
          ) : null}
        </div>
      ) : null}
    </Card>
  );
}
