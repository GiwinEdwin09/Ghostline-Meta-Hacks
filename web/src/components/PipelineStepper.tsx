"use client";

import { Button, Card, Notice, cn } from "@/components/ui";
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
  const current = phaseToStep(job.phase ?? "", phases);
  const failed = job.status === "failed";

  const doneCount = failed ? current : Math.min(current, phases.length);
  return (
    <Card className={cn("relative overflow-hidden", failed ? "border-red-400/30" : "border-ghost/30")}>
      <div aria-hidden className="graph-paper pointer-events-none absolute inset-0 opacity-60 [mask-image:linear-gradient(to_bottom,black,transparent_60%)]" />
      <div className="relative">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <p className="eyebrow">{failed ? "Run stopped" : active ? "Run in progress" : "Run"}</p>
            <p className="mt-1 font-display text-2xl leading-tight">
              {failed ? "This run stopped" : mode === "prove" ? "Predicting your second trial" : "Analyzing your clip"}
            </p>
          </div>
          <p className="font-mono text-sm tabular-nums text-muted">
            {active ? fmtDuration(elapsed) : null}
            <span className="ml-3 text-fg-2">
              {doneCount}/{phases.length}
            </span>
          </p>
        </div>
        {active ? <p className="mt-1 text-sm text-muted">You can leave this page. The analysis keeps running and will be here when you come back.</p> : null}

        <div aria-hidden className="mt-4 flex gap-1">
          {phases.map((p, i) => (
            <span
              key={p}
              className={cn(
                "h-1 flex-1 rounded-full",
                failed && i === current ? "bg-red-400" : i < current ? "bg-foreground/70" : i === current ? "bg-ghost" : "bg-line-strong",
              )}
              style={i === current && !failed ? { animation: "ghost-pulse 1.4s ease-in-out infinite" } : undefined}
            />
          ))}
        </div>

        <ol className="mt-4 grid gap-1 sm:grid-cols-2" aria-label="Pipeline progress">
          {phases.map((p, i) => {
            const state = failed && i === current ? "failed" : i < current ? "done" : i === current ? "current" : "todo";
            return (
              <li
                key={p}
                aria-current={state === "current" ? "step" : undefined}
                className={cn(
                  "flex min-h-11 items-center gap-3 rounded-md px-2.5 py-1.5",
                  state === "current" && "bg-ghost/[0.08]",
                  state === "failed" && "bg-red-400/10",
                )}
              >
                <span
                  aria-hidden
                  className={cn(
                    "flex size-6 shrink-0 items-center justify-center rounded-full border font-mono text-[0.6875rem]",
                    state === "done" && "border-foreground/40 bg-foreground/10 text-foreground",
                    state === "current" && "border-ghost bg-ghost/15 text-ghost",
                    state === "failed" && "border-red-400 bg-red-400/20 text-red-300",
                    state === "todo" && "border-line-strong text-muted",
                  )}
                >
                  {state === "done" ? "✓" : state === "failed" ? "!" : state === "current" ? <span className="size-2 rounded-full bg-ghost" style={{ animation: "ghost-pulse 1.4s ease-in-out infinite" }} /> : i + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className={cn("block text-sm font-medium", state === "todo" && "text-muted", state === "done" && "text-fg-2")}>{LABELS[p]}</span>
                  {state === "current" ? <span className="block text-xs text-muted">{HINTS[p]}</span> : null}
                </span>
              </li>
            );
          })}
        </ol>

        {failed ? (
          <div className="mt-4" role="alert">
            <Notice tone="red" title={job.error ?? "The worker stopped responding."}>
              Nothing was lost: finished steps are cached, so a retry picks up where it stopped.
              {onRetry ? (
                <div className="mt-3">
                  <Button tone="secondary" onClick={onRetry}>
                    Retry
                  </Button>
                </div>
              ) : null}
            </Notice>
          </div>
        ) : null}
      </div>
    </Card>
  );
}
