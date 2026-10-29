"use client";

import { useMemo, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { Button, Card, ErrorState, Notice, SectionTitle, cn } from "@/components/ui";
import { api } from "@/lib/api";
import { fmtClock } from "@/lib/format";
import type { EventActor, EventKind, NotebookEvent } from "@/lib/types";

/**
 * The lab notebook (build-plan Part 20): an append-only timeline colored by actor, with the
 * four moments that make the story pinned on top, a "key decisions" filter, expandable raw
 * details and a Re-run button on every fit.
 */

const ACTORS: Record<EventActor, { label: string; dot: string; text: string }> = {
  wonder: { label: "Muse Wonder", dot: "bg-sky-400", text: "text-sky-300" },
  spark_open: { label: "Muse Spark Open", dot: "bg-violet-400", text: "text-violet-300" },
  harness: { label: "Harness", dot: "bg-emerald-400", text: "text-emerald-300" },
  sam: { label: "SAM 3.1", dot: "bg-orange-400", text: "text-orange-300" },
  system: { label: "System", dot: "bg-zinc-400", text: "text-zinc-300" },
  user: { label: "You", dot: "bg-pink-400", text: "text-pink-300" },
};

const KIND_LABEL: Record<EventKind, string> = {
  observation: "Observation",
  hypothesis: "Hypothesis",
  prediction: "Prediction",
  fit: "Fit",
  rejection: "Rejection",
  acceptance: "Result",
  review: "Review",
  response: "Response",
  revision: "Revision",
  ghost: "Ghost",
  whatif: "What if",
  warning: "Warning",
};

const KEY_KINDS = new Set<EventKind>(["prediction", "rejection", "acceptance", "review", "response", "revision", "ghost", "warning"]);

type Pin = "prediction" | "rejection" | "challenge" | "ghost_verdict";
const PIN_META: Record<Pin, { title: string; blurb: string }> = {
  prediction: { title: "Prediction made before data", blurb: "Wonder's favorite, and what would refute it." },
  rejection: { title: "Hypothesis rejected", blurb: "Cut on a held-out number." },
  challenge: { title: "Reviewer challenge", blurb: "A second model pushes back." },
  ghost_verdict: { title: "Ghost verdict", blurb: "The prediction meets a new clip." },
};

function pinOf(e: NotebookEvent): Pin | null {
  const p = e.data.pinned;
  return p === "prediction" || p === "rejection" || p === "challenge" || p === "ghost_verdict" ? p : null;
}

export function NotebookTimeline({
  events,
  experimentId,
  live,
  onChanged,
}: {
  events: NotebookEvent[];
  experimentId: string;
  live?: boolean;
  onChanged?: () => void;
}) {
  const [filter, setFilter] = useState<"key" | "all">("key");
  const pins = useMemo(() => {
    const out: { pin: Pin; event: NotebookEvent }[] = [];
    for (const pin of ["prediction", "rejection", "challenge", "ghost_verdict"] as Pin[]) {
      const e = events.find((x) => pinOf(x) === pin);
      if (e) out.push({ pin, event: e });
    }
    return out;
  }, [events]);
  const shown = filter === "all" ? events : events.filter((e) => KEY_KINDS.has(e.kind));

  const jump = (seq: number) => {
    setFilter("all");
    requestAnimationFrame(() => {
      document.getElementById(`event-${seq}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  };

  if (events.length === 0) {
    return (
      <Card>
        <p className="text-sm text-muted">
          {live ? "The notebook fills in as the analysis runs." : "No notebook entries yet."}
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-5">
      {pins.length > 0 ? (
        <section aria-label="Key moments">
          <SectionTitle hint="Tap one to jump to it">Key moments</SectionTitle>
          <ul className="grid gap-3 sm:grid-cols-2">
            {pins.map(({ pin, event }) => (
              <li key={pin}>
                <button
                  type="button"
                  onClick={() => jump(event.seq)}
                  className="h-full w-full rounded-xl border border-ghost/25 bg-ghost/5 p-3.5 text-left hover:border-ghost/50"
                >
                  <span className="text-xs font-medium uppercase tracking-wide text-ghost">{PIN_META[pin].title}</span>
                  <span className="mt-1 block text-sm font-medium text-foreground">{event.title}</span>
                  <span className="mt-0.5 block text-xs text-muted">{PIN_META[pin].blurb}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div role="group" aria-label="Notebook filter" className="inline-flex rounded-lg border border-line p-0.5">
          {(["key", "all"] as const).map((f) => (
            <button
              key={f}
              type="button"
              aria-pressed={filter === f}
              onClick={() => setFilter(f)}
              className={cn(
                "min-h-10 rounded-md px-3.5 text-sm font-medium",
                filter === f ? "bg-surface-2 text-foreground" : "text-muted hover:text-foreground",
              )}
            >
              {f === "key" ? "Key decisions" : "Everything"}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted">
          {shown.length} of {events.length} entries{live ? " · updating live" : ""}
        </p>
      </div>

      <ol className="relative space-y-4 border-l border-line pl-5">
        {shown.map((e) => (
          <EventItem key={e.seq} event={e} experimentId={experimentId} onChanged={onChanged} />
        ))}
      </ol>
    </div>
  );
}

function EventItem({
  event: e,
  experimentId,
  onChanged,
}: {
  event: NotebookEvent;
  experimentId: string;
  onChanged?: () => void;
}) {
  const actor = ACTORS[e.actor];
  const [raw, setRaw] = useState(false);
  const [rerun, setRerun] = useState(false);
  const [split, setSplit] = useState(65);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [done, setDone] = useState<string | null>(null);
  const modelId = typeof e.data.model_id === "string" ? e.data.model_id : null;
  const canRerun = e.kind === "fit" && e.data.rerunnable === true && modelId !== null;
  const pin = pinOf(e);

  const submit = async () => {
    if (!modelId) return;
    setBusy(true);
    setError(null);
    try {
      const job = await api.rerunModel(experimentId, modelId, { split_pct: split });
      if (job?.error) throw new Error(job.error);
      setDone(`Re-ran ${modelId} with the split at ${split}%. A new entry was appended to the notebook.`);
      setRerun(false);
      onChanged?.();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <li id={`event-${e.seq}`} className="relative scroll-mt-24">
      <span aria-hidden className={cn("absolute -left-[1.62rem] top-1.5 size-2.5 rounded-full ring-4 ring-background", actor.dot)} />
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
        <span className={cn("font-medium", actor.text)}>{actor.label}</span>
        <span className="text-muted">· {KIND_LABEL[e.kind]}</span>
        <time className="text-muted" dateTime={e.ts}>
          · {fmtClock(e.ts)}
        </time>
        {pin ? <span className="rounded-full bg-ghost/15 px-2 py-0.5 text-[11px] font-medium text-ghost">{PIN_META[pin].title}</span> : null}
      </div>
      <h3 className="mt-0.5 text-sm font-semibold text-foreground">{e.title}</h3>
      {e.body_md ? <Markdown text={e.body_md} className="mt-1.5" /> : null}

      <div className="mt-1 flex flex-wrap gap-x-3">
        {Object.keys(e.data).length > 0 ? (
          <button type="button" aria-expanded={raw} onClick={() => setRaw((v) => !v)} className="min-h-9 text-xs font-medium text-muted hover:text-foreground">
            {raw ? "Hide raw details" : "Raw details"}
          </button>
        ) : null}
        {canRerun ? (
          <button type="button" aria-expanded={rerun} onClick={() => setRerun((v) => !v)} className="min-h-9 text-xs font-medium text-ghost">
            Re-run
          </button>
        ) : null}
      </div>

      {raw ? (
        <pre className="mt-1 max-h-72 overflow-auto rounded-lg border border-line bg-background p-3 font-mono text-xs leading-relaxed text-zinc-300">
          {JSON.stringify(e.data, null, 2)}
        </pre>
      ) : null}

      {rerun ? (
        <div className="mt-2 rounded-lg border border-line bg-surface p-3">
          <label className="block text-sm font-medium" htmlFor={`split-${e.seq}`}>
            Fit on the first {split}% of the motion, forecast the rest
          </label>
          <input
            id={`split-${e.seq}`}
            type="range"
            min={30}
            max={90}
            step={5}
            value={split}
            onChange={(ev) => setSplit(Number(ev.target.value))}
            className="mt-2 h-11 w-full accent-cyan-400"
          />
          <div className="flex gap-2">
            <Button tone="primary" onClick={submit} disabled={busy}>
              {busy ? "Re-running…" : `Re-run ${modelId}`}
            </Button>
            <Button tone="ghost" onClick={() => setRerun(false)}>
              Cancel
            </Button>
          </div>
        </div>
      ) : null}
      {error ? (
        <div className="mt-2">
          <ErrorState error={error} />
        </div>
      ) : null}
      {done ? (
        <div className="mt-2">
          <Notice tone="cyan">{done}</Notice>
        </div>
      ) : null}
    </li>
  );
}
