"use client";

import { useState } from "react";

import { ExportDialog } from "@/components/experiment/ExportDialog";
import { Markdown } from "@/components/Markdown";
import { Tex } from "@/components/Tex";
import { Badge, Button, Card, ConfidenceBadge, SectionTitle, cn } from "@/components/ui";
import { fmtNum, fmtPm } from "@/lib/format";
import type { Experiment, NotebookEvent, Quantity } from "@/lib/types";

const PIN_ORDER = ["prediction", "rejection", "challenge", "ghost_verdict"] as const;
const PIN_TITLE: Record<(typeof PIN_ORDER)[number], string> = {
  prediction: "Prediction made before data",
  rejection: "Hypothesis rejected",
  challenge: "Reviewer challenge",
  ghost_verdict: "Ghost verdict",
};

export function ResultTab({
  exp,
  events,
  goTab,
}: {
  exp: Experiment;
  events: NotebookEvent[];
  goTab: (tab: string) => void;
}) {
  const c = exp.conclusion;
  const [exporting, setExporting] = useState(false);
  if (!c) return null;
  const winner = exp.models.find((m) => m.id === c.winner_id);
  const pins = PIN_ORDER.map((p) => ({ p, e: events.find((e) => e.data.pinned === p) })).filter((x): x is { p: (typeof PIN_ORDER)[number]; e: NotebookEvent } => !!x.e);

  return (
    <div className="space-y-6">
      {c.cant_tell ? (
        <Card className="border-amber-400/40 bg-amber-400/[0.04]">
          <p className="eyebrow text-amber-300">Inconclusive</p>
          <h2 className="mt-1.5 font-display text-3xl leading-tight text-amber-100">Ghostline can&apos;t tell yet</h2>
          <p className="mt-2 text-sm text-fg-2">{c.cant_tell.reason}</p>
          <p className="mt-4 text-sm font-medium">What to change and film again</p>
          <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-fg-2 marker:text-amber-300">
            {c.cant_tell.fixes.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </Card>
      ) : null}

      <Card as="section" className="overflow-hidden p-0 sm:p-0">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3 sm:px-5">
          <p className="eyebrow">{c.cant_tell ? "Best candidate so far" : "The law that won"}</p>
          <div className="flex flex-wrap items-center gap-2">
            <ConfidenceBadge level={c.confidence} />
            {winner ? <Badge className="normal-case">{winner.id}</Badge> : null}
          </div>
        </div>
        <div className="p-4 sm:p-5">
          <h2 className="font-display text-[2rem] leading-tight">{winner ? winner.name : c.winner_id}</h2>
          {winner ? (
            <>
              <p className="mt-1 text-sm leading-relaxed text-fg-2">{winner.mechanism}</p>
              <div className="graph-paper mt-4 overflow-x-auto rounded-md border border-line bg-background/60 px-4 py-2">
                <Tex latex={winner.equations_latex} display />
              </div>
            </>
          ) : null}

          <ul className={cn("mt-5 grid gap-3", c.quantities.length > 1 && "md:grid-cols-2")}>
            {c.quantities.map((q) => (
              <QuantityRow key={q.name} q={q} />
            ))}
          </ul>

          <div className="mt-6 border-t border-line pt-4">
            <p className="text-sm font-medium">Why {c.confidence === "low" ? "not higher" : `${c.confidence} confidence`}</p>
            <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-fg-2 marker:text-muted">
              {c.confidence_reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          </div>
        </div>
      </Card>

      <Card as="section">
        <SectionTitle>In plain words</SectionTitle>
        <Markdown text={c.narrative_md} />
        {c.caveats.length > 0 ? (
          <div className="mt-4 border-t border-line pt-4">
            <p className="text-sm font-medium">Caveats</p>
            <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-fg-2 marker:text-muted">
              {c.caveats.map((cv) => (
                <li key={cv}>{cv}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </Card>

      {pins.length > 0 ? (
        <section aria-label="The story">
          <SectionTitle hint="From the lab notebook">The story</SectionTitle>
          <ol className="relative ml-3.5 space-y-5 border-l border-line pb-1">
            {pins.map(({ p, e }, i) => (
              <li key={p} className="relative pl-7">
                <span
                  aria-hidden
                  className={cn(
                    "absolute -left-3.5 top-0 flex size-7 items-center justify-center rounded-full border bg-background font-mono text-xs",
                    i === pins.length - 1 ? "border-ghost text-ghost" : "border-line-strong text-fg-2",
                  )}
                >
                  {i + 1}
                </span>
                <p className="eyebrow pt-1.5 text-ghost">{PIN_TITLE[p]}</p>
                <p className="mt-1 text-sm font-semibold">{e.title}</p>
                <Markdown text={e.body_md.split("\n\n")[0] ?? ""} className="mt-1 line-clamp-4" />
              </li>
            ))}
          </ol>
          <button type="button" onClick={() => goTab("notebook")} className="group ml-3.5 mt-2 inline-flex min-h-11 items-center gap-1.5 pl-7 text-sm font-medium text-ghost hover:text-cyan-200">
            Read the whole notebook
            <span aria-hidden className="transition-transform group-hover:translate-x-0.5">→</span>
          </button>
        </section>
      ) : null}

      <div className="flex flex-wrap gap-3 border-t border-line pt-6">
        <Button tone="primary" onClick={() => goTab("prove")}>
          Prove it with a ghost
        </Button>
        <Button onClick={() => goTab("whatif")} disabled={!exp.whatif}>
          What if…
        </Button>
        {exp.export_formats.length > 0 ? <Button onClick={() => setExporting(true)}>Export to simulator</Button> : null}
      </div>
      {exp.export_formats.length > 0 ? <ExportDialog exp={exp} open={exporting} onClose={() => setExporting(false)} /> : null}
    </div>
  );
}

function QuantityRow({ q }: { q: Quantity }) {
  const [open, setOpen] = useState(false);
  const half = (q.ci_high - q.ci_low) / 2;
  const total = Math.sqrt((q.budget ?? []).reduce((s, b) => s + b.sigma ** 2, 0));
  return (
    <li className="rounded-md border border-line bg-background/60 p-4">
      <p className="eyebrow">{q.name}</p>
      <p className="mt-2.5 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <Tex latex={`${q.symbol_latex} =`} className="text-lg text-muted" />
        <span className="font-mono text-[1.75rem] font-medium leading-none tabular-nums tracking-tight">{fmtPm(q.value, half)}</span>
        <span className="font-mono text-sm text-muted">{q.unit}</span>
      </p>
      <p className="mt-2.5 font-mono text-xs text-muted">
        95% interval {fmtNum(q.ci_low, 4)} to {fmtNum(q.ci_high, 4)} {q.unit}
      </p>
      {q.note ? <p className="mt-1.5 text-xs leading-relaxed text-muted">{q.note}</p> : null}
      {q.budget ? (
        <>
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
            className="-mb-2 mt-1 inline-flex min-h-11 items-center gap-1 text-sm font-medium text-ghost hover:text-cyan-200"
          >
            Where the uncertainty comes from
            <svg viewBox="0 0 24 24" className={cn("size-4 transition-transform", open && "rotate-180")} fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="m6 9 6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          {open ? (
            <table className="mb-1 mt-3 w-full text-sm">
              <caption className="sr-only">Uncertainty budget for {q.name}</caption>
              <tbody className="font-mono text-[0.8125rem] tabular-nums">
                {q.budget.map((b) => (
                  <tr key={b.label} className="border-t border-line">
                    <th scope="row" className="py-1.5 pr-3 text-left font-sans text-sm font-normal text-fg-2">
                      {b.label}
                    </th>
                    <td className="py-1.5 text-right text-muted">{b.sigma > 0 ? `± ${fmtNum(b.sigma, 2)}` : "none"}</td>
                  </tr>
                ))}
                <tr className="border-t border-line-strong font-medium">
                  <th scope="row" className="py-1.5 pr-3 text-left font-sans text-sm">
                    Combined (in quadrature)
                  </th>
                  <td className="py-1.5 text-right">
                    ± {fmtNum(total, 2)} {q.unit}
                  </td>
                </tr>
              </tbody>
            </table>
          ) : null}
        </>
      ) : null}
    </li>
  );
}
