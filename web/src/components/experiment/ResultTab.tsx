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
    <div className="space-y-5">
      {c.cant_tell ? (
        <Card className="border-amber-400/40 bg-amber-400/5">
          <h2 className="text-lg font-semibold text-amber-200">Ghostline can&apos;t tell yet</h2>
          <p className="mt-1 text-sm text-zinc-200">{c.cant_tell.reason}</p>
          <p className="mt-3 text-sm font-medium">What to change and film again</p>
          <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-zinc-300">
            {c.cant_tell.fixes.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </Card>
      ) : null}

      <Card as="section">
        <div className="flex flex-wrap items-center gap-2">
          <ConfidenceBadge level={c.confidence} />
          {winner ? <Badge>{winner.id}</Badge> : null}
        </div>
        <h2 className="mt-3 text-xl font-semibold tracking-tight">{winner ? winner.name : c.winner_id}</h2>
        {winner ? (
          <>
            <p className="mt-1 text-sm text-zinc-300">{winner.mechanism}</p>
            <div className="mt-2 overflow-x-auto">
              <Tex latex={winner.equations_latex} display />
            </div>
          </>
        ) : null}

        <ul className="mt-4 space-y-3">
          {c.quantities.map((q) => (
            <QuantityRow key={q.name} q={q} />
          ))}
        </ul>

        <div className="mt-5 border-t border-line pt-4">
          <p className="text-sm font-medium">Why {c.confidence === "low" ? "not higher" : `${c.confidence} confidence`}</p>
          <ul className="mt-1.5 list-disc space-y-1 pl-5 text-sm text-zinc-300">
            {c.confidence_reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      </Card>

      <Card as="section">
        <SectionTitle>In plain words</SectionTitle>
        <Markdown text={c.narrative_md} />
        {c.caveats.length > 0 ? (
          <div className="mt-4">
            <p className="text-sm font-medium">Caveats</p>
            <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-zinc-300">
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
          <ol className="space-y-3">
            {pins.map(({ p, e }, i) => (
              <li key={p} className="flex gap-3 rounded-xl border border-line bg-surface p-3.5">
                <span aria-hidden className="flex size-7 shrink-0 items-center justify-center rounded-full bg-ghost/15 text-xs font-semibold text-ghost">
                  {i + 1}
                </span>
                <div className="min-w-0">
                  <p className="text-xs font-medium uppercase tracking-wide text-ghost">{PIN_TITLE[p]}</p>
                  <p className="text-sm font-medium">{e.title}</p>
                  <Markdown text={e.body_md.split("\n\n")[0] ?? ""} className="mt-1 line-clamp-4" />
                </div>
              </li>
            ))}
          </ol>
          <button type="button" onClick={() => goTab("notebook")} className="mt-2 min-h-11 text-sm font-medium text-ghost">
            Read the whole notebook
          </button>
        </section>
      ) : null}

      <div className="flex flex-wrap gap-3">
        <Button tone="primary" onClick={() => goTab("prove")}>
          Prove it
        </Button>
        <Button onClick={() => goTab("whatif")} disabled={!exp.whatif}>
          What if
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
    <li className="rounded-xl border border-line bg-background p-3.5">
      <p className="text-sm text-muted">{q.name}</p>
      <p className="mt-0.5 flex flex-wrap items-baseline gap-x-2 text-2xl font-semibold tabular-nums">
        <Tex latex={`${q.symbol_latex} =`} className="text-lg text-muted" />
        <span>{fmtPm(q.value, half)}</span>
        <span className="text-base font-normal text-muted">{q.unit}</span>
      </p>
      <p className="text-xs text-muted">95% interval: {fmtNum(q.ci_low, 4)} to {fmtNum(q.ci_high, 4)} {q.unit}</p>
      {q.note ? <p className="mt-1 text-xs text-zinc-400">{q.note}</p> : null}
      {q.budget ? (
        <>
          <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)} className="mt-1 inline-flex min-h-11 items-center gap-1 text-sm font-medium text-ghost">
            Where the uncertainty comes from
            <svg viewBox="0 0 24 24" className={cn("size-4 transition-transform", open && "rotate-180")} fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="m6 9 6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          {open ? (
            <table className="mt-1 w-full text-sm">
              <caption className="sr-only">Uncertainty budget for {q.name}</caption>
              <tbody className="tabular-nums">
                {q.budget.map((b) => (
                  <tr key={b.label} className="border-t border-line">
                    <th scope="row" className="py-1.5 pr-3 text-left font-normal text-zinc-300">
                      {b.label}
                    </th>
                    <td className="py-1.5 text-right text-muted">
                      {b.sigma > 0 ? `± ${fmtNum(b.sigma, 2)}` : "none"}
                    </td>
                  </tr>
                ))}
                <tr className="border-t border-line-strong font-medium">
                  <th scope="row" className="py-1.5 pr-3 text-left">Combined (in quadrature)</th>
                  <td className="py-1.5 text-right">± {fmtNum(total, 2)} {q.unit}</td>
                </tr>
              </tbody>
            </table>
          ) : null}
        </>
      ) : null}
    </li>
  );
}
