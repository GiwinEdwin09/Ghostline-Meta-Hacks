"use client";

import { useState } from "react";

import { CHART_COLORS, LineChart } from "@/components/charts/LineChart";
import { Tex } from "@/components/Tex";
import { Badge, Card, cn, type BadgeTone } from "@/components/ui";
import { resolveMedia } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import type { ModelRecord, ModelStatus } from "@/lib/types";

const STATUS: Record<ModelStatus, { label: string; tone: BadgeTone }> = {
  winner: { label: "Winner", tone: "green" },
  tied: { label: "Tied: can't tell apart", tone: "amber" },
  rejected: { label: "Rejected", tone: "red" },
  proposed: { label: "Proposed", tone: "neutral" },
  invalid: { label: "Invalid code", tone: "red" },
};

/** Held-out RMSE as a bar with confidence-interval whiskers. Bars past `max` are cut and marked. */
export function RmseBar({
  value,
  ci,
  max,
  unit,
  tone,
}: {
  value: number;
  ci: [number, number];
  max: number;
  unit: string;
  tone: "good" | "tie" | "bad";
}) {
  const pct = (v: number) => Math.min(100, Math.max(0, (v / max) * 100));
  const off = value > max;
  const color = tone === "good" ? "bg-emerald-400" : tone === "tie" ? "bg-amber-400" : "bg-red-400";
  return (
    <div>
      <div
        className="relative h-4 rounded-sm bg-surface-2"
        role="img"
        aria-label={`Held-out error ${fmtNum(value)} ${unit}, interval ${fmtNum(ci[0])} to ${fmtNum(ci[1])}${off ? ", off the scale" : ""}`}
      >
        <div className={cn("absolute inset-y-0 left-0 rounded-sm", color)} style={{ width: `${Math.max(1.5, pct(value))}%`, opacity: 0.8 }} />
        {!off ? (
          <>
            <div className="absolute top-1/2 h-px bg-foreground" style={{ left: `${pct(ci[0])}%`, width: `${Math.max(0, pct(ci[1]) - pct(ci[0]))}%` }} />
            <div className="absolute inset-y-0.5 w-px bg-foreground" style={{ left: `${pct(ci[0])}%` }} />
            <div className="absolute inset-y-0.5 w-px bg-foreground" style={{ left: `${pct(ci[1])}%` }} />
          </>
        ) : (
          <span aria-hidden className="absolute right-1 top-1/2 -translate-y-1/2 text-xs font-bold text-background">
            ▸▸
          </span>
        )}
      </div>
      <p className="mt-1.5 font-mono text-xs tabular-nums text-muted">
        {fmtNum(value)} {unit}
        <span className="opacity-80">
          {" "}
          ({fmtNum(ci[0])} to {fmtNum(ci[1])})
        </span>
        {off ? " · off the scale" : ""}
      </p>
    </div>
  );
}

export function ModelCard({
  model,
  barMax,
  defaultOpen = false,
}: {
  model: ModelRecord;
  barMax: number;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const fit = model.fit;
  const st = STATUS[model.status];
  const tone = model.status === "winner" ? "good" : model.status === "tied" ? "tie" : "bad";
  return (
    <Card as="article" className={cn(model.status === "winner" && "border-emerald-400/40", model.status === "rejected" && "bg-surface/60")}>
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-xs text-muted">{model.id}</span>
        <h3 className={cn("font-display text-[1.6rem] leading-tight", model.status === "rejected" && "text-fg-2")}>{model.name}</h3>
      </div>
      <div className="mt-2.5 flex flex-wrap gap-2">
        <Badge tone={st.tone}>{st.label}</Badge>
        {model.prior_rank === 1 ? <Badge tone="violet">Wonder&apos;s first guess</Badge> : null}
        {model.source === "wonder" ? <Badge>Written by Wonder</Badge> : null}
        {fit ? <Badge>Lowest on {fit.splits_won} of 3 splits</Badge> : null}
      </div>
      <p className="mt-3 text-sm leading-relaxed text-fg-2">{model.mechanism}</p>
      <div className="mt-3 overflow-x-auto rounded-md border border-line bg-background/60 px-3 py-1">
        <Tex latex={model.equations_latex} display />
      </div>

      {fit ? (
        <div className="mt-4">
          <p className="eyebrow mb-2">Held-out error · forecast of the last 35%</p>
          <RmseBar value={fit.heldout_rmse} ci={fit.heldout_ci} max={barMax} unit={fit.rmse_unit} tone={tone} />
        </div>
      ) : null}

      {model.rejection_reason ? (
        <p className="mt-4 rounded-r-md border-l-2 border-red-400 bg-red-400/[0.06] px-3.5 py-3 text-sm text-fg-2">
          <span className="font-medium text-red-200">Why it was rejected. </span>
          {model.rejection_reason}
        </p>
      ) : null}

      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="-mb-2 mt-2 inline-flex min-h-11 items-center gap-1.5 text-sm font-medium text-ghost hover:text-cyan-200"
      >
        {open ? "Hide details" : "Parameters, plots and diagnostics"}
        <svg viewBox="0 0 24 24" className={cn("size-4 transition-transform", open && "rotate-180")} fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="m6 9 6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && fit ? (
        <div className="mt-4 space-y-5 border-t border-line pt-4">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[22rem] text-sm">
              <thead>
                <tr className="text-left">
                  <th className="eyebrow pb-2 font-medium">Parameter</th>
                  <th className="eyebrow pb-2 font-medium">Value</th>
                  <th className="eyebrow pb-2 font-medium">95% interval</th>
                </tr>
              </thead>
              <tbody className="font-mono text-[0.8125rem] tabular-nums">
                {fit.params.map((p) => (
                  <tr key={p.name} className="border-t border-line">
                    <td className="py-1.5 pr-3 font-sans text-sm">
                      {p.label}
                      {p.at_bound ? <span className="ml-1.5 text-xs text-amber-300">at bound</span> : null}
                    </td>
                    <td className="py-1.5 pr-3">
                      {fmtNum(p.value, 4)} <span className="text-muted">{p.unit}</span>
                    </td>
                    <td className="py-1.5 text-muted">
                      {fmtNum(p.ci_low, 3)} to {fmtNum(p.ci_high, 3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
            <Metric label="Training error" value={`${fmtNum(fit.train_rmse)} ${fit.rmse_unit}`} />
            <Metric label="Error / motion range" value={`${(fit.nrmse * 100).toFixed(fit.nrmse < 0.01 ? 2 : 1)}%`} />
            <Metric
              label="Diverges after"
              value={fit.time_to_divergence_s === null ? "never" : `${fmtNum(fit.time_to_divergence_s, 2)} s`}
            />
            <Metric label="AIC / BIC" value={`${fmtNum(fit.aic, 4)} / ${fmtNum(fit.bic, 4)}`} />
            <Metric label="Lag-1 autocorrelation" value={fmtNum(fit.diagnostics.lag1_autocorr, 2)} />
            <Metric label="Durbin–Watson" value={fmtNum(fit.diagnostics.durbin_watson, 2)} />
            <Metric label="Runs test p" value={fmtNum(fit.diagnostics.runs_p, 2)} />
            <Metric label="Reduced χ²" value={fmtNum(fit.diagnostics.reduced_chi2, 3)} />
            <Metric
              label="Residual pattern"
              value={fit.diagnostics.systematic_residual ? "systematic" : "none found"}
              warn={fit.diagnostics.systematic_residual}
            />
          </dl>

          {fit.plot_urls.fit ? (
            <div className="space-y-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={resolveMedia(fit.plot_urls.fit)} alt={`${model.name}: data and fit`} className="w-full rounded-md border border-line" loading="lazy" />
              {fit.plot_urls.residuals ? (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img src={resolveMedia(fit.plot_urls.residuals)} alt={`${model.name}: residuals`} className="w-full rounded-md border border-line" loading="lazy" />
              ) : null}
            </div>
          ) : fit.series ? (
            <FitCharts model={model} />
          ) : null}

          <div>
            <p className="eyebrow">What would refute it</p>
            <p className="mt-1 text-sm text-fg-2">{model.refuting_signature}</p>
          </div>
        </div>
      ) : null}
    </Card>
  );
}

function Metric({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className={cn("mt-0.5 font-mono text-[0.8125rem] tabular-nums", warn && "text-amber-300")}>{value}</dd>
    </div>
  );
}

function FitCharts({ model }: { model: ModelRecord }) {
  const s = model.fit!.series!;
  const resid = s.observed.map((v, i) => v - s.predicted[i]);
  const end = s.t[s.t.length - 1];
  const region = { x0: s.split_t, x1: end, color: "rgba(167,139,250,0.12)", label: "held out" };
  return (
    <div className="space-y-3">
      <LineChart
        ariaLabel={`${model.name}: measured ${s.label} and the fitted curve, with the held-out window shaded`}
        series={[
          { id: "obs", label: "Measured", color: CHART_COLORS.data, x: s.t, y: s.observed, kind: "points", opacity: 0.55 },
          { id: "fit", label: model.id, color: CHART_COLORS.ghost, x: s.t, y: s.predicted },
        ]}
        regions={[region]}
        xLabel="time (s)"
        yLabel={`${s.label} (${s.unit})`}
        height={200}
      />
      <LineChart
        ariaLabel={`${model.name}: residuals over time`}
        series={[{ id: "res", label: "Measured − fit", color: CHART_COLORS.amber, x: s.t, y: resid, width: 1.2 }]}
        regions={[region]}
        zeroLine
        xLabel="time (s)"
        yLabel={`residual (${s.unit})`}
        height={150}
      />
    </div>
  );
}
