"use client";

import Link from "next/link";

import { useElementWidth, useResource } from "@/lib/hooks";
import { Badge, Card, ConfidenceBadge, ErrorState, Notice, SectionTitle, Skeleton } from "@/components/ui";
import { CHART_COLORS, niceTicks } from "@/components/charts/LineChart";
import { USE_MOCKS, api } from "@/lib/api";
import { SYSTEM_LABELS, fmtNum, fmtPm } from "@/lib/format";
import type { BenchmarkRow } from "@/lib/types";

/**
 * Benchmark results (build-plan Parts 24 and 25): the table, a plot of every measurement
 * against its ground truth with error bars, the coverage number and the model-selection
 * accuracy, and the deliberately bad clips.
 */
export function BenchmarkView() {
  const bench = useResource("benchmark", () => api.getBenchmark());
  if (bench.error && !bench.data) return <ErrorState error={bench.error} onRetry={bench.reload} />;
  if (!bench.data) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-72 w-full" />
      </div>
    );
  }
  const { rows, summary } = bench.data;
  const scored = rows.filter((r) => !r.is_failure_case);
  const failures = rows.filter((r) => r.is_failure_case);
  return (
    <div className="space-y-8">
      {USE_MOCKS ? (
        <Notice tone="amber" title="Demo data">
          These numbers are placeholders so the page can be judged without a back end. The real table comes from <code>bench/results.md</code>.
        </Notice>
      ) : null}

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Tile label="Clips" value={String(summary.n_clips)} />
        <Tile label="95% intervals that contain the truth" value={`${Math.round(summary.coverage * 100)}%`} hint="Target: 90 to 98%" />
        <Tile label="Right physics chosen" value={`${Math.round(summary.model_selection_accuracy * 100)}%`} />
        <Tile label="Bad clips flagged" value={summary.failure_cases_flagged} />
      </dl>

      <section>
        <SectionTitle hint="Each dot is a clip; bars are 95% intervals">Measured against ground truth</SectionTitle>
        <Card>
          <ErrorPlot rows={scored} />
          <p className="mt-2 text-xs text-muted">Relative error is (measured − truth) / truth. A good measurement sits near zero with a bar that crosses it.</p>
        </Card>
      </section>

      <section>
        <SectionTitle>All clips</SectionTitle>
        <div className="overflow-x-auto rounded-xl border border-line">
          <table className="w-full min-w-[46rem] text-sm">
            <thead className="bg-surface text-left text-xs text-muted">
              <tr>
                {["Clip", "System", "Truth", "Measured", "z", "Interval has truth", "Model", "Confidence"].map((h) => (
                  <th key={h} scope="col" className="px-3 py-2 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {scored.map((r) => (
                <tr key={r.clip_id} className="border-t border-line">
                  <th scope="row" className="px-3 py-2 text-left font-normal">
                    {r.experiment_id ? (
                      <Link href={`/e/${r.experiment_id}`} className="text-ghost underline-offset-4 hover:underline">
                        {r.clip_id}
                      </Link>
                    ) : (
                      r.clip_id
                    )}
                    <span className="block text-xs text-muted">{r.object}</span>
                  </th>
                  <td className="px-3 py-2">{SYSTEM_LABELS[r.system]}</td>
                  <td className="px-3 py-2">
                    {r.quantity} = {fmtNum(r.truth, 4)} {r.unit}
                    <span className="block text-xs text-muted">± {fmtNum(r.truth_sigma, 2)} · {r.truth_method}</span>
                  </td>
                  <td className="px-3 py-2">
                    {r.measured === null || r.ci_low === null || r.ci_high === null ? "–" : `${fmtPm(r.measured, (r.ci_high - r.ci_low) / 2)} ${r.unit}`}
                  </td>
                  <td className="px-3 py-2">{r.z === null ? "–" : r.z.toFixed(1)}</td>
                  <td className="px-3 py-2">{mark(r.covered)}</td>
                  <td className="px-3 py-2">
                    {r.model_selected ?? "–"} {r.model_correct === null ? null : mark(r.model_correct)}
                  </td>
                  <td className="px-3 py-2"><ConfidenceBadge level={r.confidence} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <SectionTitle hint="Expected result: low confidence, for the right reason">Failure cases</SectionTitle>
        <ul className="grid gap-3 sm:grid-cols-3">
          {failures.map((r) => (
            <li key={r.clip_id}>
              <Card className="h-full">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone={r.failure_flagged ? "green" : "red"}>{r.failure_flagged ? "Flagged" : "Missed"}</Badge>
                  <ConfidenceBadge level={r.confidence} />
                </div>
                <p className="mt-2 text-sm font-medium">{r.object}</p>
                <p className="text-sm text-muted">Expected: {r.failure_expected}</p>
                {r.note ? <p className="mt-1 text-sm text-zinc-300">{r.note}</p> : null}
                {r.experiment_id ? (
                  <Link href={`/e/${r.experiment_id}`} className="mt-2 inline-flex min-h-11 items-center text-sm text-ghost">
                    Open the experiment →
                  </Link>
                ) : null}
              </Card>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function mark(v: boolean | null) {
  if (v === null) return <span className="text-muted">–</span>;
  return v ? <span className="text-emerald-300" aria-label="yes">✓</span> : <span className="text-red-300" aria-label="no">✗</span>;
}

function Tile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-1 text-2xl font-semibold tabular-nums">{value}</dd>
      {hint ? <dd className="text-xs text-muted">{hint}</dd> : null}
    </div>
  );
}

/** Relative error with 95% bars, one row per clip. */
function ErrorPlot({ rows }: { rows: BenchmarkRow[] }) {
  const [ref, width] = useElementWidth<HTMLDivElement>(560);
  const data = rows
    .filter((r) => r.measured !== null && r.ci_low !== null && r.ci_high !== null)
    .map((r) => ({
      id: r.clip_id,
      err: (r.measured! - r.truth) / Math.abs(r.truth),
      lo: (r.ci_low! - r.truth) / Math.abs(r.truth),
      hi: (r.ci_high! - r.truth) / Math.abs(r.truth),
      ok: r.covered,
    }));
  const extent = Math.max(0.05, ...data.flatMap((d) => [Math.abs(d.lo), Math.abs(d.hi)])) * 1.1;
  const left = Math.min(110, width * 0.28);
  const row = 22;
  const h = data.length * row + 34;
  const sx = (v: number) => left + ((v + extent) / (2 * extent)) * (width - left - 12);
  const ticks = niceTicks(-extent, extent, 5);
  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={h} role="img" aria-label="Relative error of every benchmark clip with its 95% interval">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={6} y2={h - 22} stroke={t === 0 ? "#9aa9ba" : CHART_COLORS.grid} />
            <text x={sx(t)} y={h - 6} textAnchor="middle" fontSize="11" fill={CHART_COLORS.muted}>
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {data.map((d, i) => {
          const y = 12 + i * row;
          const color = d.ok ? CHART_COLORS.green : CHART_COLORS.red;
          return (
            <g key={d.id}>
              <text x={left - 8} y={y + 4} textAnchor="end" fontSize="11" fill={CHART_COLORS.muted}>
                {d.id}
              </text>
              <line x1={sx(d.lo)} x2={sx(d.hi)} y1={y} y2={y} stroke={color} strokeWidth={2} />
              <line x1={sx(d.lo)} x2={sx(d.lo)} y1={y - 4} y2={y + 4} stroke={color} strokeWidth={2} />
              <line x1={sx(d.hi)} x2={sx(d.hi)} y1={y - 4} y2={y + 4} stroke={color} strokeWidth={2} />
              <circle cx={sx(d.err)} cy={y} r={3.5} fill={color} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}
