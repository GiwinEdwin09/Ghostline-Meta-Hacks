"use client";

import Link from "next/link";

import { Badge, Card, ErrorState, SectionTitle, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { useResource } from "@/lib/hooks";

/** Jobs, failures, cost per experiment, cache hit rate and the daily budget (build-plan Part 24). */
export function AdminView() {
  // The back end requires the passcode; the API client asks for it the first time.
  const admin = useResource("admin", () => api.getAdminOverview(), { intervalMs: 5000 });
  if (admin.error && !admin.data) return <ErrorState error={admin.error} onRetry={admin.reload} />;
  if (!admin.data) return <Skeleton className="h-64 w-full" />;
  const { budget, cache, jobs, experiments } = admin.data;
  const used = Math.min(1, budget.spent_today_usd / budget.daily_usd);
  const failed = jobs.filter((j) => j.status === "failed");
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <SectionTitle>Daily budget</SectionTitle>
          <p className="font-mono text-3xl font-medium tabular-nums tracking-tight">
            ${budget.spent_today_usd.toFixed(2)} <span className="text-base font-normal text-muted">of ${budget.daily_usd.toFixed(2)}</span>
          </p>
          <div role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(used * 100)} aria-label="Daily budget used" className="mt-3 h-2 overflow-hidden rounded-full bg-surface-2">
            <div className={used > 0.8 ? "h-full bg-red-400" : "h-full bg-ghost"} style={{ width: `${used * 100}%` }} />
          </div>
          <p className="mt-2 text-xs text-muted">Each experiment is capped at ${budget.per_experiment_usd.toFixed(2)}; past that it finishes in library-only mode.</p>
        </Card>
        <Card>
          <SectionTitle>Cache</SectionTitle>
          <p className="font-mono text-3xl font-medium tabular-nums tracking-tight">{Math.round(cache.hit_rate * 100)}%</p>
          <p className="text-sm text-muted">hit rate over {cache.calls_24h} model calls in the last 24 hours</p>
        </Card>
      </div>

      <Card>
        <SectionTitle hint={failed.length ? `${failed.length} failed` : undefined}>Jobs</SectionTitle>
        {jobs.length === 0 ? (
          <p className="text-sm text-muted">No jobs are running or recently failed.</p>
        ) : (
          <ul className="divide-y divide-line">
            {jobs.map((j) => (
              <li key={j.id} className="flex flex-wrap items-center justify-between gap-2 py-2.5 text-sm">
                <span>
                  <Link href={`/e/${j.experiment_id}`} className="text-ghost underline-offset-4 hover:underline">
                    {j.experiment_title}
                  </Link>
                  <span className="ml-2 font-mono text-xs text-muted">
                    {j.kind} · {j.phase}
                  </span>
                </span>
                <Badge tone={j.status === "failed" ? "red" : j.status === "succeeded" ? "green" : "cyan"}>{j.status}</Badge>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card>
        <SectionTitle>Cost per experiment</SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[24rem] text-sm">
            <thead className="text-left">
              <tr>
                <th scope="col" className="eyebrow pb-2 font-medium">Experiment</th>
                <th scope="col" className="eyebrow pb-2 font-medium">Mode</th>
                <th scope="col" className="eyebrow pb-2 text-right font-medium">Cost</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {experiments.map((e) => (
                <tr key={e.id} className="border-t border-line">
                  <td className="py-2 pr-3">
                    <Link href={`/e/${e.id}`} className="text-ghost underline-offset-4 hover:underline">
                      {e.title}
                    </Link>
                  </td>
                  <td className="py-2 pr-3 font-mono text-xs text-muted">{e.llm_mode}</td>
                  <td className="py-2 text-right font-mono">${e.cost_usd.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
