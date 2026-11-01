"use client";

import { ModelCard } from "@/components/ModelCard";
import { Notice } from "@/components/ui";
import type { Experiment } from "@/lib/types";

/** One card per model, best held-out error first (build-plan Part 24). */
export function HypothesesTab({ exp }: { exp: Experiment }) {
  const models = [...exp.models].sort((a, b) => (a.fit?.heldout_rmse ?? Infinity) - (b.fit?.heldout_rmse ?? Infinity));
  if (models.length === 0) return <Notice tone="cyan">Hypotheses appear once Wonder has proposed them.</Notice>;
  const best = models[0].fit?.heldout_rmse ?? 1;
  const worst = Math.max(...models.map((m) => m.fit?.heldout_rmse ?? 0));
  // Keep the good models readable: a refuted model can be a hundred times worse, so cut its bar.
  const barMax = Math.min(worst, best * 4) * 1.15;
  const favorite = exp.models.find((m) => m.prior_rank === 1);
  return (
    <div className="space-y-4">
      <p className="max-w-3xl text-sm leading-relaxed text-fg-2">
        Wonder named {favorite ? <strong className="font-semibold text-foreground">{favorite.name}</strong> : "its favorite"} as its first guess before any fit, along with what would refute it.
        Each model was fitted on the first 65% of the motion and judged on how well it forecast the rest. Lower is better; the whiskers are 95% intervals.
      </p>
      {models.map((m) => (
        <ModelCard key={m.id} model={m} barMax={barMax} />
      ))}
    </div>
  );
}
