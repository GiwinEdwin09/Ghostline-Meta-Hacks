import type { EventActor, EventKind, Experiment, NotebookEvent } from "@/lib/types";
import { fmt, type AnalysisOutput } from "./analysis";

/**
 * Builds a lab notebook (build-plan Part 20) from a mock experiment's structured data. Every
 * number in an event body comes from the same objects the other tabs show.
 */

const BASE = Date.parse("2026-10-27T16:40:00Z");

export function buildNotebook(
  exp: Experiment,
  out: AnalysisOutput,
  opts: { favorite: string; rerunnable: boolean },
): NotebookEvent[] {
  const events: NotebookEvent[] = [];
  let seq = 0;
  let clock = 0;
  const push = (
    phase: string,
    actor: EventActor,
    kind: EventKind,
    title: string,
    body: string,
    data: Record<string, unknown> = {},
  ) => {
    seq += 1;
    clock += 2 + (seq % 5);
    events.push({
      id: seq,
      experiment_id: exp.id,
      seq,
      ts: new Date(BASE + clock * 1000).toISOString(),
      phase,
      actor,
      kind,
      title,
      body_md: body,
      data,
      attachments: [],
      refs: [],
    });
  };

  const trial = exp.trials[0];
  const timing = trial.timing!;
  const cal = trial.calibration!;
  const unit = out.input.unit;
  const byRef = new Map(exp.models.map((m) => [m.id, m]));
  const winner = byRef.get(out.winnerRef)!;

  push(
    "normalize",
    "system",
    "observation",
    "Clip normalized",
    `Read **${trial.n_frames} frames** at ${timing.detected_fps} fps (${timing.classification === "cfr" ? "constant frame rate" : timing.classification}). Time factor ${timing.time_factor}, jitter ${timing.jitter_pct}%. Every frame has a real timestamp.`,
    { timing },
  );
  push("watch", "wonder", "observation", "Wonder watched the clip", exp.scene?.description ?? "", {
    tracked_objects: exp.scene?.tracked_objects,
    quality_issues: exp.scene?.quality_issues,
  });
  push(
    "track",
    "sam",
    "observation",
    `SAM 3.1 tracked ${exp.scene?.tracked_objects.length ?? 1} object${(exp.scene?.tracked_objects.length ?? 1) > 1 ? "s" : ""}`,
    (exp.scene?.tracked_objects ?? []).map((o) => `- "${o.sam_prompt}" (${o.role}): ${o.why}`).join("\n"),
  );
  push(
    "calibrate",
    "harness",
    "observation",
    "Scale and timing calibrated",
    `The ${cal.paper === "a4" ? "A4" : "Letter"} sheet gives **${cal.mm_per_px.toFixed(2)} mm per pixel** with ${(cal.sigma_scale_rel * 100).toFixed(2)}% uncertainty. Plane offset ${cal.plane_offset_cm} cm; camera moved ${cal.camera_motion_px.toFixed(1)} px over the clip.`,
    { calibration: { ...cal, world_to_px: undefined } },
  );

  for (const m of [...exp.models].sort((a, b) => a.prior_rank - b.prior_rank)) {
    push(
      "hypothesize",
      "wonder",
      "hypothesis",
      `Hypothesis ${m.id}: ${m.name}`,
      `${m.mechanism}\n\n$$${m.equations_latex}$$\n\n**Refuted if:** ${m.refuting_signature}`,
      { model_id: m.id, prior_rank: m.prior_rank },
    );
  }
  const fav = byRef.get(opts.favorite)!;
  push(
    "hypothesize",
    "wonder",
    "prediction",
    `Prediction made before any fit: ${fav.name}`,
    `Before looking at a single fit, I expect **${fav.name}**. I will count it refuted if this happens: ${fav.refuting_signature}\n\nThe harness scores every model on the last 35% of the motion, which I never see.`,
    { model_id: fav.id, pinned: "prediction" },
  );

  for (const m of exp.models) {
    const f = m.fit!;
    push(
      "fit",
      "harness",
      "fit",
      `Fit ${m.id}: ${m.name}`,
      `Fitted on the first 65% of the motion, then forecast the rest.\n\n` +
        f.params.map((p) => `- ${p.label}: **${fmt(p.value, 4)}** (${fmt(p.ci_low, 3)} to ${fmt(p.ci_high, 3)}) ${p.unit}`).join("\n") +
        `\n\nTraining RMSE **${fmt(f.train_rmse)} ${unit}**. Held-out RMSE **${fmt(f.heldout_rmse)} ${unit}** (${fmt(f.heldout_ci[0])} to ${fmt(f.heldout_ci[1])}). Lowest on ${f.splits_won} of 3 splits.`,
      { model_id: m.id, split: 0.65, rerunnable: opts.rerunnable, fit: f.params, diagnostics: f.diagnostics },
    );
  }

  let first = true;
  for (const m of exp.models.filter((x) => x.status === "rejected")) {
    push("fit", "wonder", "rejection", `Rejected ${m.id}: ${m.name}`, m.rejection_reason ?? "", {
      model_id: m.id,
      pinned: first ? "rejection" : undefined,
    });
    first = false;
  }
  const tied = exp.models.filter((m) => m.status === "tied");
  push(
    "fit",
    "wonder",
    "acceptance",
    `Winner: ${winner.name}`,
    tied.length
      ? `${winner.name} has the lowest held-out error among the models that cannot be told apart (${tied.map((m) => m.id).join(", ")} overlap with it). The data can't distinguish them; I report the simplest.`
      : `${winner.name} predicts the held-out frames better than every rival, by more than the uncertainty.`,
    { model_id: winner.id },
  );

  if (exp.review) {
    let challenged = false;
    for (const r of exp.review.rounds) {
      const issues = r.issues
        .map((i) => `- **${i.severity}:** ${i.claim} ${i.evidence} *(asks: ${i.requested_action.replace("_", " ")})*`)
        .join("\n");
      const isChallenge = r.verdict === "challenge" && !challenged;
      if (r.verdict === "challenge") challenged = true;
      push(
        "review",
        "spark_open",
        "review",
        `Reviewer, round ${r.round}: ${r.verdict.replace(/_/g, " ")}`,
        `${r.summary}${issues ? `\n\n${issues}` : ""}`,
        { verdict: r.verdict, issues: r.issues, pinned: isChallenge ? "challenge" : undefined },
      );
      if (r.response_md) push("review", "wonder", "response", `Wonder answers round ${r.round}`, r.response_md);
    }
  }
  const c = exp.conclusion!;
  push(
    "review",
    "harness",
    "acceptance",
    `Confidence: ${c.confidence}`,
    c.confidence_reasons.map((r) => `- ${r}`).join("\n"),
    { confidence: c.confidence },
  );

  exp.ghosts.forEach((g, i) => {
    if (g.revision) {
      push("predict", "wonder", "revision", `Why did it drift? Wonder's diagnosis`, g.revision.diagnosis, { ghost_version: g.version });
      push("predict", "harness", "revision", `Revision tested: ${g.model_name}`, g.revision.note, { ghost_version: g.version });
    }
    const last = i === exp.ghosts.length - 1;
    push(
      "render",
      "harness",
      "ghost",
      `Ghost v${g.version} verdict`,
      g.divergence_onset_s === null
        ? `The ghost stayed within **${fmt(g.threshold_mm, 3)} mm** of the real object for the whole clip, predicting ${fmt(g.predicted_ahead_s, 2)} s ahead. Largest error **${fmt(g.max_error_mm, 3)} mm**.`
        : `The ghost stayed within **${fmt(g.threshold_mm, 3)} mm** until ${fmt(g.divergence_onset_s, 3)} s, then drifted. Largest error **${fmt(g.max_error_mm, 3)} mm**.`,
      { ghost_version: g.version, pinned: last ? "ghost_verdict" : undefined },
    );
  });
  return events;
}
