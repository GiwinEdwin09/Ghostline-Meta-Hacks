"use client";

import { Markdown } from "@/components/Markdown";
import { Badge, Card, Notice, ReviewBadge, SectionTitle, cn } from "@/components/ui";
import type { Experiment, ReviewIssue } from "@/lib/types";

const ACTION: Record<ReviewIssue["requested_action"], string> = {
  run_model: "Run another model",
  refit: "Refit",
  lower_confidence: "Lower confidence",
  add_caveat: "Add a caveat",
  none: "No action",
};

const VERDICT: Record<string, string> = {
  accept: "Accepts",
  accept_with_caveats: "Accepts with caveats",
  challenge: "Challenges",
};

/** The referee's thread (build-plan Part 14) with Wonder's answers, and the final status. */
export function ReviewTab({ exp }: { exp: Experiment }) {
  const review = exp.review;
  if (!review || review.rounds.length === 0) {
    return (
      <Notice tone="amber" title="Not reviewed">
        The reviewer endpoint wasn&apos;t available for this analysis, so the result has not had a second opinion.
      </Notice>
    );
  }
  return (
    <div className="space-y-5">
      <Card className={cn(review.status === "contested" && "border-red-400/40", review.status === "accepted" && "border-emerald-400/40")}>
        <div className="flex flex-wrap items-center gap-2">
          <ReviewBadge status={review.status} />
          <span className="text-sm text-muted">
            after {review.rounds.length} round{review.rounds.length === 1 ? "" : "s"}
          </span>
        </div>
        <p className="mt-2 text-sm text-zinc-300">
          {review.status === "contested"
            ? "The reviewer still disagrees after two rounds. Both sides are shown below, and the confidence reflects it."
            : "Muse Spark Open sees only the claim, the comparison table, the uncertainty budget, the quality flags and two plots, the way a real referee would. It never sees Wonder's reasoning."}
        </p>
      </Card>

      {review.rounds.map((r) => (
        <section key={r.round} aria-label={`Round ${r.round}`} className="space-y-3">
          <SectionTitle>Round {r.round}</SectionTitle>
          <div className="rounded-xl border border-violet-400/25 bg-violet-400/5 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-violet-300">Muse Spark Open</span>
              <Badge tone={r.verdict === "challenge" ? "red" : r.verdict === "accept" ? "green" : "amber"}>{VERDICT[r.verdict]}</Badge>
            </div>
            <p className="mt-2 text-sm text-zinc-200">{r.summary}</p>
            {r.issues.length > 0 ? (
              <ul className="mt-3 space-y-3">
                {r.issues.map((i) => (
                  <li key={i.id} className="rounded-lg border border-line bg-background p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={i.severity === "major" ? "red" : "amber"}>{i.severity}</Badge>
                      <Badge>{ACTION[i.requested_action]}</Badge>
                    </div>
                    <p className="mt-2 text-sm font-medium">{i.claim}</p>
                    <p className="mt-1 text-sm text-zinc-300">{i.evidence}</p>
                    <p className="mt-1 text-sm text-muted">{i.detail}</p>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
          {r.response_md ? (
            <div className="rounded-xl border border-sky-400/25 bg-sky-400/5 p-4 sm:ml-6">
              <span className="text-sm font-medium text-sky-300">Muse Wonder answers</span>
              <Markdown text={r.response_md} className="mt-2" />
            </div>
          ) : null}
        </section>
      ))}
    </div>
  );
}
