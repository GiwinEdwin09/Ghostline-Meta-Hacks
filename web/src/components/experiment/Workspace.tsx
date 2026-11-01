"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { HypothesesTab } from "@/components/experiment/HypothesesTab";
import { MeasurementTab } from "@/components/experiment/MeasurementTab";
import { ProveTab } from "@/components/experiment/ProveTab";
import { ResultTab } from "@/components/experiment/ResultTab";
import { ReviewTab } from "@/components/experiment/ReviewTab";
import { NotebookTimeline } from "@/components/NotebookTimeline";
import { PipelineStepper } from "@/components/PipelineStepper";
import { Tabs, type TabSpec } from "@/components/Tabs";
import {
  ArrowLink,
  Button,
  ConfidenceBadge,
  ErrorState,
  Notice,
  ReviewBadge,
  Skeleton,
} from "@/components/ui";
import { api } from "@/lib/api";
import { SYSTEM_LABELS } from "@/lib/format";
import { useExperimentLive } from "@/lib/hooks";
import type { Experiment } from "@/lib/types";

const WhatIf = dynamic(() => import("@/components/experiment/WhatIf"), {
  ssr: false,
  loading: () => <Skeleton className="aspect-video w-full" />,
});

type TabId = "result" | "measurement" | "hypotheses" | "review" | "notebook" | "prove" | "whatif";
const TAB_IDS: TabId[] = ["result", "measurement", "hypotheses", "review", "notebook", "prove", "whatif"];

export function Workspace({ id }: { id: string }) {
  const live = useExperimentLive(id);
  const router = useRouter();
  const params = useSearchParams();
  const requested = params.get("tab");
  const [retryError, setRetryError] = useState<unknown>(null);
  const pendingScroll = useRef(false);
  const exp = live.experiment;

  // After a tab change, if the reader was below the tabs, put the top of the new panel just under
  // the sticky tab bar. It runs once the panel has rendered, so a height change can't cut it short.
  useEffect(() => {
    if (!pendingScroll.current) return;
    pendingScroll.current = false;
    const panel = document.querySelector<HTMLElement>('[role="tabpanel"]');
    const bar = document.querySelector<HTMLElement>('[role="tablist"]')?.parentElement;
    if (!panel || !bar) return;
    const stuckAt = parseFloat(getComputedStyle(bar).top) || 0;
    const target = panel.getBoundingClientRect().top + window.scrollY - stuckAt - bar.offsetHeight - 16;
    if (window.scrollY > target) window.scrollTo({ top: Math.max(0, target) });
  }, [requested]);

  if (live.loading) return <WorkspaceSkeleton />;
  if (live.error && !exp) {
    return (
      <Shell>
        <ErrorState error={live.error} onRetry={live.refresh} />
        <ArrowLink href="/examples" className="mt-4">
          Back to the examples
        </ArrowLink>
      </Shell>
    );
  }
  if (!exp) return <WorkspaceSkeleton />;

  const complete = exp.status === "complete" && exp.conclusion !== null;
  const tab: TabId = TAB_IDS.includes(requested as TabId) && isEnabled(requested as TabId, exp, complete) ? (requested as TabId) : complete ? "result" : exp.trials.length > 0 ? "notebook" : "result";
  const setTab = (next: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    pendingScroll.current = true;
    router.replace(`${url.pathname}${url.search}`, { scroll: false });
  };

  const retry = async () => {
    setRetryError(null);
    try {
      await (live.job?.kind === "prove_trial"
        ? api.prove(exp.id, exp.trials.find((t) => t.role === "prove")?.id ?? "")
        : api.analyze(exp.id));
      live.refresh();
    } catch (err) {
      setRetryError(err);
    }
  };

  const tabs: TabSpec[] = [
    { id: "result", label: "Result", disabled: !complete },
    { id: "measurement", label: "Measurement", disabled: exp.trials.length === 0 },
    { id: "hypotheses", label: "Hypotheses", hint: exp.models.length || undefined, disabled: exp.models.length === 0 },
    { id: "review", label: "Review", disabled: !complete },
    { id: "notebook", label: "Notebook", hint: live.events.length || undefined, disabled: live.events.length === 0 && !live.active },
    { id: "prove", label: "Prove it", disabled: !complete },
    { id: "whatif", label: "What if", disabled: !complete || !exp.whatif },
  ];

  const running = live.active && live.job;
  return (
    <Shell>
      <header>
        <Link href="/examples" className="-ml-1 inline-flex min-h-11 items-center gap-1.5 rounded-md px-1 text-sm text-muted hover:text-foreground">
          <span aria-hidden>←</span> Examples
        </Link>
        <p className="eyebrow mt-3">
          {SYSTEM_LABELS[exp.system_type]} · <StatusLabel exp={exp} />
        </p>
        <h1 className="mt-2 font-display text-[2.4rem] leading-[1.04] tracking-[-0.01em] sm:text-5xl">{exp.title}</h1>
        {exp.headline ? <p className="mt-4 font-mono text-lg text-ghost sm:text-xl">{exp.headline}</p> : null}
        {exp.confidence || exp.review_status ? (
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <ConfidenceBadge level={exp.confidence} />
            <ReviewBadge status={exp.review_status} />
          </div>
        ) : null}
        {exp.description ? <p className="mt-3 max-w-2xl text-sm leading-relaxed text-muted">{exp.description}</p> : null}
      </header>

      {running ? (
        <PipelineStepper job={live.job!} mode={live.job!.kind === "prove_trial" ? "prove" : "analyze"} onRetry={retry} />
      ) : live.job?.status === "failed" ? (
        <PipelineStepper job={live.job} onRetry={retry} />
      ) : null}
      {retryError ? <ErrorState error={retryError} /> : null}

      {exp.status === "failed" || exp.status === "blocked" ? (
        <Notice tone="red" title={exp.status === "blocked" ? "Wonder couldn't find something to measure" : "The analysis failed"}>
          {exp.error ?? "Try again, or film the motion again following the setup tips."}
          <div className="mt-3">
            <Button onClick={retry}>Retry</Button>
          </div>
        </Notice>
      ) : null}

      {exp.status === "ready" && !live.active ? (
        <Notice tone="cyan" title="Your clip is ready">
          <p>It has not been analyzed yet.</p>
          <Button tone="primary" className="mt-3" onClick={retry}>
            Analyze
          </Button>
        </Notice>
      ) : null}

      {exp.trials.length > 0 ? (
        <>
          <Tabs tabs={tabs} active={tab} onChange={setTab} idPrefix="exp" />
          <div role="tabpanel" id={`exp-panel-${tab}`} aria-labelledby={`exp-tab-${tab}`} tabIndex={0} className="min-h-[40vh] rounded-sm pt-1 focus-visible:outline-offset-4">
            {tab === "result" && complete ? <ResultTab exp={exp} events={live.events} goTab={setTab} /> : null}
            {tab === "measurement" ? <MeasurementTab exp={exp} /> : null}
            {tab === "hypotheses" ? <HypothesesTab exp={exp} /> : null}
            {tab === "review" ? <ReviewTab exp={exp} /> : null}
            {tab === "notebook" ? (
              <NotebookTimeline events={live.events} experimentId={exp.id} live={live.active} onChanged={live.refresh} />
            ) : null}
            {tab === "prove" && complete ? <ProveTab exp={exp} job={live.job} onChanged={live.refresh} /> : null}
            {tab === "whatif" && complete && exp.whatif ? <WhatIf exp={exp} /> : null}
          </div>
        </>
      ) : null}
    </Shell>
  );
}

function isEnabled(tab: TabId, exp: Experiment, complete: boolean): boolean {
  switch (tab) {
    case "result":
    case "review":
    case "prove":
      return complete;
    case "whatif":
      return complete && exp.whatif !== null;
    case "measurement":
      return exp.trials.length > 0;
    case "hypotheses":
      return exp.models.length > 0;
    case "notebook":
      return true;
  }
}

function StatusLabel({ exp }: { exp: Experiment }) {
  const label: Record<Experiment["status"], string> = {
    draft: "Draft",
    ingesting: "Preparing the clip",
    ready: "Ready to analyze",
    analyzing: "Analyzing",
    complete: exp.is_example ? "Example" : "Complete",
    failed: "Failed",
    blocked: "Can't measure this",
  };
  return <span className={exp.status === "failed" || exp.status === "blocked" ? "text-red-300" : undefined}>{label[exp.status]}</span>;
}

function Shell({ children }: { children: React.ReactNode }) {
  return <main className="mx-auto w-full max-w-5xl flex-1 space-y-6 px-4 py-6 sm:py-10">{children}</main>;
}

function WorkspaceSkeleton() {
  return (
    <Shell>
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-3 w-40" />
      <Skeleton className="h-12 w-3/4" />
      <Skeleton className="h-6 w-1/2" />
      <Skeleton className="h-11 w-full" />
      <Skeleton className="h-64 w-full" />
    </Shell>
  );
}
