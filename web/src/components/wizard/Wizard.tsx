"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { Button, Card, ErrorState, Notice, Skeleton, chipClass, cn, inputClass } from "@/components/ui";
import { UploadPanel } from "@/components/wizard/UploadPanel";
import { VideoPlayer } from "@/components/VideoPlayer";
import { api, resolveMedia } from "@/lib/api";
import { SYSTEM_LABELS, fmtTime } from "@/lib/format";
import { useResource } from "@/lib/hooks";
import type { CaptureMode, PaperSize, SystemType } from "@/lib/types";

type Choice = SystemType | "unknown";

const SYSTEMS: { id: Choice; blurb: string; icon: ReactNode }[] = [
  { id: "pendulum", blurb: "A weight on a string, swinging", icon: <Svg d="M12 3v8M12 11l6 8M12 11 6 19" circle={[12, 20]} /> },
  { id: "free_fall", blurb: "A ball dropped from your hand", icon: <Svg d="M12 3v10M9 10l3 3 3-3" circle={[12, 19]} /> },
  { id: "ramp", blurb: "A block or ball on a slope", icon: <Svg d="M4 19 20 9M4 19h16" circle={[15, 11]} /> },
  { id: "bounce", blurb: "A ball bouncing on the floor", icon: <Svg d="M5 4c3 0 4 12 7 12s4-12 7-12M4 20h16" /> },
  { id: "spin_down", blurb: "A fidget spinner slowing down", icon: <Svg d="M12 12m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M12 10V4M13.7 13l5-3M10.3 13l-5-3" /> },
  { id: "unknown", blurb: "Let Ghostline work it out", icon: <Svg d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 1-1 1.7M12 17h.01" /> },
];

function Svg({ d, circle }: { d: string; circle?: [number, number] }) {
  return (
    <svg viewBox="0 0 24 24" className="size-7" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
      {circle ? <circle cx={circle[0]} cy={circle[1]} r="2" /> : null}
    </svg>
  );
}

const COMMON_CHECKLIST = [
  { item: "Phone on a tripod (or a stack of books), level, in landscape", why: "Handheld shake looks like physics to the tracker." },
  { item: "1x lens, with focus and exposure locked (tap and hold)", why: "The ultra-wide lens bends straight lines, and changing exposure confuses tracking." },
  { item: "An A4 sheet, flat, taped at the corners, fully visible against a dark background", why: "The sheet is the ruler: it turns pixels into meters." },
  { item: "Stand at least 2 m back (use the 2x lens if you have it)", why: "Distance shrinks the error from an object that isn't exactly in the sheet's plane." },
  { item: "Bright, steady daylight; start recording before the release", why: "Flickering LED light shows up as bands in slow motion." },
];

const SYSTEM_TIPS: Partial<Record<Choice, string>> = {
  pendulum: "String at least 50 cm, a heavy nut, a rigid pivot. Try one swing under 15° and one near 40°, and measure the length from the pivot to the bob's center.",
  free_fall: "Drop the ball 1 to 1.5 m, 2 to 3 cm in front of a wall with the sheet on it. A foam ball, which air slows down, makes a more interesting experiment.",
  ramp: "A board at 10° to 30°, filmed side-on, with the sheet on a board behind it. Slide a block, then roll a solid ball.",
  bounce: "Drop a bouncy ball from about 1 m next to the wall. 240 fps slow-mo helps; record sound on a second phone to check the result.",
  spin_down: "Spinner flat on a table, camera directly above, sheet flat on the table, a dot sticker on one arm. A gentle flick at 60 fps.",
  unknown: "Film the motion as steadily as you can. Wonder will watch it and decide what it is.",
};

const STEPS = ["What", "Set up", "Upload", "Confirm"] as const;

export function Wizard() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [system, setSystem] = useState<Choice | null>(null);
  const [offsetCm, setOffsetCm] = useState("2");
  const [paper, setPaper] = useState<PaperSize>("a4");
  const [lengthCm, setLengthCm] = useState("");
  const [mode, setMode] = useState<CaptureMode>("normal");
  const [fps, setFps] = useState("60");
  const [description, setDescription] = useState("");
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [trialId, setTrialId] = useState<string | null>(null);
  const creating = useRef<Promise<string> | null>(null);
  const root = useRef<HTMLElement>(null);
  const shownStep = useRef(step);

  // Each step starts at the top of the page, with focus on its heading for screen readers and keyboards.
  useEffect(() => {
    if (shownStep.current === step) return;
    shownStep.current = step;
    window.scrollTo({ top: 0 });
    root.current?.querySelector<HTMLElement>("h1")?.focus({ preventScroll: true });
  }, [step]);

  const ensureExperiment = () => {
    if (experimentId) return Promise.resolve(experimentId);
    if (!creating.current) {
      const sys = system ?? "unknown";
      creating.current = api
        .createExperiment({
          title: sys === "unknown" ? "New experiment" : `${SYSTEM_LABELS[sys]} experiment`,
          system_hint: sys,
          description,
          declared_capture: { mode, fps: Number(fps) || 60 },
          plane_offset_cm: Number(offsetCm) || 2,
          paper,
          pendulum_length_cm: lengthCm ? Number(lengthCm) : null,
        })
        .then(({ id }) => {
          setExperimentId(id);
          return id;
        })
        .finally(() => {
          creating.current = null;
        });
    }
    return creating.current;
  };

  return (
    <main ref={root} className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:py-12">
      <nav aria-label="Progress" className="mb-8">
        <p className="eyebrow mb-3">New experiment</p>
        <ol className="grid grid-cols-4 gap-1.5">
          {STEPS.map((s, i) => (
            <li key={s} aria-current={i === step ? "step" : undefined}>
              <span
                aria-hidden
                className={cn("block h-1 rounded-full transition-colors", i < step ? "bg-foreground/60" : i === step ? "bg-ghost" : "bg-line-strong")}
              />
              <span className={cn("mt-2 flex items-baseline gap-1.5 text-sm", i === step ? "text-foreground" : "text-muted")}>
                <span className="font-mono text-xs">{i < step ? "✓" : String(i + 1).padStart(2, "0")}</span>
                <span className="hidden sm:inline">{s}</span>
                <span className="sr-only sm:hidden">{s}</span>
              </span>
            </li>
          ))}
        </ol>
        <p className="mt-2 text-sm text-muted sm:hidden">
          Step {step + 1} of {STEPS.length}: {STEPS[step]}
        </p>
      </nav>

      {step === 0 ? (
        <section aria-labelledby="step1">
          <h1 id="step1" tabIndex={-1} className="font-display outline-none text-[2.6rem] leading-[1.05] sm:text-5xl">
            What are you filming?
          </h1>
          <p className="mt-2 text-sm text-muted">Pick the closest match. If none fits, choose Not sure.</p>
          <ul className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3">
            {SYSTEMS.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  aria-pressed={system === s.id}
                  onClick={() => {
                    setSystem(s.id);
                    setStep(1);
                  }}
                  className={cn(
                    "group relative flex min-h-36 w-full flex-col items-start gap-2 overflow-hidden rounded-md border p-4 text-left transition-colors",
                    system === s.id ? "border-ghost bg-ghost/[0.08]" : "border-line bg-surface hover:border-muted/50 hover:bg-surface-2",
                  )}
                >
                  <span aria-hidden className="graph-paper absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100" />
                  <span className="relative text-ghost">{s.icon}</span>
                  <span className="relative mt-auto font-medium">{s.id === "unknown" ? "Not sure" : SYSTEM_LABELS[s.id as SystemType]}</span>
                  <span className="relative text-xs leading-relaxed text-muted">{s.blurb}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {step === 1 && system ? (
        <section aria-labelledby="step2" className="space-y-6">
          <div>
            <h1 id="step2" tabIndex={-1} className="font-display outline-none text-[2.6rem] leading-[1.05] sm:text-5xl">
              Set up the shot
            </h1>
            <p className="mt-1 text-sm text-muted">Most accuracy problems start at the camera. Five things to check first.</p>
          </div>
          <ul className="space-y-2">
            {COMMON_CHECKLIST.map((c, i) => (
              <li key={i}>
                <label className="flex min-h-11 cursor-pointer items-start gap-3 rounded-md border border-line bg-surface p-3.5 transition-colors hover:border-line-strong has-[:checked]:border-ghost/40 has-[:checked]:bg-ghost/[0.04]">
                  <input type="checkbox" className="mt-0.5 size-5 shrink-0 accent-ghost" />
                  <span>
                    <span className="block text-sm font-medium">{c.item}</span>
                    <span className="block text-xs text-muted">{c.why}</span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <Notice tone="cyan" title={`For ${system === "unknown" ? "your clip" : SYSTEM_LABELS[system].toLowerCase()}`}>
            {SYSTEM_TIPS[system]}
          </Notice>

          <Card className="space-y-5">
            <Field
              id="offset"
              label="How far in front of the sheet does the object move?"
              hint="In cm. Ghostline corrects for it. Measure roughly; 2 cm is a good guess for an object near a wall."
            >
              <input
                id="offset"
                inputMode="decimal"
                value={offsetCm}
                onChange={(e) => setOffsetCm(e.target.value)}
                className={cn(inputClass, "h-11 w-32 font-mono")}
              />
            </Field>
            <fieldset>
              <legend className="text-sm font-medium">Paper size</legend>
              <p className="text-xs text-muted">Ghostline checks this from the picture, so this is only a confirmation.</p>
              <div className="mt-2 flex flex-wrap gap-2">
                {(["a4", "letter"] as const).map((p) => (
                  <Radio key={p} name="paper" checked={paper === p} onChange={() => setPaper(p)} label={p === "a4" ? "A4" : "US Letter"} />
                ))}
              </div>
            </fieldset>
            <fieldset>
              <legend className="text-sm font-medium">How did you record it?</legend>
              <div className="mt-2 flex flex-wrap gap-2">
                <Radio
                  name="capture"
                  checked={mode === "normal"}
                  onChange={() => {
                    setMode("normal");
                    setFps("60");
                  }}
                  label="Normal video, 60 fps"
                />
                <Radio
                  name="capture"
                  checked={mode === "slow_mo"}
                  onChange={() => {
                    setMode("slow_mo");
                    setFps("240");
                  }}
                  label="Slow-mo, 240 fps"
                />
              </div>
              {mode === "slow_mo" ? (
                <p className="mt-2 text-xs text-muted">
                  Slow-mo is best for drops and bounces. Phones sometimes re-time slow-mo clips when they are exported; Ghostline checks for that.
                </p>
              ) : null}
            </fieldset>
            {system === "pendulum" ? (
              <Field id="length" label="Measured pendulum length (optional)" hint="In cm, from the pivot to the center of the bob. Lets Ghostline report g both ways.">
                <input
                  id="length"
                  inputMode="decimal"
                  value={lengthCm}
                  onChange={(e) => setLengthCm(e.target.value)}
                  className={cn(inputClass, "h-11 w-32 font-mono")}
                />
              </Field>
            ) : null}
            <Field id="desc" label="Anything Ghostline should know? (optional)" hint="For example: a steel nut on 62 cm of string.">
              <textarea
                id="desc"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className={cn(inputClass, "block w-full py-2")}
              />
            </Field>
          </Card>
          <div className="flex justify-between gap-2 border-t border-line pt-5">
            <Button tone="ghost" onClick={() => setStep(0)}>
              ← Back
            </Button>
            <Button tone="primary" onClick={() => setStep(2)}>
              Continue to upload →
            </Button>
          </div>
        </section>
      ) : null}

      {step === 2 ? (
        <section aria-labelledby="step3" className="space-y-5">
          <div>
            <h1 id="step3" tabIndex={-1} className="font-display outline-none text-[2.6rem] leading-[1.05] sm:text-5xl">
              Upload your clip
            </h1>
            <p className="mt-1 text-sm text-muted">Record it with your camera app first, then pick it here.</p>
          </div>
          <UploadPanel
            role="fit"
            ensureExperiment={ensureExperiment}
            existingExperimentId={experimentId}
            onUploaded={(expId, tId) => {
              setExperimentId(expId);
              setTrialId(tId);
              setStep(3);
            }}
          />
          <div className="border-t border-line pt-5">
            <Button tone="ghost" onClick={() => setStep(1)}>
              ← Back
            </Button>
          </div>
        </section>
      ) : null}

      {step === 3 && experimentId && trialId ? (
        <Confirm
          experimentId={experimentId}
          onAnalyze={() => router.push(`/e/${experimentId}`)}
        />
      ) : null}
    </main>
  );
}

function Field({ id, label, hint, children }: { id: string; label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium">
        {label}
      </label>
      {hint ? <p className="mt-0.5 text-xs leading-relaxed text-muted">{hint}</p> : null}
      <div className="mt-2">{children}</div>
    </div>
  );
}

function Radio({ name, checked, onChange, label }: { name: string; checked: boolean; onChange: () => void; label: string }) {
  return (
    <label className={chipClass(checked)}>
      <input type="radio" name={name} checked={checked} onChange={onChange} className="size-4 accent-ghost" />
      {label}
    </label>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Step 4: confirm                                                                       */
/* ------------------------------------------------------------------------------------ */

function Confirm({ experimentId, onAnalyze }: { experimentId: string; onAnalyze: () => void }) {
  const exp = useResource(`wizard:${experimentId}`, () => api.getExperiment(experimentId), {
    intervalMs: 1000,
  });
  const trial = exp.data?.trials[0];
  const ready = trial?.status === "ready" || trial?.status === "tracked";
  const [start, setStart] = useState<number | null>(null);
  const [end, setEnd] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  if (exp.error && !exp.data) return <ErrorState error={exp.error} onRetry={exp.reload} />;
  if (!trial || !ready) {
    return (
      <section aria-live="polite" className="space-y-4">
        <h1 tabIndex={-1} className="font-display text-[2.6rem] leading-[1.05] outline-none sm:text-5xl">Preparing your clip</h1>
        <p className="text-sm text-muted">Fixing orientation and color, and reading the real timestamp of every frame.</p>
        <Skeleton className="aspect-video w-full" />
        {trial?.status === "failed" ? <ErrorState error={new Error(trial.error ?? "The clip could not be read.")} /> : null}
      </section>
    );
  }

  const duration = trial.duration_s;
  const s = start ?? trial.trim.suggested_start_s;
  const e = end ?? trial.trim.suggested_end_s;
  const timing = trial.timing;
  const gap = 0.5;

  const analyze = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.setTrim(trial.id, { start_s: s, end_s: e });
      await api.analyze(experimentId);
      onAnalyze();
    } catch (err) {
      setError(err);
      setBusy(false);
    }
  };

  return (
    <section aria-labelledby="step4" className="space-y-5">
      <div>
        <h1 id="step4" tabIndex={-1} className="font-display outline-none text-[2.6rem] leading-[1.05] sm:text-5xl">
          Check the clip
        </h1>
        <p className="mt-1 text-sm text-muted">Ghostline suggested the stretch where something moves. Drag the handles to change it.</p>
      </div>
      <VideoPlayer
        src={resolveMedia(trial.media.playback_url)!}
        poster={resolveMedia(trial.media.poster_url)}
        aspect={trial.width / trial.height}
        label="Your uploaded clip"
        rates={false}
      />
      <Card className="space-y-4">
        <div>
          <p className="eyebrow">Analysis window</p>
          <p className="mt-1.5 text-sm font-medium">
            Analyze from <span className="font-mono text-ghost">{fmtTime(s)}</span> to <span className="font-mono text-ghost">{fmtTime(e)}</span>
            <span className="font-normal text-muted"> of {fmtTime(duration)}</span>
          </p>
          <TrimBar duration={duration} start={s} end={e} />
          <div className="mt-3 space-y-1">
            <label className="text-xs text-muted" htmlFor="trim-start">
              Start
            </label>
            <input
              id="trim-start"
              type="range"
              min={0}
              max={duration}
              step={0.05}
              value={s}
              onChange={(ev) => setStart(Math.min(Number(ev.target.value), e - gap))}
              className="h-11 w-full accent-ghost"
            />
            <label className="text-xs text-muted" htmlFor="trim-end">
              End
            </label>
            <input
              id="trim-end"
              type="range"
              min={0}
              max={duration}
              step={0.05}
              value={e}
              onChange={(ev) => setEnd(Math.max(Number(ev.target.value), s + gap))}
              className="h-11 w-full accent-ghost"
            />
          </div>
        </div>
        {timing ? (
          <dl className="grid grid-cols-2 gap-3 border-t border-line pt-4 text-sm">
            <div>
              <dt className="eyebrow">Capture</dt>
              <dd className="mt-1">
                {timing.declared_mode === "slow_mo" ? "Slow-mo" : "Normal video"}, {timing.detected_fps} fps
              </dd>
            </div>
            <div>
              <dt className="eyebrow">Timing</dt>
              <dd className="mt-1">{timing.classification === "cfr" ? "Steady frame rate" : timing.classification.replace("_", " ")}</dd>
            </div>
          </dl>
        ) : null}
        {timing && (timing.classification === "retimed" || timing.time_factor !== 1 || timing.warnings.length > 0) ? (
          <Notice tone="amber" title="This clip looks re-timed">
            {timing.warnings.join(" ") || "The frame spacing does not match what you said. Confirm only if you are sure, or upload the original from a computer."}
          </Notice>
        ) : null}
      </Card>
      {error ? <ErrorState error={error} /> : null}
      <div className="flex justify-end border-t border-line pt-5">
        <Button tone="primary" onClick={analyze} disabled={busy} className="min-h-12 px-6 text-base">
          {busy ? "Starting…" : "Analyze →"}
        </Button>
      </div>
    </section>
  );
}

/** The chosen stretch of the clip, drawn on a timeline above the two sliders. */
function TrimBar({ duration, start, end }: { duration: number; start: number; end: number }) {
  const pct = (v: number) => `${Math.min(100, Math.max(0, (v / (duration || 1)) * 100))}%`;
  return (
    <div aria-hidden className="relative mt-4 h-8 rounded-sm border border-line bg-background">
      <div className="graph-paper absolute inset-0 opacity-70" />
      <div className="absolute inset-y-0 border-x-2 border-ghost bg-ghost/15" style={{ left: pct(start), width: `calc(${pct(end)} - ${pct(start)})` }} />
    </div>
  );
}
