import type { Metadata } from "next";

import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "About" };

const MODELS = [
  {
    name: "Muse Wonder",
    role: "The physicist",
    body: "Watches the clip and decides what to measure. Proposes three to five competing laws, names its favorite and what would refute each, then reads the harness's results, rejects what fails and writes the conclusion. It never computes a number: every number it cites must match the harness.",
  },
  {
    name: "SAM 3.1",
    role: "The tracker",
    body: "Follows the object through every frame and finds the A4 sheet, which turns pixels into meters. Wonder tells it what to look for in a few words, the way Meta recommends.",
  },
  {
    name: "Muse Spark Open",
    role: "The skeptical referee",
    body: "Runs on a Modal Shared Endpoint. Sees only the artifacts, never Wonder's reasoning, and challenges anything the evidence does not support. If it is not convinced after two rounds, the result is shown as contested.",
  },
];

const LIMITS = [
  "The object has to move in the plane of the A4 sheet. Motion toward the camera is flagged, corrected a little, and costs confidence.",
  "The camera must be still. More than 3 px of movement stops the analysis.",
  "Slow-motion clips exported from a phone are sometimes re-timed. Ghostline checks, warns and asks you to confirm, but an original file is always better.",
  "Short clips cannot resolve small effects. When two laws are tied, Ghostline says it can't tell, and says what to film next.",
  "Rotation rates are limited by the frame rate: a spinner that turns more than about a third of a turn per frame will alias.",
  "The what-if simulator runs the law you measured. Beyond the range you measured it is a prediction, and the sliders warn you.",
];

const PRIVACY = [
  "Experiments are private by default, behind unguessable links. Only the examples are public.",
  "Your clip is processed by the Meta Model API in Standard mode, so it is not used to train Meta's models, and stored on Modal.",
  "Copies sent to Meta expire after 7 days. Clips that are not examples are deleted from Ghostline after 14 days, and you can delete an experiment at any time.",
  "Text that appears in a video cannot change a result: the harness computes every number and checks every claim.",
];

const SECTIONS = [
  { id: "arch", label: "The pipeline" },
  { id: "models", label: "Three models" },
  { id: "limits", label: "Limitations" },
  { id: "privacy", label: "Privacy" },
];

const ROLE_COLOR: Record<string, string> = {
  "Muse Wonder": "text-wonder",
  "SAM 3.1": "text-sam",
  "Muse Spark Open": "text-spark",
};

export default function AboutPage() {
  return (
    <main className="flex-1">
      <PageHeader eyebrow="§ About" title="How Ghostline works">
        A physics lab in your phone, with an AI lab partner that shows its work. The phone is only the camera and the screen: all the computing happens on Modal and
        the Meta Model API.
      </PageHeader>

      <div className="mx-auto grid w-full max-w-6xl gap-10 px-4 pt-10 lg:grid-cols-[11rem_minmax(0,1fr)] lg:gap-14 lg:pt-14">
        <nav aria-label="On this page" className="hidden lg:block">
          <ol className="sticky top-24 space-y-1 border-l border-line">
            {SECTIONS.map((s, i) => (
              <li key={s.id}>
                <a href={`#${s.id}`} className="-ml-px flex min-h-9 items-center gap-2.5 border-l border-transparent pl-4 text-sm text-muted hover:border-ghost hover:text-foreground">
                  <span className="font-mono text-xs">{String(i + 1).padStart(2, "0")}</span>
                  {s.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <div className="min-w-0 max-w-4xl space-y-16">
          <section aria-labelledby="arch">
            <SectionHeading id="arch" n="01">
              The pipeline
            </SectionHeading>
            <figure>
              <Architecture />
              <figcaption className="mt-3 flex gap-3 text-sm text-muted">
                <span className="eyebrow shrink-0 pt-px text-fg-2">Fig. 1</span>
                <span>From the phone to the API on Modal, which calls the Meta Model API, a sandboxed fitting harness and the reviewer.</span>
              </figcaption>
            </figure>
          </section>

          <section aria-labelledby="models">
            <SectionHeading id="models" n="02">
              Three Meta models, three jobs
            </SectionHeading>
            <ul className="grid gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-3">
              {MODELS.map((m) => (
                <li key={m.name} className="bg-surface p-5">
                  <p className={`eyebrow ${ROLE_COLOR[m.name] ?? ""}`}>{m.role}</p>
                  <h3 className="mt-2 font-display text-[1.65rem] leading-tight">{m.name}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-fg-2">{m.body}</p>
                </li>
              ))}
            </ul>
            <blockquote className="mt-6 border-l-2 border-ghost pl-5 text-[0.95rem] leading-relaxed text-fg-2">
              <strong className="font-semibold text-foreground">Wonder writes physics; a trusted harness keeps score.</strong> Fitting, held-out scoring and every number on
              screen come from deterministic code that Wonder cannot change, and Wonder never sees the frames it is tested on. A law wins by predicting the future of the
              motion, not by fitting the past.
            </blockquote>
          </section>

          <section aria-labelledby="limits">
            <SectionHeading id="limits" n="03">
              Limitations
            </SectionHeading>
            <ol className="grid gap-x-10 md:grid-cols-2">
              {LIMITS.map((l, i) => (
                <li key={l} className="flex gap-4 border-t border-line py-4 text-sm leading-relaxed text-fg-2">
                  <span className="font-mono text-xs text-muted">{String(i + 1).padStart(2, "0")}</span>
                  <span>{l}</span>
                </li>
              ))}
            </ol>
          </section>

          <section aria-labelledby="privacy">
            <SectionHeading id="privacy" n="04">
              Privacy
            </SectionHeading>
            <ul className="space-y-3 rounded-lg border border-line bg-surface p-5 text-sm leading-relaxed text-fg-2">
              {PRIVACY.map((p) => (
                <li key={p} className="flex gap-3">
                  <span aria-hidden className="mt-2 size-1.5 shrink-0 bg-ghost" />
                  <span>{p}</span>
                </li>
              ))}
            </ul>
          </section>
        </div>
      </div>
    </main>
  );
}

function SectionHeading({ id, n, children }: { id: string; n: string; children: React.ReactNode }) {
  return (
    <div className="mb-5">
      <p className="eyebrow">§ {n}</p>
      <h2 id={id} className="mt-1.5 font-display text-4xl leading-tight">
        {children}
      </h2>
    </div>
  );
}

/** Boxes and arrows, drawn inline so it works offline and in both themes. */
function Architecture() {
  const box = (x: number, y: number, w: number, h: number, title: string, sub: string, accent = false) => (
    <g key={title}>
      <rect x={x} y={y} width={w} height={h} rx={5} fill={accent ? "rgb(34 211 238 / 0.08)" : "var(--color-surface-2)"} stroke={accent ? "var(--color-ghost)" : "var(--color-line-strong)"} />
      <text x={x + w / 2} y={y + h / 2 - 3} textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--color-foreground)">
        {title}
      </text>
      <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize="10.5" fill="var(--color-muted)">
        {sub}
      </text>
    </g>
  );
  const arrow = (x1: number, y1: number, x2: number, y2: number, key: string) => (
    <line key={key} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--color-muted)" strokeOpacity={0.55} strokeWidth={1.25} markerEnd="url(#arr)" />
  );
  return (
    <div className="graph-paper overflow-x-auto rounded-lg border border-line bg-surface p-3 sm:p-5">
      <svg viewBox="0 0 760 330" className="mx-auto h-auto w-full min-w-[620px] max-w-3xl" role="img" aria-label="Phone to web app to API on Modal, which calls the Meta Model API, a sandbox and the reviewer, and stores to Postgres and a volume">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 10 5 0 10z" fill="var(--color-muted)" />
          </marker>
        </defs>
        {/* Arrows first, so the boxes sit on top of any line that crosses them. */}
        {arrow(160, 48, 198, 48, "a1")}
        {arrow(360, 48, 398, 48, "a2")}
        {arrow(570, 48, 598, 48, "a3")}
        {arrow(675, 76, 400, 150, "b1")}
        {arrow(675, 76, 540, 150, "b2")}
        {arrow(675, 76, 190, 150, "b3")}
        {arrow(485, 76, 175, 250, "c1")}
        {arrow(485, 76, 375, 250, "c2")}
        {arrow(485, 76, 580, 250, "c3")}
        {box(10, 20, 150, 56, "Phone or laptop", "camera and screen")}
        {box(200, 20, 160, 56, "Next.js on Vercel", "chunked upload, polling")}
        {box(400, 20, 170, 56, "FastAPI on Modal", "API and job queue", true)}
        {box(600, 20, 150, 56, "Pipeline workers", "ffmpeg, PyAV, OpenCV")}
        {box(80, 150, 190, 60, "Muse Wonder", "scene, hypotheses, conclusion")}
        {box(290, 150, 170, 60, "SAM 3.1", "tracks the object and the sheet")}
        {box(480, 150, 200, 60, "Modal Sandbox", "fitting harness, no network")}
        {box(80, 250, 190, 56, "Muse Spark Open", "the skeptical referee")}
        {box(290, 250, 170, 56, "Postgres", "experiments, events, fits")}
        {box(480, 250, 200, 56, "Modal Volume", "videos, plots, tracks")}
      </svg>
    </div>
  );
}
