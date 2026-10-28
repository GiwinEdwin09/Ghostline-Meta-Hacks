import type { Metadata } from "next";

import { Card } from "@/components/ui";

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

export default function AboutPage() {
  return (
    <main className="mx-auto w-full max-w-4xl space-y-10 px-4 py-8 sm:py-12">
      <header>
        <h1 className="text-3xl font-semibold tracking-tight">How Ghostline works</h1>
        <p className="mt-2 text-muted">
          A physics lab in your phone, with an AI lab partner that shows its work. The phone is only the camera and the screen: all the computing happens on Modal and the
          Meta Model API.
        </p>
      </header>

      <section aria-labelledby="arch">
        <h2 id="arch" className="mb-3 text-xl font-semibold tracking-tight">
          The pipeline
        </h2>
        <Architecture />
      </section>

      <section aria-labelledby="models">
        <h2 id="models" className="mb-3 text-xl font-semibold tracking-tight">
          Three Meta models, three jobs
        </h2>
        <ul className="grid gap-4 md:grid-cols-3">
          {MODELS.map((m) => (
            <li key={m.name}>
              <Card className="h-full">
                <p className="text-xs font-medium uppercase tracking-wide text-ghost">{m.role}</p>
                <h3 className="mt-1 font-semibold">{m.name}</h3>
                <p className="mt-2 text-sm text-zinc-300">{m.body}</p>
              </Card>
            </li>
          ))}
        </ul>
        <p className="mt-4 text-sm text-zinc-300">
          <strong className="font-semibold text-foreground">Wonder writes physics; a trusted harness keeps score.</strong> Fitting, held-out scoring and every number on screen come
          from deterministic code that Wonder cannot change, and Wonder never sees the frames it is tested on. A law wins by predicting the future of the motion, not by fitting the past.
        </p>
      </section>

      <section aria-labelledby="limits">
        <h2 id="limits" className="mb-3 text-xl font-semibold tracking-tight">
          Limitations
        </h2>
        <ul className="list-disc space-y-2 pl-5 text-sm text-zinc-300">
          {LIMITS.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="privacy">
        <h2 id="privacy" className="mb-3 text-xl font-semibold tracking-tight">
          Privacy
        </h2>
        <Card>
          <ul className="list-disc space-y-2 pl-5 text-sm text-zinc-300">
            <li>Experiments are private by default, behind unguessable links. Only the examples are public.</li>
            <li>Your clip is processed by the Meta Model API in Standard mode, so it is not used to train Meta&apos;s models, and stored on Modal.</li>
            <li>Copies sent to Meta expire after 7 days. Clips that are not examples are deleted from Ghostline after 14 days, and you can delete an experiment at any time.</li>
            <li>Text that appears in a video cannot change a result: the harness computes every number and checks every claim.</li>
          </ul>
        </Card>
      </section>
    </main>
  );
}

/** Boxes and arrows, drawn inline so it works offline and in both themes. */
function Architecture() {
  const box = (x: number, y: number, w: number, h: number, title: string, sub: string, accent = false) => (
    <g key={title}>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={accent ? "rgba(34,211,238,0.1)" : "#0e1319"} stroke={accent ? "#22d3ee" : "#34414f"} />
      <text x={x + w / 2} y={y + h / 2 - 3} textAnchor="middle" fontSize="13" fontWeight="600" fill="#e7ecf2">
        {title}
      </text>
      <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize="10.5" fill="#9aa9ba">
        {sub}
      </text>
    </g>
  );
  const arrow = (x1: number, y1: number, x2: number, y2: number, key: string) => (
    <line key={key} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#5b6b7d" strokeWidth={1.5} markerEnd="url(#arr)" />
  );
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface p-3">
      <svg viewBox="0 0 760 330" className="mx-auto h-auto w-full min-w-[620px] max-w-3xl" role="img" aria-label="Phone to web app to API on Modal, which calls the Meta Model API, a sandbox and the reviewer, and stores to Postgres and a volume">
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 10 5 0 10z" fill="#5b6b7d" />
          </marker>
        </defs>
        {box(10, 20, 150, 56, "Phone or laptop", "camera and screen")}
        {box(200, 20, 160, 56, "Next.js on Vercel", "chunked upload, polling")}
        {box(400, 20, 170, 56, "FastAPI on Modal", "API and job queue", true)}
        {box(600, 20, 150, 56, "Pipeline workers", "ffmpeg, PyAV, OpenCV")}
        {arrow(160, 48, 198, 48, "a1")}
        {arrow(360, 48, 398, 48, "a2")}
        {arrow(570, 48, 598, 48, "a3")}
        {box(80, 150, 190, 60, "Muse Wonder", "scene, hypotheses, conclusion")}
        {box(290, 150, 170, 60, "SAM 3.1", "tracks the object and the sheet")}
        {box(480, 150, 200, 60, "Modal Sandbox", "fitting harness, no network")}
        {box(80, 250, 190, 56, "Muse Spark Open", "the skeptical referee")}
        {box(290, 250, 170, 56, "Postgres", "experiments, events, fits")}
        {box(480, 250, 200, 56, "Modal Volume", "videos, plots, tracks")}
        {arrow(675, 76, 400, 150, "b1")}
        {arrow(675, 76, 540, 150, "b2")}
        {arrow(675, 76, 190, 150, "b3")}
        {arrow(485, 76, 175, 250, "c1")}
        {arrow(485, 76, 375, 250, "c2")}
        {arrow(485, 76, 580, 250, "c3")}
      </svg>
    </div>
  );
}
