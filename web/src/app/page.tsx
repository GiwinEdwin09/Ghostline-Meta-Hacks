import Link from "next/link";

import { ApiStatus } from "@/components/ApiStatus";
import { HeroVideo } from "@/components/HeroVideo";
import { ButtonLink } from "@/components/ui";

const STEPS = [
  {
    n: "1",
    title: "Film it",
    body: "Put an A4 sheet in the frame and record the motion with your phone: a swing, a drop, a spinner slowing down.",
  },
  {
    n: "2",
    title: "Let the AI physicist work",
    body: "Muse Wonder proposes competing laws and says in advance what would refute each. SAM 3.1 tracks the object. A harness scores every law on frames it never saw.",
  },
  {
    n: "3",
    title: "Prove it with a ghost",
    body: "Film a second trial. A translucent ghost shows where the law says the object will go, and Ghostline measures how far reality drifts.",
  },
];

export default function Home() {
  return (
    <main className="flex-1">
      <section className="mx-auto grid w-full max-w-6xl gap-8 px-4 pb-10 pt-8 lg:grid-cols-2 lg:items-center lg:gap-12 lg:pt-16">
        <div>
          <h1 className="text-4xl font-semibold leading-[1.1] tracking-tight sm:text-5xl">
            Film anything that moves. Ghostline finds the law.
          </h1>
          <p className="mt-4 max-w-xl text-lg text-muted">
            An AI physicist measures your video, proposes the laws that could explain it, rejects the ones the data disagrees with, and proves the
            winner with a ghost that predicts where the object goes next.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <ButtonLink href="/examples" tone="primary" className="min-h-12 px-6 text-base">
              Try an example
            </ButtonLink>
            <ButtonLink href="/new" className="min-h-12 px-6 text-base">
              Run your own
            </ButtonLink>
          </div>
          <p className="mt-4 text-sm text-muted">
            No account needed to look around. Starting your own analysis needs a passcode.
          </p>
        </div>
        <HeroVideo src="/hero-ghost.mp4" poster="/hero-poster.jpg" />
      </section>

      <section className="mx-auto w-full max-w-6xl px-4 py-8" aria-labelledby="how">
        <h2 id="how" className="text-xl font-semibold tracking-tight">
          How it works
        </h2>
        <ol className="mt-4 grid gap-4 md:grid-cols-3">
          {STEPS.map((s) => (
            <li key={s.n} className="rounded-xl border border-line bg-surface p-5">
              <span className="flex size-8 items-center justify-center rounded-full bg-ghost/15 text-sm font-semibold text-ghost">{s.n}</span>
              <h3 className="mt-3 font-semibold">{s.title}</h3>
              <p className="mt-1 text-sm text-zinc-300">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mx-auto w-full max-w-6xl px-4 py-8">
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-line bg-surface p-5 sm:p-6">
          <div className="max-w-xl">
            <h2 className="text-lg font-semibold">Does it measure real physics correctly?</h2>
            <p className="mt-1 text-sm text-zinc-300">
              Every claim carries an honest uncertainty. The benchmark checks it against ground truth from a tape measure, a stopwatch and sound.
            </p>
          </div>
          <Link
            href="/benchmark"
            className="inline-flex min-h-11 items-center rounded-lg border border-line-strong px-4 text-sm font-medium hover:bg-surface-2"
          >
            See the benchmark
          </Link>
        </div>
        <div className="mt-6">
          <ApiStatus />
        </div>
      </section>
    </main>
  );
}
