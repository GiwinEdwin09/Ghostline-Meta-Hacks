import { ApiStatus } from "@/components/ApiStatus";
import { HeroVideo } from "@/components/HeroVideo";
import { ArrowLink, ButtonLink } from "@/components/ui";

const STEPS = [
  {
    n: "01",
    title: "Film it",
    body: "Put an A4 sheet in the frame and record the motion with your phone: a swing, a drop, a spinner slowing down.",
  },
  {
    n: "02",
    title: "Let the AI physicist work",
    body: "Muse Wonder proposes competing laws and says in advance what would refute each. SAM 3.1 tracks the object. A harness scores every law on frames it never saw.",
  },
  {
    n: "03",
    title: "Prove it with a ghost",
    body: "Film a second trial. A translucent ghost shows where the law says the object will go, and Ghostline measures how far reality drifts.",
  },
];

export default function Home() {
  return (
    <main className="flex-1">
      <section className="relative overflow-hidden border-b border-line">
        <div aria-hidden className="graph-paper absolute inset-0 [mask-image:radial-gradient(ellipse_at_70%_40%,black_20%,transparent_75%)]" />
        <div className="relative mx-auto grid w-full max-w-6xl gap-12 px-4 pb-14 pt-12 lg:grid-cols-[1.05fr_1fr] lg:items-center lg:gap-14 lg:pb-20 lg:pt-20">
          <div>
            <p className="eyebrow">Film · Measure · Predict</p>
            <h1 className="mt-4 font-display text-[3.1rem] leading-[0.98] tracking-[-0.015em] sm:text-7xl">
              Film anything that moves. <em className="text-ghost">Ghostline finds the law.</em>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-fg-2">
              An AI physicist measures your video, proposes the laws that could explain it, rejects the ones the data disagrees with, and proves the
              winner with a ghost that predicts where the object goes next.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <ButtonLink href="/examples" tone="primary" className="min-h-12 px-6 text-base">
                Try an example
              </ButtonLink>
              <ButtonLink href="/new" className="min-h-12 px-6 text-base">
                Run your own
              </ButtonLink>
            </div>
            <p className="mt-4 text-sm text-muted">No account needed to look around. Starting your own analysis needs a passcode.</p>
          </div>
          <figure className="min-w-0">
            <HeroVideo src="/hero-ghost.mp4" poster="/hero-poster.jpg" />
            <figcaption className="mt-4 flex gap-3 text-sm text-muted">
              <span className="eyebrow shrink-0 pt-px text-fg-2">Fig. 1</span>
              <span>A real pendulum. The cyan ghost is where the fitted law says the bob will be, drawn before it gets there.</span>
            </figcaption>
          </figure>
        </div>
      </section>

      <section className="mx-auto w-full max-w-6xl px-4 pt-16 sm:pt-20" aria-labelledby="how">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">§ Method</p>
            <h2 id="how" className="mt-2 font-display text-4xl leading-tight sm:text-5xl">
              How it works
            </h2>
          </div>
          <ArrowLink href="/about">The full pipeline</ArrowLink>
        </div>

        <Trajectory />

        <ol className="mt-8 grid gap-10 md:mt-14 md:grid-cols-3 md:gap-8">
          {STEPS.map((s, i) => (
            <li key={s.n} className="relative border-l border-line pl-5 md:border-l-0 md:pl-0">
              <span
                aria-hidden
                className={
                  i === STEPS.length - 1
                    ? "absolute -left-[5px] top-1.5 size-[9px] rounded-full border border-ghost bg-background md:hidden"
                    : "absolute -left-[4px] top-1.5 size-[7px] rounded-full bg-foreground md:hidden"
                }
              />
              <p className="font-mono text-sm text-muted">{s.n}</p>
              <h3 className="mt-1 text-lg font-semibold tracking-tight">{s.title}</h3>
              <p className="mt-2 text-[0.95rem] leading-relaxed text-fg-2">{s.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mx-auto w-full max-w-6xl px-4 pt-16 sm:pt-24">
        <div className="grid items-center gap-8 rounded-lg border border-line bg-surface p-6 sm:p-8 lg:grid-cols-[1.1fr_1fr] lg:gap-12">
          <div>
            <p className="eyebrow">§ Honesty</p>
            <h2 className="mt-2 font-display text-3xl leading-tight sm:text-4xl">Does it measure real physics correctly?</h2>
            <p className="mt-3 max-w-xl leading-relaxed text-fg-2">
              Every claim carries an honest uncertainty. The benchmark checks it against ground truth from a tape measure, a stopwatch and sound, and counts how
              often the 95% interval really contains the truth.
            </p>
            <div className="mt-5">
              <ButtonLink href="/benchmark">See the benchmark</ButtonLink>
            </div>
          </div>
          <figure>
            <CoverageSketch />
            <figcaption className="mt-3 flex gap-3 text-sm text-muted">
              <span className="eyebrow shrink-0 pt-px text-fg-2">Fig. 2</span>
              <span>How a measurement is scored: its 95% bar should cross the true value. An illustration, not real results.</span>
            </figcaption>
          </figure>
        </div>
        <div className="mt-6">
          <ApiStatus />
        </div>
      </section>
    </main>
  );
}

/**
 * The three steps drawn as a throw: two measured positions, then the dashed line the law
 * predicts and the ghost at the end of it. Desktop only; on a phone the steps get a rail.
 */
function Trajectory() {
  // y = 22 + 0.005 (x - 150)², launched from the floor at x = 25.1: it passes through the three
  // column centers (x = 50, 150, 250 of 300) at y = 72, 22 and 72.
  const dots = [
    { left: "16.667%", top: "72%" },
    { left: "50%", top: "22%" },
    { left: "83.333%", top: "72%" },
  ];
  return (
    <div aria-hidden className="relative mt-10 hidden h-24 md:block">
      <svg viewBox="0 0 300 100" preserveAspectRatio="none" className="absolute inset-0 size-full overflow-visible">
        <line x1="0" y1="99.5" x2="300" y2="99.5" stroke="var(--color-line)" vectorEffect="non-scaling-stroke" />
        {[50, 100, 150, 200, 250].map((x) => (
          <line key={x} x1={x} y1="96" x2={x} y2="100" stroke="var(--color-line-strong)" vectorEffect="non-scaling-stroke" />
        ))}
        {[
          [50, 72],
          [150, 22],
          [250, 72],
        ].map(([x, y], i) => (
          <line
            key={x}
            x1={x}
            y1={y}
            x2={x}
            y2="100"
            stroke={i === 2 ? "var(--color-ghost)" : "var(--color-muted)"}
            strokeOpacity={i === 2 ? 0.5 : 0.35}
            strokeDasharray="2 3"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        <path d="M25.1 100 Q87.55 22 150 22" fill="none" stroke="var(--color-foreground)" strokeOpacity="0.35" strokeWidth="1.25" vectorEffect="non-scaling-stroke" />
        <path
          d="M150 22 Q212.45 22 274.9 100"
          fill="none"
          stroke="var(--color-ghost)"
          strokeWidth="1.5"
          strokeDasharray="5 5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      {dots.map((d, i) => (
        <span
          key={i}
          className={
            i === 2
              ? "absolute size-4 rounded-full border-[1.5px] border-ghost bg-ghost/20 shadow-[0_0_24px_rgb(34_211_238/0.55)]"
              : "absolute size-3 rounded-full bg-foreground"
          }
          style={{ left: d.left, top: d.top, transform: "translate(-50%, -50%)", animation: `strobe-in 500ms ${150 + i * 260}ms both` }}
        />
      ))}
      {dots.map((d, i) => (
        <span
          key={`t${i}`}
          className={`absolute top-full mt-2 -translate-x-1/2 whitespace-nowrap font-mono text-xs ${i === 2 ? "text-ghost" : "text-muted"}`}
          style={{ left: d.left }}
        >
          t{["₁", "₂", "₃"][i]}
          {i === 2 ? " · predicted" : " · measured"}
        </span>
      ))}
    </div>
  );
}

/** Decorative: five made-up measurements with 95% bars around a true value. */
function CoverageSketch() {
  const rows = [
    { m: 0.42, lo: 0.3, hi: 0.55, ok: true },
    { m: 0.55, lo: 0.47, hi: 0.63, ok: true },
    { m: 0.48, lo: 0.4, hi: 0.57, ok: true },
    { m: 0.68, lo: 0.62, hi: 0.74, ok: false },
    { m: 0.52, lo: 0.36, hi: 0.66, ok: true },
  ];
  const W = 360;
  const truth = 0.5;
  const x = (v: number) => 20 + v * (W - 40);
  return (
    <svg viewBox={`0 0 ${W} 170`} className="h-auto w-full" role="img" aria-label="Illustration: five measurements with error bars, four of which cross the true value">
      <line x1={x(truth)} x2={x(truth)} y1={8} y2={150} stroke="var(--color-ghost)" strokeDasharray="4 4" />
      <text x={x(truth)} y={166} textAnchor="middle" fontSize="11" fontFamily="var(--font-mono)" fill="var(--color-ghost)">
        truth
      </text>
      {rows.map((r, i) => {
        const y = 22 + i * 28;
        const c = r.ok ? "#34d399" : "#f87171";
        return (
          <g key={i}>
            <line x1={x(r.lo)} x2={x(r.hi)} y1={y} y2={y} stroke={c} strokeWidth="1.75" />
            <line x1={x(r.lo)} x2={x(r.lo)} y1={y - 5} y2={y + 5} stroke={c} strokeWidth="1.75" />
            <line x1={x(r.hi)} x2={x(r.hi)} y1={y - 5} y2={y + 5} stroke={c} strokeWidth="1.75" />
            <circle cx={x(r.m)} cy={y} r="4" fill={c} />
          </g>
        );
      })}
    </svg>
  );
}
