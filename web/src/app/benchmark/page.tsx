import type { Metadata } from "next";

import { BenchmarkView } from "@/components/BenchmarkView";

export const metadata: Metadata = { title: "Benchmark" };

export default function BenchmarkPage() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:py-12">
      <h1 className="text-3xl font-semibold tracking-tight">Benchmark</h1>
      <p className="mb-8 mt-2 max-w-2xl text-muted">
        Ghostline is checked against ground truth it cannot see: local gravity from latitude, a tape measure and stopwatch for the pendulum, the tilt at which a
        block slides at constant speed, and the click timing of a bouncing ball. The score that matters is not how close it gets but how often its 95% interval
        contains the truth.
      </p>
      <BenchmarkView />
    </main>
  );
}
