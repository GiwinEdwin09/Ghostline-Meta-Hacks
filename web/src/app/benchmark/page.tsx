import type { Metadata } from "next";

import { BenchmarkView } from "@/components/BenchmarkView";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Benchmark" };

export default function BenchmarkPage() {
  return (
    <main className="flex-1">
      <PageHeader eyebrow="§ Validation · ground truth" title="Benchmark">
        Ghostline is checked against ground truth it cannot see: local gravity from latitude, a tape measure and stopwatch for the pendulum, the tilt at which a
        block slides at constant speed, and the click timing of a bouncing ball. The score that matters is not how close it gets but how often its 95% interval
        contains the truth.
      </PageHeader>
      <div className="mx-auto w-full max-w-6xl px-4 pt-8 sm:pt-10">
        <BenchmarkView />
      </div>
    </main>
  );
}
