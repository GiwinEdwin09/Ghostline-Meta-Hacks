import type { Metadata } from "next";

import { ExamplesGrid } from "@/components/ExamplesGrid";

export const metadata: Metadata = { title: "Examples" };

export default function ExamplesPage() {
  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:py-12">
      <h1 className="text-3xl font-semibold tracking-tight">Examples</h1>
      <p className="mt-2 max-w-2xl text-muted">
        Finished experiments you can open right now: the hypotheses, the rejections, the reviewer&apos;s challenge, the ghost, and a what-if
        simulator for the law that won.
      </p>
      <ExamplesGrid />
    </main>
  );
}
