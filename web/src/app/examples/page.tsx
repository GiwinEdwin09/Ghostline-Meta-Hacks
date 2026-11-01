import type { Metadata } from "next";

import { ExamplesGrid } from "@/components/ExamplesGrid";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Examples" };

export default function ExamplesPage() {
  return (
    <main className="flex-1">
      <PageHeader eyebrow="§ Library · finished experiments" title="Examples">
        Finished experiments you can open right now: the hypotheses, the rejections, the reviewer&apos;s challenge, the ghost, and a what-if simulator for the
        law that won.
      </PageHeader>
      <div className="mx-auto w-full max-w-6xl px-4 pt-8 sm:pt-10">
        <ExamplesGrid />
      </div>
    </main>
  );
}
