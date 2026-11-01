import type { Metadata } from "next";
import { Suspense } from "react";

import { Workspace } from "@/components/experiment/Workspace";
import { Skeleton } from "@/components/ui";

export const metadata: Metadata = { title: "Experiment" };

function Fallback() {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 space-y-6 px-4 py-6 sm:py-10">
      <Skeleton className="h-9 w-3/4" />
      <Skeleton className="h-6 w-1/2" />
      <Skeleton className="h-64 w-full" />
    </main>
  );
}

// The id is only known at request time (experiments are UUIDs), so it is read inside Suspense and
// the static shell above renders instantly. The workspace itself loads and polls on the client.
export default function ExperimentPage({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Fallback />}>
      {params.then(({ id }) => (
        <Suspense fallback={<Fallback />}>
          <Workspace key={id} id={id} />
        </Suspense>
      ))}
    </Suspense>
  );
}
