"use client";

import Link from "next/link";

import { Card, ConfidenceBadge, ErrorState, Skeleton, SystemBadge } from "@/components/ui";
import { api, resolveMedia } from "@/lib/api";
import { useResource } from "@/lib/hooks";

export function ExamplesGrid() {
  const list = useResource("examples", () => api.listExamples());
  if (list.error && !list.data) {
    return (
      <div className="mt-6">
        <ErrorState error={list.error} onRetry={list.reload} />
      </div>
    );
  }
  if (!list.data) {
    return (
      <ul className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true" aria-label="Loading examples">
        {[0, 1, 2].map((i) => (
          <li key={i}>
            <Skeleton className="aspect-video w-full" />
            <Skeleton className="mt-3 h-5 w-2/3" />
            <Skeleton className="mt-2 h-4 w-full" />
          </li>
        ))}
      </ul>
    );
  }
  return (
    <ul className="mt-6 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
      {list.data.map((e) => (
        <li key={e.id}>
          <Link
            href={`/e/${e.id}`}
            className="group block h-full overflow-hidden rounded-xl border border-line bg-surface transition-colors hover:border-ghost/50"
          >
            <div className="relative aspect-video bg-black">
              {e.thumbnail_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={resolveMedia(e.thumbnail_url)} alt="" loading="lazy" className="size-full object-cover opacity-90 transition-opacity group-hover:opacity-100" />
              ) : null}
            </div>
            <div className="space-y-2 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <SystemBadge system={e.system_type} />
                <ConfidenceBadge level={e.confidence} />
              </div>
              <h2 className="text-lg font-semibold leading-snug">{e.title}</h2>
              {e.headline ? <p className="text-sm font-medium text-ghost">{e.headline}</p> : null}
              {e.blurb ? <p className="text-sm text-muted">{e.blurb}</p> : null}
            </div>
          </Link>
        </li>
      ))}
      <li className="sm:col-span-2 lg:col-span-1">
        <Card className="flex h-full flex-col items-start justify-center gap-2">
          <p className="font-medium">Have a clip of your own?</p>
          <p className="text-sm text-muted">Film a swing, a drop or a spinner and let Ghostline measure it.</p>
          <Link href="/new" className="mt-1 inline-flex min-h-11 items-center text-sm font-medium text-ghost">
            Run your own →
          </Link>
        </Card>
      </li>
    </ul>
  );
}
