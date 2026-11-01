"use client";

import Link from "next/link";

import { ArrowLink, ConfidenceBadge, ErrorState, Skeleton, SystemBadge } from "@/components/ui";
import { api, resolveMedia } from "@/lib/api";
import { useResource } from "@/lib/hooks";

export function ExamplesGrid() {
  const list = useResource("examples", () => api.listExamples());
  if (list.error && !list.data) {
    return <ErrorState error={list.error} onRetry={list.reload} />;
  }
  if (!list.data) {
    return (
      <ul className="grid gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-3" aria-busy="true" aria-label="Loading examples">
        {[0, 1, 2].map((i) => (
          <li key={i}>
            <Skeleton className="aspect-video w-full" />
            <Skeleton className="mt-4 h-5 w-2/3" />
            <Skeleton className="mt-2 h-4 w-full" />
          </li>
        ))}
      </ul>
    );
  }
  return (
    <ul className="grid gap-x-6 gap-y-10 sm:grid-cols-2 lg:grid-cols-3">
      {list.data.map((e, i) => (
        <li key={e.id}>
          <Link href={`/e/${e.id}`} className="group block h-full rounded-lg focus-visible:outline-offset-4">
            <div className="relative aspect-video overflow-hidden rounded-md border border-line bg-black transition-colors group-hover:border-ghost/60">
              {e.thumbnail_url ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={resolveMedia(e.thumbnail_url)}
                  alt=""
                  loading="lazy"
                  className="size-full object-cover opacity-85 transition-[opacity,transform] duration-300 group-hover:scale-[1.02] group-hover:opacity-100"
                />
              ) : null}
              <span className="absolute left-2.5 top-2.5 rounded-[3px] bg-black/65 px-1.5 py-1 font-mono text-[0.625rem] uppercase leading-none tracking-[0.08em] text-white/85 backdrop-blur">
                Exp. {String(i + 1).padStart(2, "0")}
              </span>
            </div>
            <div className="pt-4">
              <div className="flex flex-wrap items-center gap-2">
                <SystemBadge system={e.system_type} />
                <ConfidenceBadge level={e.confidence} />
              </div>
              <h2 className="mt-3 text-lg font-semibold leading-snug tracking-tight group-hover:text-white">{e.title}</h2>
              {e.headline ? <p className="mt-1 font-mono text-sm text-ghost">{e.headline}</p> : null}
              {e.blurb ? <p className="mt-2 text-sm leading-relaxed text-muted">{e.blurb}</p> : null}
            </div>
          </Link>
        </li>
      ))}
      <li>
        <div className="flex h-full min-h-56 flex-col justify-center rounded-md border border-dashed border-line-strong p-6">
          <p className="eyebrow">Your turn</p>
          <p className="mt-2 font-display text-[1.75rem] leading-tight">Have a clip of your own?</p>
          <p className="mt-2 text-sm text-muted">Film a swing, a drop or a spinner and let Ghostline measure it.</p>
          <ArrowLink href="/new" className="mt-2 self-start">
            Run your own
          </ArrowLink>
        </div>
      </li>
    </ul>
  );
}
