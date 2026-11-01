"use client";

import { useEffect, useRef, type KeyboardEvent, type ReactNode } from "react";

import { cn } from "@/components/ui";

export interface TabSpec {
  id: string;
  label: string;
  /** A small hint after the label, such as a count. */
  hint?: ReactNode;
  disabled?: boolean;
}

/** An accessible tab list that scrolls sideways on a phone instead of wrapping. */
export function Tabs({
  tabs,
  active,
  onChange,
  idPrefix,
}: {
  tabs: TabSpec[];
  active: string;
  onChange: (id: string) => void;
  idPrefix: string;
}) {
  const listRef = useRef<HTMLDivElement>(null);

  // On a phone the list scrolls sideways: center the selected tab, without scrolling the page.
  useEffect(() => {
    const list = listRef.current;
    const el = list?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (!list || !el || list.scrollWidth <= list.clientWidth) return;
    list.scrollTo({ left: el.offsetLeft - (list.clientWidth - el.offsetWidth) / 2, behavior: "smooth" });
  }, [active]);

  const enabled = tabs.filter((t) => !t.disabled);
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = enabled.findIndex((t) => t.id === active);
    let next = -1;
    if (e.key === "ArrowRight") next = (i + 1) % enabled.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + enabled.length) % enabled.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = enabled.length - 1;
    if (next >= 0) {
      e.preventDefault();
      onChange(enabled[next].id);
      listRef.current?.querySelector<HTMLElement>(`#${idPrefix}-tab-${enabled[next].id}`)?.focus();
    }
  };

  return (
    <div className="sticky top-14 z-20 -mx-4 border-b border-line bg-background/90 backdrop-blur-md supports-[backdrop-filter]:bg-background/75 sm:mx-0">
      <div
        ref={listRef}
        role="tablist"
        aria-label="Experiment sections"
        onKeyDown={onKey}
        className="scroll-hide relative flex gap-0.5 overflow-x-auto px-4 sm:px-0"
      >
        {tabs.map((t) => {
          const selected = t.id === active;
          return (
            <button
              key={t.id}
              id={`${idPrefix}-tab-${t.id}`}
              role="tab"
              type="button"
              aria-selected={selected}
              aria-controls={`${idPrefix}-panel-${t.id}`}
              tabIndex={selected ? 0 : -1}
              disabled={t.disabled}
              onClick={() => onChange(t.id)}
              className={cn(
                "relative inline-flex min-h-12 shrink-0 items-center gap-2 whitespace-nowrap px-3 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-35",
                selected ? "text-foreground" : "text-muted enabled:hover:text-foreground",
              )}
            >
              {t.label}
              {t.hint ? (
                <span className={cn("rounded-[3px] border px-1 py-px font-mono text-[0.6875rem] leading-none", selected ? "border-ghost/50 text-ghost" : "border-line-strong text-muted")}>
                  {t.hint}
                </span>
              ) : null}
              {selected ? <span aria-hidden className="absolute inset-x-3 -bottom-px h-0.5 bg-ghost" /> : null}
            </button>
          );
        })}
      </div>
      <div aria-hidden className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-background to-transparent sm:hidden" />
    </div>
  );
}
