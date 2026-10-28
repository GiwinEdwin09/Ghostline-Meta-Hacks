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

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>('[aria-selected="true"]');
    el?.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
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
    <div className="relative -mx-4 border-b border-line sm:mx-0">
      <div
        ref={listRef}
        role="tablist"
        aria-label="Experiment sections"
        onKeyDown={onKey}
        className="scroll-hide flex gap-1 overflow-x-auto px-4 sm:px-0"
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
                "relative inline-flex min-h-11 shrink-0 items-center gap-1.5 whitespace-nowrap px-3 text-sm font-medium transition-colors disabled:opacity-40",
                selected ? "text-ghost" : "text-muted hover:text-foreground",
              )}
            >
              {t.label}
              {t.hint ? <span className="text-xs text-muted">{t.hint}</span> : null}
              {selected ? <span aria-hidden className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-ghost" /> : null}
            </button>
          );
        })}
      </div>
      <div aria-hidden className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-gradient-to-l from-background to-transparent sm:hidden" />
    </div>
  );
}
