"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";

import { cn } from "@/components/ui";
import { NAV_LINKS } from "@/lib/nav";

function isActive(pathname: string | null, href: string) {
  return pathname !== null && (pathname === href || pathname.startsWith(`${href}/`));
}

/*
 * Each piece comes in two parts: a reader that calls usePathname, and a view that takes the
 * pathname as a prop. The header renders the view with `null` as the Suspense fallback,
 * because usePathname suspends on /e/[id] while prerendering.
 */

export function MainNav() {
  return <MainNavView pathname={usePathname()} />;
}

/** Desktop navigation with the current page marked. */
export function MainNavView({ pathname }: { pathname: string | null }) {
  return (
    <nav aria-label="Main" className="ml-auto hidden items-center gap-1 sm:flex">
      {NAV_LINKS.map((l) => {
        const active = isActive(pathname, l.href);
        return (
          <Link
            key={l.href}
            href={l.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "relative inline-flex min-h-11 items-center rounded-md px-3 text-sm transition-colors hover:text-foreground",
              active ? "text-foreground" : "text-muted",
            )}
          >
            {l.label}
            {active ? <span aria-hidden className="absolute inset-x-3 bottom-1.5 h-px bg-ghost" /> : null}
          </Link>
        );
      })}
    </nav>
  );
}

export function MobileMenu() {
  return <MobileMenuView pathname={usePathname()} />;
}

/**
 * The phone menu. It closes when a link is followed, on Escape and on a tap outside it,
 * which a bare <details> does not do across client-side navigations.
 */
export function MobileMenuView({ pathname }: { pathname: string | null }) {
  const [open, setOpen] = useState(false);
  const [openedOn, setOpenedOn] = useState(pathname);
  const wrap = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const panelId = useId();

  // Arriving on another page closes the menu.
  if (open && openedOn !== pathname) {
    setOpen(false);
    setOpenedOn(pathname);
  }

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={wrap} className="relative sm:hidden">
      <button
        ref={button}
        type="button"
        aria-label={open ? "Close menu" : "Menu"}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => {
          setOpenedOn(pathname);
          setOpen((o) => !o);
        }}
        className="flex size-11 items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-foreground"
      >
        <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          {open ? <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" /> : <path d="M4 8h16M4 16h16" strokeLinecap="round" />}
        </svg>
      </button>
      <div
        id={panelId}
        hidden={!open}
        className="absolute right-0 top-12 w-56 rounded-lg border border-line-strong bg-surface p-1.5 shadow-2xl shadow-black/60"
      >
        {NAV_LINKS.map((l) => {
          const active = isActive(pathname, l.href);
          return (
            <Link
              key={l.href}
              href={l.href}
              aria-current={active ? "page" : undefined}
              onClick={() => setOpen(false)}
              className={cn(
                "flex min-h-11 items-center justify-between rounded-md px-3 text-sm hover:bg-surface-2",
                active ? "text-foreground" : "text-fg-2",
              )}
            >
              {l.label}
              {active ? <span aria-hidden className="size-1.5 bg-ghost" /> : null}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
