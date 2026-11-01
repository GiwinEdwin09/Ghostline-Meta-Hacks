"use client";

import Link from "next/link";
import { useEffect, useRef, type ButtonHTMLAttributes, type ReactNode } from "react";

import { CONFIDENCE_LABELS, REVIEW_LABELS, SYSTEM_LABELS } from "@/lib/format";
import { describeError } from "@/lib/errors";
import type { Confidence, ReviewStatus, SystemType } from "@/lib/types";

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/* ---------------------------------- buttons ---------------------------------- */

type ButtonTone = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_BASE =
  "inline-flex min-h-11 select-none items-center justify-center gap-2 rounded-md px-4 text-sm font-medium transition-[background-color,border-color,color,transform] duration-150 active:translate-y-px disabled:cursor-not-allowed disabled:opacity-45 disabled:active:translate-y-0";

const BUTTON_TONES: Record<ButtonTone, string> = {
  primary: "bg-foreground text-background hover:bg-white disabled:hover:bg-foreground",
  secondary: "border border-line-strong bg-surface text-foreground hover:border-muted/50 hover:bg-surface-2 disabled:hover:border-line-strong disabled:hover:bg-surface",
  ghost: "text-muted hover:bg-surface-2 hover:text-foreground disabled:hover:bg-transparent disabled:hover:text-muted",
  danger: "border border-red-400/40 bg-red-400/10 text-red-200 hover:bg-red-400/20",
};

export function buttonClass(tone: ButtonTone = "secondary", extra?: string): string {
  return cn(BUTTON_BASE, BUTTON_TONES[tone], extra);
}

export function Button({
  tone = "secondary",
  className,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: ButtonTone }) {
  return <button type={type} className={buttonClass(tone, className)} {...props} />;
}

export function ButtonLink({
  href,
  tone = "secondary",
  className,
  children,
  prefetch,
}: {
  href: string;
  tone?: ButtonTone;
  className?: string;
  children: ReactNode;
  prefetch?: boolean;
}) {
  return (
    <Link href={href} prefetch={prefetch} className={buttonClass(tone, className)}>
      {children}
    </Link>
  );
}

/** A text link with an arrow, for "see more" style actions. */
export function ArrowLink({ href, children, className }: { href: string; children: ReactNode; className?: string }) {
  return (
    <Link
      href={href}
      className={cn("group inline-flex min-h-11 items-center gap-1.5 text-sm font-medium text-ghost hover:text-cyan-200", className)}
    >
      {children}
      <span aria-hidden className="transition-transform duration-150 group-hover:translate-x-0.5">
        →
      </span>
    </Link>
  );
}

/** Pills, chips and toggles that sit in a row, for example overlays or playback speed. */
export function chipClass(selected: boolean, extra?: string): string {
  return cn(
    "inline-flex min-h-11 cursor-pointer select-none items-center gap-2 rounded-md border px-3.5 text-sm transition-colors",
    selected ? "border-ghost/60 bg-ghost/10 text-cyan-100" : "border-line text-muted hover:border-line-strong hover:text-foreground",
    extra,
  );
}

/** Text inputs and text areas. */
export const inputClass =
  "rounded-md border border-line-strong bg-background px-3 text-base text-foreground transition-colors placeholder:text-muted/70 hover:border-muted/50 focus-visible:border-ghost aria-[invalid=true]:border-red-400/70";

/* ----------------------------------- badges ----------------------------------- */

export type BadgeTone = "neutral" | "cyan" | "green" | "amber" | "red" | "violet";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "border-line-strong text-fg-2",
  cyan: "border-ghost/35 bg-ghost/[0.07] text-cyan-200",
  green: "border-emerald-400/35 bg-emerald-400/[0.07] text-emerald-300",
  amber: "border-amber-400/35 bg-amber-400/[0.07] text-amber-300",
  red: "border-red-400/35 bg-red-400/[0.07] text-red-300",
  violet: "border-violet-400/35 bg-violet-400/[0.07] text-violet-300",
};

export function Badge({ tone = "neutral", children, className }: { tone?: BadgeTone; children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-[4px] border px-2 py-1 font-mono text-[0.6875rem] font-medium uppercase leading-none tracking-[0.06em]",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function ConfidenceBadge({ level }: { level: Confidence | null | undefined }) {
  if (!level) return null;
  const tone: BadgeTone = level === "high" ? "green" : level === "medium" ? "amber" : "red";
  return (
    <Badge tone={tone}>
      <span aria-hidden className="size-1.5 rounded-full bg-current" />
      {CONFIDENCE_LABELS[level]}
    </Badge>
  );
}

export function ReviewBadge({ status }: { status: ReviewStatus | null | undefined }) {
  if (!status) return null;
  const tone: BadgeTone =
    status === "accepted" ? "green" : status === "accepted_with_caveats" ? "amber" : status === "contested" ? "red" : "neutral";
  return <Badge tone={tone}>{REVIEW_LABELS[status]}</Badge>;
}

export function SystemBadge({ system }: { system: SystemType }) {
  return <Badge tone="cyan">{SYSTEM_LABELS[system]}</Badge>;
}

/* ------------------------------ layout pieces --------------------------------- */

export function Card({ className, children, as: Tag = "div" }: { className?: string; children: ReactNode; as?: "div" | "section" | "article" }) {
  return <Tag className={cn("rounded-lg border border-line bg-surface p-4 sm:p-5", className)}>{children}</Tag>;
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
      <h2 className="flex items-center gap-2 text-[0.95rem] font-semibold tracking-tight text-foreground">
        <span aria-hidden className="size-1.5 shrink-0 bg-ghost" />
        {children}
      </h2>
      {hint ? <span className="text-xs text-muted">{hint}</span> : null}
    </div>
  );
}

/**
 * The band at the top of a page: graph paper, a mono eyebrow, a serif title and a lede.
 * It is full width, so pages render it outside their content column.
 */
export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="relative border-b border-line">
      <div aria-hidden className="graph-paper absolute inset-0 [mask-image:linear-gradient(to_bottom,black,transparent)]" />
      <div className="relative mx-auto w-full max-w-6xl px-4 pb-8 pt-10 sm:pb-10 sm:pt-14">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="mt-3 font-display text-[2.6rem] leading-[1.02] tracking-[-0.01em] sm:text-6xl">{title}</h1>
        {children ? <div className="mt-4 max-w-2xl text-base leading-relaxed text-fg-2 sm:text-lg">{children}</div> : null}
      </div>
    </header>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("skeleton", className)} />;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" role="status">
      <svg className="size-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden>
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
        <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      {label}
    </span>
  );
}

export function ErrorState({
  error,
  onRetry,
  title,
}: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  const f = describeError(error);
  return (
    <div role="alert" className="rounded-r-md border-l-2 border-red-400 bg-red-400/[0.06] px-4 py-3.5">
      <p className="font-medium text-red-200">{title ?? f.title}</p>
      <p className="mt-1 text-sm text-fg-2">{f.hint}</p>
      {onRetry ? (
        <Button tone="secondary" className="mt-3" onClick={onRetry}>
          Retry
        </Button>
      ) : null}
    </div>
  );
}

export function Notice({
  tone = "amber",
  title,
  children,
}: {
  tone?: "amber" | "red" | "cyan";
  title?: string;
  children?: ReactNode;
}) {
  const styles = {
    amber: { box: "border-amber-400 bg-amber-400/[0.06]", title: "text-amber-200" },
    red: { box: "border-red-400 bg-red-400/[0.06]", title: "text-red-200" },
    cyan: { box: "border-ghost bg-ghost/[0.05]", title: "text-cyan-100" },
  }[tone];
  return (
    <div className={cn("rounded-r-md border-l-2 px-3.5 py-3 text-sm", styles.box)}>
      {title ? <p className={cn("font-medium", styles.title)}>{title}</p> : null}
      {children ? <div className={cn(title && "mt-0.5", "text-fg-2")}>{children}</div> : null}
    </div>
  );
}

/* ---------------------------------- dialog ------------------------------------ */

/** A modal built on the native <dialog>: focus trap, Escape and backdrop come for free. */
export function Dialog({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
      className="m-auto max-h-[calc(100dvh-2rem)] w-[min(40rem,calc(100vw-1.5rem))] overflow-y-auto rounded-lg border border-line-strong bg-surface p-0 text-foreground shadow-2xl shadow-black/60"
    >
      <div className="p-4 sm:p-6">
        <div className="mb-4 flex items-start justify-between gap-4 border-b border-line pb-3">
          <h2 className="font-display text-[1.75rem] leading-tight">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-m-2 inline-flex size-11 shrink-0 items-center justify-center rounded-md text-muted hover:bg-surface-2 hover:text-foreground"
          >
            <svg viewBox="0 0 24 24" className="size-5" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        {open ? children : null}
      </div>
    </dialog>
  );
}

/* --------------------------------- the mark ----------------------------------- */

/**
 * Ghostline's mark: three measured positions of a thrown ball (a stroboscope's exposures),
 * then the dashed line the fitted law predicts and the hollow ghost where it lands.
 */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden fill="none">
      <path d="M4 16.06Q8 5 12 5" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.2" />
      <path d="M12 5Q16 5 20 16.06" stroke="var(--color-ghost)" strokeWidth="1.4" strokeDasharray="1.6 1.9" strokeLinecap="round" />
      <circle cx="4" cy="16.06" r="2.1" fill="currentColor" />
      <circle cx="8" cy="7.77" r="1.75" fill="currentColor" />
      <circle cx="12" cy="5" r="1.75" fill="currentColor" />
      <circle cx="20" cy="16.06" r="2.3" stroke="var(--color-ghost)" strokeWidth="1.4" fill="rgb(34 211 238 / 0.18)" />
    </svg>
  );
}
