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
  "inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-4 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50";

const BUTTON_TONES: Record<ButtonTone, string> = {
  primary: "bg-ghost text-zinc-950 hover:bg-cyan-300 active:bg-cyan-500",
  secondary: "border border-line-strong bg-surface-2 text-foreground hover:bg-zinc-800",
  ghost: "text-muted hover:bg-surface-2 hover:text-foreground",
  danger: "border border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/20",
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

/* ----------------------------------- badges ----------------------------------- */

export type BadgeTone = "neutral" | "cyan" | "green" | "amber" | "red" | "violet";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "border-line-strong bg-surface-2 text-zinc-300",
  cyan: "border-cyan-400/40 bg-cyan-400/10 text-cyan-300",
  green: "border-emerald-400/40 bg-emerald-400/10 text-emerald-300",
  amber: "border-amber-400/40 bg-amber-400/10 text-amber-300",
  red: "border-red-400/40 bg-red-400/10 text-red-300",
  violet: "border-violet-400/40 bg-violet-400/10 text-violet-300",
};

export function Badge({ tone = "neutral", children, className }: { tone?: BadgeTone; children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium whitespace-nowrap",
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
  return <Tag className={cn("rounded-xl border border-line bg-surface p-4 sm:p-5", className)}>{children}</Tag>;
}

export function SectionTitle({ children, hint }: { children: ReactNode; hint?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
      <h2 className="text-base font-semibold tracking-tight text-foreground">{children}</h2>
      {hint ? <span className="text-xs text-muted">{hint}</span> : null}
    </div>
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
    <div role="alert" className="rounded-xl border border-red-500/30 bg-red-500/5 p-4">
      <p className="font-medium text-red-200">{title ?? f.title}</p>
      <p className="mt-1 text-sm text-zinc-300">{f.hint}</p>
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
    amber: "border-amber-400/30 bg-amber-400/5 text-amber-100",
    red: "border-red-400/30 bg-red-400/5 text-red-100",
    cyan: "border-cyan-400/30 bg-cyan-400/5 text-cyan-100",
  }[tone];
  return (
    <div className={cn("rounded-lg border px-3 py-2.5 text-sm", styles)}>
      {title ? <p className="font-medium">{title}</p> : null}
      {children ? <div className={cn(title && "mt-0.5", "text-zinc-300")}>{children}</div> : null}
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
      className="m-auto w-[min(40rem,calc(100vw-1.5rem))] max-h-[calc(100dvh-2rem)] overflow-y-auto rounded-2xl border border-line-strong bg-surface p-0 text-foreground"
    >
      <div className="p-4 sm:p-6">
        <div className="mb-4 flex items-start justify-between gap-4">
          <h2 className="text-lg font-semibold">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-m-2 inline-flex size-11 items-center justify-center rounded-lg text-muted hover:bg-surface-2 hover:text-foreground"
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

/* ------------------------------- ghost glyph ---------------------------------- */

export function GhostMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round">
      <path d="M5 21V11a7 7 0 0 1 14 0v10l-2.4-1.8L14.2 21 12 19.2 9.8 21l-2.4-1.8L5 21Z" />
      <circle cx="9.5" cy="11" r="1.1" fill="currentColor" stroke="none" />
      <circle cx="14.5" cy="11" r="1.1" fill="currentColor" stroke="none" />
    </svg>
  );
}
