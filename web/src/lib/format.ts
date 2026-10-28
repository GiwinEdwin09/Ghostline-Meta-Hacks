import type { Confidence, ReviewStatus, SystemType } from "@/lib/types";

export const SYSTEM_LABELS: Record<SystemType, string> = {
  free_fall: "Free fall",
  projectile: "Projectile",
  pendulum: "Pendulum",
  ramp: "Ramp",
  bounce: "Bounce",
  spin_down: "Spin-down",
  oscillator: "Spring and mass",
  other: "Other",
};

export const CONFIDENCE_LABELS: Record<Confidence, string> = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Can't tell yet",
};

export const REVIEW_LABELS: Record<ReviewStatus, string> = {
  accepted: "Reviewer accepted",
  accepted_with_caveats: "Accepted with caveats",
  contested: "Contested",
  not_reviewed: "Not reviewed",
  pending: "Review pending",
};

/** A number with sensible significant digits and no stray trailing zeros. */
export function fmtNum(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return x.toExponential(Math.max(1, digits - 1));
  return Number(x.toPrecision(digits)).toString();
}

/** "9.80 ± 0.10": the value rounded to the second digit of the uncertainty. */
export function fmtPm(value: number, half: number): string {
  if (!(half > 0) || !Number.isFinite(half)) return fmtNum(value);
  const decimals = Math.min(8, Math.max(0, 1 - Math.floor(Math.log10(half))));
  return `${value.toFixed(decimals)} ± ${half.toFixed(decimals)}`;
}

export function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  return `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
}

export function fmtBytes(n: number): string {
  if (n < 1024 * 1024) return `${Math.max(1, Math.round(n / 1024))} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(n < 10 * 1024 * 1024 ? 1 : 0)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function fmtTime(seconds: number): string {
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
}

export function fmtClock(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
