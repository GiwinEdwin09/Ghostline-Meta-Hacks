"use client";

import { useId, useRef, type PointerEvent } from "react";

import { useElementWidth } from "@/lib/hooks";
import { fmtNum } from "@/lib/format";

export const CHART_COLORS = {
  ghost: "#22d3ee",
  data: "#e7ecf2",
  muted: "#9aa9ba",
  amber: "#fbbf24",
  red: "#f87171",
  green: "#34d399",
  violet: "#a78bfa",
  grid: "#232d39",
} as const;

export interface Series {
  id: string;
  label: string;
  color: string;
  x: ArrayLike<number>;
  y: ArrayLike<number | null>;
  kind?: "line" | "points";
  width?: number;
  dash?: string;
  opacity?: number;
}

export interface Band {
  x: ArrayLike<number>;
  lo: ArrayLike<number>;
  hi: ArrayLike<number>;
  color: string;
}

export interface Region {
  x0: number;
  x1: number;
  color: string;
  label?: string;
}

export interface Mark {
  value: number;
  label?: string;
  color: string;
  dash?: string;
}

interface Props {
  series: Series[];
  bands?: Band[];
  regions?: Region[];
  vlines?: Mark[];
  hlines?: Mark[];
  xLabel?: string;
  yLabel?: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  height?: number;
  /** Controlled cursor position on the x axis. */
  cursor?: number | null;
  onScrub?: (x: number) => void;
  ariaLabel: string;
  legend?: boolean;
  zeroLine?: boolean;
}

const M = { left: 48, right: 12, top: 12, bottom: 30 };

function niceStep(range: number, count: number): number {
  const raw = range / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}

export function niceTicks(lo: number, hi: number, count: number): number[] {
  if (!(hi > lo)) return [lo];
  const step = niceStep(hi - lo, count);
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function extent(values: ArrayLike<number | null>[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const arr of values) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v === null || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo === Infinity ? [0, 1] : [lo, hi];
}

/** Keep the min and max of every pixel column so thousands of points draw as fast as hundreds. */
function decimate(x: ArrayLike<number>, y: ArrayLike<number | null>, sx: (v: number) => number, maxCols: number) {
  const n = x.length;
  const idx: number[] = [];
  if (n <= maxCols * 2) {
    for (let i = 0; i < n; i++) idx.push(i);
    return idx;
  }
  const buckets = new Map<number, { min: number; max: number }>();
  const order: number[] = [];
  for (let i = 0; i < n; i++) {
    if (y[i] === null) continue;
    const col = Math.round(sx(x[i]));
    const b = buckets.get(col);
    if (!b) {
      buckets.set(col, { min: i, max: i });
      order.push(col);
    } else {
      if ((y[i] as number) < (y[b.min] as number)) b.min = i;
      if ((y[i] as number) > (y[b.max] as number)) b.max = i;
    }
  }
  for (const col of order) {
    const b = buckets.get(col)!;
    if (b.min === b.max) idx.push(b.min);
    else if (b.min < b.max) idx.push(b.min, b.max);
    else idx.push(b.max, b.min);
  }
  return idx;
}

export function LineChart({
  series,
  bands = [],
  regions = [],
  vlines = [],
  hlines = [],
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  height = 220,
  cursor,
  onScrub,
  ariaLabel,
  legend = true,
  zeroLine = false,
}: Props) {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>(560);
  const clipId = useId();
  const svgRef = useRef<SVGSVGElement>(null);
  const w = Math.max(160, width);
  const plotW = w - M.left - M.right;
  const plotH = height - M.top - M.bottom;

  const [x0, x1] = xDomain ?? extent(series.map((s) => s.x));
  const ys = [...series.map((s) => s.y), ...bands.flatMap((b) => [b.lo, b.hi])];
  let [y0, y1] = yDomain ?? extent(ys);
  for (const h of hlines) {
    y0 = Math.min(y0, h.value);
    y1 = Math.max(y1, h.value);
  }
  if (zeroLine) {
    y0 = Math.min(y0, 0);
    y1 = Math.max(y1, 0);
  }
  if (!yDomain) {
    const pad = (y1 - y0) * 0.06 || Math.abs(y1) * 0.1 || 1;
    y0 -= pad;
    y1 += pad;
  }
  const sx = (v: number) => M.left + ((v - x0) / (x1 - x0 || 1)) * plotW;
  const sy = (v: number) => M.top + (1 - (v - y0) / (y1 - y0 || 1)) * plotH;

  const xTicks = niceTicks(x0, x1, Math.max(2, Math.floor(plotW / 90)));
  const yTicks = niceTicks(y0, y1, Math.max(2, Math.floor(plotH / 40)));

  const scrub = (e: PointerEvent<SVGRectElement>) => {
    if (!onScrub) return;
    if (e.pointerType === "mouse" && e.buttons === 0) return;
    const rect = svgRef.current!.getBoundingClientRect();
    const px = Math.min(Math.max(e.clientX - rect.left, M.left), M.left + plotW);
    onScrub(x0 + ((px - M.left) / plotW) * (x1 - x0));
  };

  return (
    <div ref={wrapRef} className="w-full">
      <svg
        ref={svgRef}
        width={w}
        height={height}
        role="img"
        aria-label={ariaLabel}
        className="block max-w-full select-none"
      >
        <defs>
          <clipPath id={clipId}>
            <rect x={M.left} y={M.top} width={plotW} height={plotH} />
          </clipPath>
        </defs>
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line x1={M.left} x2={M.left + plotW} y1={sy(v)} y2={sy(v)} stroke={CHART_COLORS.grid} />
            <text x={M.left - 6} y={sy(v) + 4} textAnchor="end" fontSize="11" fill={CHART_COLORS.muted}>
              {fmtNum(v, 3)}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text key={`x${v}`} x={sx(v)} y={height - 10} textAnchor="middle" fontSize="11" fill={CHART_COLORS.muted}>
            {fmtNum(v, 3)}
          </text>
        ))}
        {xLabel ? (
          <text x={M.left + plotW} y={height - 1} textAnchor="end" fontSize="10.5" fill={CHART_COLORS.muted}>
            {xLabel}
          </text>
        ) : null}
        {yLabel ? (
          <text x={M.left} y={9} fontSize="10.5" fill={CHART_COLORS.muted}>
            {yLabel}
          </text>
        ) : null}

        <g clipPath={`url(#${clipId})`}>
          {regions.map((r, i) => (
            <g key={`r${i}`}>
              <rect x={sx(r.x0)} y={M.top} width={Math.max(0, sx(r.x1) - sx(r.x0))} height={plotH} fill={r.color} />
              {r.label ? (
                <text x={sx(r.x0) + 6} y={M.top + 13} fontSize="10.5" fill={CHART_COLORS.muted}>
                  {r.label}
                </text>
              ) : null}
            </g>
          ))}
          {zeroLine ? <line x1={M.left} x2={M.left + plotW} y1={sy(0)} y2={sy(0)} stroke="#34414f" /> : null}
          {bands.map((b, i) => {
            const n = b.x.length;
            if (n === 0) return null;
            let d = `M${sx(b.x[0])},${sy(b.hi[0])}`;
            for (let k = 1; k < n; k++) d += `L${sx(b.x[k])},${sy(b.hi[k])}`;
            for (let k = n - 1; k >= 0; k--) d += `L${sx(b.x[k])},${sy(b.lo[k])}`;
            return <path key={`b${i}`} d={`${d}Z`} fill={b.color} opacity={0.18} />;
          })}
          {hlines.map((h, i) => (
            <g key={`h${i}`}>
              <line x1={M.left} x2={M.left + plotW} y1={sy(h.value)} y2={sy(h.value)} stroke={h.color} strokeDasharray={h.dash ?? "5 4"} />
              {h.label ? (
                <text x={M.left + plotW - 4} y={sy(h.value) - 5} textAnchor="end" fontSize="10.5" fill={h.color}>
                  {h.label}
                </text>
              ) : null}
            </g>
          ))}
          {series.map((s) => {
            const idx = decimate(s.x, s.y, sx, plotW);
            if (s.kind === "points") {
              const step = Math.max(1, Math.ceil(idx.length / 500));
              return (
                <g key={s.id} fill={s.color} opacity={s.opacity ?? 0.7}>
                  {idx.filter((_, k) => k % step === 0).map((i) =>
                    s.y[i] === null ? null : <circle key={i} cx={sx(s.x[i])} cy={sy(s.y[i] as number)} r={1.8} />,
                  )}
                </g>
              );
            }
            let d = "";
            let pen = false;
            for (const i of idx) {
              const v = s.y[i];
              if (v === null || !Number.isFinite(v)) {
                pen = false;
                continue;
              }
              d += `${pen ? "L" : "M"}${sx(s.x[i]).toFixed(1)},${sy(v).toFixed(1)}`;
              pen = true;
            }
            return (
              <path
                key={s.id}
                d={d}
                fill="none"
                stroke={s.color}
                strokeWidth={s.width ?? 1.8}
                strokeDasharray={s.dash}
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity={s.opacity ?? 1}
              />
            );
          })}
          {vlines.map((v, i) => (
            <g key={`v${i}`}>
              <line x1={sx(v.value)} x2={sx(v.value)} y1={M.top} y2={M.top + plotH} stroke={v.color} strokeDasharray={v.dash ?? "4 3"} strokeWidth={1.4} />
              {v.label ? (
                <text x={sx(v.value) + 4} y={M.top + plotH - 6} fontSize="10.5" fill={v.color}>
                  {v.label}
                </text>
              ) : null}
            </g>
          ))}
          {cursor !== null && cursor !== undefined && cursor >= x0 && cursor <= x1 ? (
            <line x1={sx(cursor)} x2={sx(cursor)} y1={M.top} y2={M.top + plotH} stroke="#fff" strokeOpacity={0.85} strokeWidth={1.2} />
          ) : null}
        </g>

        {onScrub ? (
          <rect
            x={M.left}
            y={M.top}
            width={plotW}
            height={plotH}
            fill="transparent"
            style={{ touchAction: "pan-y", cursor: "col-resize" }}
            onPointerDown={(e) => {
              (e.currentTarget as SVGRectElement).setPointerCapture(e.pointerId);
              scrub(e);
            }}
            onPointerMove={scrub}
          />
        ) : null}
      </svg>
      {legend && series.length > 0 ? (
        <ul className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
          {series.map((s) => (
            <li key={s.id} className="inline-flex items-center gap-1.5">
              <span aria-hidden className="inline-block h-0.5 w-4 rounded" style={{ background: s.color, opacity: s.kind === "points" ? 0.6 : 1 }} />
              {s.label}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
