"use client";

import { useMemo, useRef, useState } from "react";

import { CHART_COLORS, LineChart } from "@/components/charts/LineChart";
import { Button, Card, ErrorState, Notice, SectionTitle, Skeleton, cn } from "@/components/ui";
import { seekVideo, VideoPlayer } from "@/components/VideoPlayer";
import { api, resolveMedia } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import { useResource } from "@/lib/hooks";
import type { GhostRender, GhostTrack } from "@/lib/types";

/**
 * The server-rendered ghost video (build-plan Part 16) with the error chart beside it.
 * The chart's cursor follows the video, and dragging on the chart seeks the video. A
 * "v1 / v2" toggle compares the first prediction with the revised one (Part 17).
 */

function nearest(t: readonly number[], x: number): number {
  let lo = 0;
  let hi = t.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(t[lo] - x) <= Math.abs(t[hi] - x) ? lo : hi;
}

export function GhostPlayer({
  ghosts,
  onRevise,
  revising,
  reviseError,
  aspect,
}: {
  ghosts: GhostRender[];
  onRevise?: () => void;
  revising?: boolean;
  reviseError?: unknown;
  aspect: number;
}) {
  const sorted = useMemo(() => [...ghosts].sort((a, b) => a.version - b.version), [ghosts]);
  const [version, setVersion] = useState<number>(sorted[sorted.length - 1].version);
  const [slow, setSlow] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const ghost = sorted.find((g) => g.version === version) ?? sorted[sorted.length - 1];
  const t1 = useResource(`track:${sorted[0].track_url}`, () => api.getGhostTrack(sorted[0].track_url));
  const t2 = useResource(
    `track:${sorted[1]?.track_url ?? "none"}`,
    () => api.getGhostTrack(sorted[1]!.track_url),
    { enabled: sorted.length > 1 },
  );
  const tracks: Record<number, GhostTrack | undefined> = { 1: t1.data, 2: t2.data };
  const track = tracks[ghost.version];

  const useSlow = slow && ghost.slow_video_url !== null;
  const src = resolveMedia(useSlow ? ghost.slow_video_url : ghost.video_url)!;
  const span = track ? track.t[track.t.length - 1] + (track.t[1] - track.t[0]) : 0;
  // Chart time is real time. A slow-motion file is longer, so scale its clock back.
  const scale = useSlow && duration > 0 && span > 0 ? span / duration : 1;
  const realTime = time * scale;

  const errNow = track ? track.error_mm[nearest(track.t, realTime)] : null;
  const within = errNow !== null && track ? errNow < track.tau_mm : null;

  const series = [
    ...(tracks[1]
      ? [{
          id: "v1",
          label: sorted.length > 1 ? "Ghost v1" : "Error",
          color: sorted.length > 1 ? CHART_COLORS.amber : CHART_COLORS.ghost,
          x: tracks[1].t,
          y: tracks[1].error_mm,
          opacity: ghost.version === 1 || sorted.length === 1 ? 1 : 0.45,
        }]
      : []),
    ...(tracks[2]
      ? [{
          id: "v2",
          label: "Ghost v2",
          color: CHART_COLORS.ghost,
          x: tracks[2].t,
          y: tracks[2].error_mm,
          opacity: ghost.version === 2 ? 1 : 0.45,
        }]
      : []),
  ];

  const lastRevision = sorted.find((g) => g.revision)?.revision ?? null;
  const needsRevision = ghost.divergence_onset_s !== null && !lastRevision && onRevise;

  return (
    <div className="space-y-4">
      {sorted.length > 1 ? (
        <div role="group" aria-label="Ghost version" className="flex flex-wrap items-center gap-2">
          {sorted.map((g) => (
            <button
              key={g.version}
              type="button"
              aria-pressed={g.version === version}
              onClick={() => setVersion(g.version)}
              className={cn(
                "min-h-11 rounded-lg border px-4 text-sm font-medium",
                g.version === version ? "border-ghost/60 bg-ghost/10 text-ghost" : "border-line text-muted hover:text-foreground",
              )}
            >
              Ghost v{g.version}
              <span className="ml-2 text-xs font-normal opacity-80">{g.model_name.split(":")[0]}</span>
            </button>
          ))}
          {ghost.slow_video_url ? (
            <button
              type="button"
              aria-pressed={slow}
              onClick={() => setSlow((s) => !s)}
              className={cn("min-h-11 rounded-lg border px-4 text-sm", slow ? "border-ghost/60 bg-ghost/10 text-ghost" : "border-line text-muted")}
            >
              Slow motion
            </button>
          ) : null}
        </div>
      ) : ghost.slow_video_url ? (
        <div className="flex gap-2">
          <button
            type="button"
            aria-pressed={slow}
            onClick={() => setSlow((s) => !s)}
            className={cn("min-h-11 rounded-lg border px-4 text-sm", slow ? "border-ghost/60 bg-ghost/10 text-ghost" : "border-line text-muted")}
          >
            {slow ? "Slow motion on" : "Slow motion"}
          </button>
        </div>
      ) : null}

      <VideoPlayer
        src={src}
        poster={resolveMedia(ghost.poster_url)}
        aspect={aspect}
        videoRef={videoRef}
        onTime={setTime}
        onDuration={setDuration}
        label={`Ghost v${ghost.version} overlaid on the second trial`}
      />

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="Error now"
          value={errNow === null ? "–" : `Δ ${fmtNum(errNow, 2)} mm`}
          tone={within === null ? undefined : within ? "good" : "bad"}
        />
        <Stat label="Predicting ahead" value={`${fmtNum(ghost.predicted_ahead_s, 2)} s`} />
        <Stat label="Largest error" value={`${fmtNum(ghost.max_error_mm, 3)} mm`} />
        <Stat
          label={`Within τ = ${fmtNum(ghost.threshold_mm, 2)} mm`}
          value={ghost.divergence_onset_s === null ? "the whole clip" : `until ${fmtNum(ghost.divergence_onset_s, 2)} s`}
          tone={ghost.divergence_onset_s === null ? "good" : "warn"}
        />
      </dl>

      <Card>
        <SectionTitle hint="Drag on the chart to scrub the video">How far reality drifted from the ghost</SectionTitle>
        {t1.error ? (
          <ErrorState error={t1.error} onRetry={t1.reload} />
        ) : !track ? (
          <Skeleton className="h-56" />
        ) : (
          <LineChart
            ariaLabel={`Ghost error over time. Largest error ${fmtNum(ghost.max_error_mm, 3)} millimetres.`}
            series={series}
            hlines={[{ value: track.tau_mm, label: `τ = ${fmtNum(track.tau_mm, 2)} mm`, color: CHART_COLORS.green }]}
            vlines={ghost.divergence_onset_s !== null ? [{ value: ghost.divergence_onset_s, label: "drift begins", color: CHART_COLORS.red }] : []}
            regions={[{ x0: 0, x1: track.launch_window_s, color: "rgba(167,139,250,0.14)", label: "launch window" }]}
            xLabel="time (s)"
            yLabel="error (mm)"
            cursor={realTime}
            onScrub={(x) => seekVideo(videoRef.current, x / scale)}
            height={230}
            yDomain={[0, Math.max(track.tau_mm * 2.2, ...sorted.map((g) => g.max_error_mm * 1.1))]}
          />
        )}
      </Card>

      {ghost.divergence_onset_s !== null && !ghost.revision ? (
        <Notice tone="amber" title={`The ghost drifted at ${fmtNum(ghost.divergence_onset_s, 2)} s`}>
          After the launch window the ghost is a pure prediction. It stayed within {fmtNum(ghost.threshold_mm, 2)} mm for{" "}
          {fmtNum(ghost.divergence_onset_s - (track?.launch_window_s ?? 0), 2)} s, then reality pulled away. That is useful: it points at physics the model is missing.
        </Notice>
      ) : null}

      {needsRevision ? (
        <div>
          <Button tone="primary" onClick={onRevise} disabled={revising}>
            {revising ? "Wonder is looking at the drift…" : "Why did it drift?"}
          </Button>
          {reviseError ? <div className="mt-3"><ErrorState error={reviseError} /></div> : null}
        </div>
      ) : null}

      {lastRevision ? (
        <Card>
          <SectionTitle>Wonder&apos;s diagnosis</SectionTitle>
          <p className="text-sm text-zinc-300">{lastRevision.diagnosis}</p>
          <p className="mt-2 text-sm text-zinc-300">
            <span className="font-medium text-foreground">{lastRevision.adopted ? "Revision adopted. " : "Revision not adopted. "}</span>
            {lastRevision.note}
          </p>
        </Card>
      ) : null}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "good" | "bad" | "warn" }) {
  return (
    <div className="rounded-lg border border-line bg-surface p-3">
      <dt className="text-xs text-muted">{label}</dt>
      <dd
        className={cn(
          "mt-0.5 text-base font-semibold tabular-nums",
          tone === "good" && "text-emerald-300",
          tone === "bad" && "text-red-300",
          tone === "warn" && "text-amber-300",
        )}
      >
        {value}
      </dd>
    </div>
  );
}
