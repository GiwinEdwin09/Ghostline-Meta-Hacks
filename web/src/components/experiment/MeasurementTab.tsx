"use client";

import dynamic from "next/dynamic";
import { useMemo, useRef, useState } from "react";

import { CHART_COLORS, LineChart, type Series } from "@/components/charts/LineChart";
import { Badge, Card, ErrorState, Notice, SectionTitle, Skeleton, cn } from "@/components/ui";
import { seekVideo, VideoPlayer } from "@/components/VideoPlayer";
import { api, resolveMedia } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import { useResource } from "@/lib/hooks";
import type { Experiment, Trajectory, Trial } from "@/lib/types";

const SamMaskVideo = dynamic(() => import("@/components/experiment/SamMaskVideo"), {
  ssr: false,
  loading: () => <Skeleton className="aspect-video w-full" />,
});

function indexAt(t: readonly number[], x: number): number {
  let lo = 0;
  let hi = t.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= x) lo = mid;
    else hi = mid;
  }
  return x - t[lo] < t[hi] - x ? lo : hi;
}

export function MeasurementTab({ exp }: { exp: Experiment }) {
  const trial = exp.trials.find((t) => t.role === "fit") ?? exp.trials[0];
  const url = trial?.media.trajectory_url ?? null;
  const traj = useResource(`traj:${url}`, () => api.getTrajectory(url!), { enabled: url !== null });
  const [time, setTime] = useState(0);
  const [showTrail, setShowTrail] = useState(true);
  const [showMask, setShowMask] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  if (!trial) return <Notice tone="cyan">There is no measurement yet.</Notice>;
  const cal = trial.calibration;
  const timing = trial.timing;
  const samUrl = trial.media.sam_events_url;
  const winner = exp.models.find((m) => m.id === exp.conclusion?.winner_id);
  const series = winner?.fit?.series ?? null;

  return (
    <div className="space-y-5">
      <section aria-label="Tracked video" className="space-y-3">
        <div className="flex flex-wrap gap-2" role="group" aria-label="Overlays">
          <Toggle checked={showTrail} onChange={setShowTrail} label="Trail" />
          {samUrl ? <Toggle checked={showMask} onChange={setShowMask} label="SAM 3.1 mask" /> : null}
        </div>
        {showMask && samUrl ? (
          <SamMaskVideo videoUrl={trial.media.playback_url} eventsUrl={samUrl} />
        ) : (
          <VideoPlayer
            src={resolveMedia(trial.media.playback_url)!}
            poster={resolveMedia(trial.media.poster_url)}
            aspect={trial.width / trial.height}
            videoRef={videoRef}
            onTime={setTime}
            label="Your clip with the tracked position drawn on it"
          >
            {traj.data ? <TrailOverlay traj={traj.data} trial={trial} time={time} trail={showTrail} /> : null}
          </VideoPlayer>
        )}
      </section>

      <section aria-label="Measured motion">
        <SectionTitle hint="Drag to scrub the video">Position against time</SectionTitle>
        {traj.error ? (
          <ErrorState error={traj.error} onRetry={traj.reload} />
        ) : !traj.data ? (
          <Skeleton className="h-56" />
        ) : (
          <MotionCharts
            traj={traj.data}
            fitSeries={series}
            time={time}
            onScrub={(x) => {
              seekVideo(videoRef.current, x);
              setTime(x);
            }}
          />
        )}
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <SectionTitle>Scale and timing</SectionTitle>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            {cal ? (
              <>
                <Fact label="Scale" value={`${fmtNum(cal.mm_per_px, 3)} mm per pixel`} sub={`± ${(cal.sigma_scale_rel * 100).toFixed(2)}%`} />
                <Fact label="Paper" value={cal.paper === "a4" ? "A4" : "US Letter"} />
                <Fact label="Plane offset" value={`${cal.plane_offset_cm} cm`} />
                <Fact label="Which way is down" value={{ data: "From the motion", plumb: "From a plumb line", camera_level: "From the camera level" }[cal.gravity_source]} />
                <Fact label="Camera movement" value={`${cal.camera_motion_px.toFixed(1)} px`} sub={cal.camera_motion_px > 1 ? "above the 1 px warning level" : "steady"} />
              </>
            ) : null}
            {timing ? (
              <>
                <Fact label="Frame rate" value={`${timing.detected_fps} fps`} sub={timing.declared_mode === "slowmo" ? "slow-mo" : "normal video"} />
                <Fact label="Timing" value={timing.classification === "cfr" ? "Steady" : timing.classification.replace("_", " ")} sub={`jitter ${timing.jitter_pct}%`} />
                <Fact label="Time factor" value={`${timing.time_factor}×`} sub={timing.time_factor === 1 ? "original file" : "re-timed"} />
              </>
            ) : null}
          </dl>
        </Card>
        {trial.media.calib_overlay_url ? (
          <Card>
            <SectionTitle>Calibration overlay</SectionTitle>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={resolveMedia(trial.media.calib_overlay_url)}
              alt="The detected sheet outline with a 5 cm grid drawn through the calibration"
              className="w-full rounded-lg border border-line bg-black"
              style={{ aspectRatio: String(trial.width / trial.height) }}
              loading="lazy"
            />
            <p className="mt-2 text-xs text-muted">The outline is the sheet SAM found; the grid is a 5 cm lattice in the plane of motion.</p>
          </Card>
        ) : null}
      </div>

      <Card>
        <SectionTitle>Quality flags</SectionTitle>
        {exp.quality_flags.length === 0 && trial.quality_flags.length === 0 ? (
          <p className="text-sm text-muted">No problems found: steady camera, sheet fully visible, steady frame rate.</p>
        ) : (
          <ul className="space-y-3">
            {[...exp.quality_flags, ...trial.quality_flags.filter((f) => !exp.quality_flags.some((g) => g.id === f.id))].map((f) => (
              <li key={f.id} className="flex gap-3">
                <Badge tone={f.severity === "major" ? "red" : f.severity === "minor" ? "amber" : "neutral"}>{f.severity}</Badge>
                <div>
                  <p className="text-sm font-medium">{f.title}</p>
                  <p className="text-sm text-zinc-300">{f.detail}</p>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label
      className={cn(
        "inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-lg border px-3.5 text-sm",
        checked ? "border-ghost/60 bg-ghost/10 text-ghost" : "border-line text-muted",
      )}
    >
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="size-4 accent-cyan-400" />
      {label}
    </label>
  );
}

function Fact({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="font-medium tabular-nums">{value}</dd>
      {sub ? <dd className="text-xs text-muted">{sub}</dd> : null}
    </div>
  );
}

function TrailOverlay({ traj, trial, time, trail }: { traj: Trajectory; trial: Trial; time: number; trail: boolean }) {
  const i = indexAt(traj.t, time);
  const pts = useMemo(() => {
    const from = indexAt(traj.t, Math.max(0, time - 0.5));
    return traj.px.slice(from, i + 1).filter((p): p is [number, number] => p !== null);
  }, [traj, time, i]);
  const here = traj.px[i];
  return (
    <svg viewBox={`0 0 ${trial.width} ${trial.height}`} className="size-full" preserveAspectRatio="xMidYMid meet" aria-hidden>
      {trail && pts.length > 1 ? (
        <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="#fff" strokeOpacity={0.85} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
      ) : null}
      {here ? (
        <>
          <circle cx={here[0]} cy={here[1]} r={16} fill="none" stroke={CHART_COLORS.ghost} strokeWidth={3} />
          <circle cx={here[0]} cy={here[1]} r={3} fill={CHART_COLORS.ghost} />
        </>
      ) : null}
    </svg>
  );
}

function MotionCharts({
  traj,
  fitSeries,
  time,
  onScrub,
}: {
  traj: Trajectory;
  fitSeries: NonNullable<NonNullable<Experiment["models"][number]["fit"]>["series"]> | null;
  time: number;
  onScrub: (x: number) => void;
}) {
  const end = traj.t[traj.t.length - 1];
  const region = { x0: traj.split_t, x1: end, color: "rgba(167,139,250,0.12)", label: "held out" };
  const marks = traj.events.map((e) => ({
    value: e.t,
    label: e.kind,
    color: CHART_COLORS.muted,
  }));
  const defs: { key: "x" | "y" | "theta" | "phi"; label: string; unit: string }[] = [
    { key: "x", label: "x", unit: "m" },
    { key: "y", label: "y", unit: "m" },
    { key: "theta", label: "θ", unit: "rad" },
    { key: "phi", label: "φ", unit: "rad" },
  ];
  const shown = defs.filter((d) => traj[d.key] !== null);
  return (
    <div className="space-y-4">
      {shown.map((d) => {
        const data: Series = {
          id: d.key,
          label: `${d.label} measured`,
          color: CHART_COLORS.data,
          x: traj.t,
          y: traj[d.key] as (number | null)[],
          kind: "points",
          opacity: 0.55,
        };
        const fit: Series[] =
          traj.main === d.key && fitSeries
            ? [{ id: "fit", label: "winning model", color: CHART_COLORS.ghost, x: fitSeries.t, y: fitSeries.predicted }]
            : [];
        return (
          <Card key={d.key}>
            <LineChart
              ariaLabel={`${d.label} against time, ${d.unit}`}
              series={[data, ...fit]}
              regions={[region]}
              vlines={marks}
              xLabel="time (s)"
              yLabel={`${d.label} (${d.unit})`}
              height={190}
              cursor={time}
              onScrub={onScrub}
            />
          </Card>
        );
      })}
      <p className="text-xs text-muted">
        Noise about {fmtNum(traj.noise_sigma, 2)} {traj.main_unit}. The shaded stretch is held out: no model sees it while fitting.
      </p>
    </div>
  );
}
