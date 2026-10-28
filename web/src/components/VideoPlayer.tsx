"use client";

import { useEffect, useEffectEvent, useRef, useState, type ReactNode, type RefObject } from "react";

import { cn } from "@/components/ui";

/**
 * A video with the details Safari needs (playsInline, preload="metadata", a poster, a
 * fixed aspect ratio so nothing shifts when it loads) plus a smooth time callback and
 * playback-speed chips, which a 240 fps drop needs more than anything.
 */
export function VideoPlayer({
  src,
  poster,
  aspect = 16 / 9,
  videoRef,
  onTime,
  onDuration,
  rates = true,
  children,
  className,
  label,
}: {
  src: string;
  poster?: string;
  aspect?: number;
  videoRef?: RefObject<HTMLVideoElement | null>;
  onTime?: (t: number) => void;
  onDuration?: (d: number) => void;
  rates?: boolean;
  /** Drawn over the picture, for example an SVG trail. */
  children?: ReactNode;
  className?: string;
  label: string;
}) {
  const inner = useRef<HTMLVideoElement | null>(null);
  const ref = videoRef ?? inner;
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [failed, setFailed] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const report = useEffectEvent((t: number) => onTime?.(t));

  // While playing, report the time every animation frame so a chart cursor moves smoothly.
  useEffect(() => {
    if (!playing) return;
    let id = 0;
    const tick = () => {
      const v = ref.current;
      if (v) report(v.currentTime);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, ref]);

  const choose = (r: number) => {
    setRate(r);
    if (ref.current) ref.current.playbackRate = r;
  };

  return (
    <div className={className}>
      <div className="relative w-full overflow-hidden rounded-xl border border-line bg-black" style={{ aspectRatio: String(aspect) }}>
        {failed > 0 && !loaded ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-4 text-center">
            <p className="text-sm text-zinc-300">This video couldn&apos;t be loaded. Check your connection and try again.</p>
            <button
              type="button"
              className="min-h-11 rounded-lg border border-line-strong px-4 text-sm"
              onClick={() => {
                setFailed(0);
                ref.current?.load();
              }}
            >
              Retry
            </button>
          </div>
        ) : null}
        <video
          key={src}
          ref={ref}
          src={src}
          poster={poster}
          playsInline
          preload="metadata"
          controls
          aria-label={label}
          className="absolute inset-0 size-full object-contain"
          onLoadedMetadata={(e) => {
            setLoaded(true);
            e.currentTarget.playbackRate = rate;
            onDuration?.(e.currentTarget.duration);
          }}
          onPlay={() => setPlaying(true)}
          onPause={(e) => {
            setPlaying(false);
            onTime?.(e.currentTarget.currentTime);
          }}
          onEnded={() => setPlaying(false)}
          onSeeked={(e) => onTime?.(e.currentTarget.currentTime)}
          onTimeUpdate={(e) => {
            if (!playing) onTime?.(e.currentTarget.currentTime);
          }}
          onError={() => setFailed((n) => n + 1)}
        />
        {children ? <div className="pointer-events-none absolute inset-0">{children}</div> : null}
      </div>
      {rates ? (
        <div className="mt-2 flex items-center gap-1.5" role="group" aria-label="Playback speed">
          <span className="mr-1 text-xs text-muted">Speed</span>
          {[1, 0.5, 0.25, 0.1].map((r) => (
            <button
              key={r}
              type="button"
              aria-pressed={rate === r}
              onClick={() => choose(r)}
              className={cn(
                "min-h-9 min-w-11 rounded-md border px-2 text-xs font-medium",
                rate === r ? "border-ghost/60 bg-ghost/10 text-ghost" : "border-line text-muted hover:text-foreground",
              )}
            >
              {r}×
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/** Seek a video element to a time, clamped to its length. */
export function seekVideo(v: HTMLVideoElement | null, t: number) {
  if (!v || !Number.isFinite(v.duration)) return;
  v.currentTime = Math.min(Math.max(0, t), v.duration);
}
