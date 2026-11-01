"use client";

import { useEffect, useEffectEvent, useRef, useState, type ReactNode, type RefObject } from "react";

import { buttonClass, cn } from "@/components/ui";

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
  framed = false,
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
  /** Cyan viewfinder corners, used where the video shows a ghost. */
  framed?: boolean;
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
      <div className={framed ? "viewfinder [--vf-color:rgb(34_211_238/0.7)]" : undefined}>
        <div className="relative w-full overflow-hidden rounded-md border border-line bg-black" style={{ aspectRatio: String(aspect) }}>
          {failed > 0 && !loaded ? (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-black p-4 text-center">
              <p className="text-sm text-fg-2">This video couldn&apos;t be loaded. Check your connection and try again.</p>
              <button
                type="button"
                className={buttonClass("secondary")}
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
      </div>
      {rates ? (
        <div className="mt-3 flex flex-wrap items-center gap-1.5" role="group" aria-label="Playback speed">
          <span className="eyebrow mr-1.5">Speed</span>
          {[1, 0.5, 0.25, 0.1].map((r) => (
            <button
              key={r}
              type="button"
              aria-pressed={rate === r}
              onClick={() => choose(r)}
              className={cn(
                "min-h-9 min-w-11 rounded-md border px-2 font-mono text-xs font-medium tabular-nums transition-colors",
                rate === r ? "border-ghost/60 bg-ghost/10 text-cyan-100" : "border-line text-muted hover:border-line-strong hover:text-foreground",
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
