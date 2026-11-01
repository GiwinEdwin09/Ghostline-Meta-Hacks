"use client";

import { useEffect, useRef } from "react";

/**
 * The landing page's muted, looping ghost-overlay video (build-plan Part 24). It keeps the
 * poster when the browser refuses to autoplay (iOS Low Power Mode) or the visitor prefers
 * reduced motion.
 */
export function HeroVideo({ src, poster }: { src: string; poster: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      v.pause();
      return;
    }
    v.play().catch(() => {
      /* autoplay blocked: the poster stays */
    });
  }, []);
  return (
    <div className="viewfinder [--vf-color:var(--color-ghost)]">
      <div className="relative overflow-hidden rounded-md border border-line bg-black" style={{ aspectRatio: "16 / 9" }}>
        <video
          ref={ref}
          src={src}
          poster={poster}
          autoPlay
          muted
          loop
          playsInline
          preload="metadata"
          aria-label="A real pendulum with a translucent cyan ghost that predicts where it will swing"
          className="absolute inset-0 size-full object-cover"
        />
        <p className="absolute bottom-3 left-3 inline-flex items-center gap-2 rounded-[4px] bg-black/65 px-2 py-1.5 font-mono text-[0.6875rem] uppercase leading-none tracking-[0.08em] text-white backdrop-blur">
          <span aria-hidden className="size-2 rounded-full border border-ghost bg-ghost/30" />
          This ghost is a prediction
        </p>
      </div>
    </div>
  );
}
