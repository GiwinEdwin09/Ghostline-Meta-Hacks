"use client";

import { Video } from "@meta-sam/react";
import type { VideoSegmentationResult } from "@meta-sam/parser";
import { useEffect, useState } from "react";

import { ErrorState, Skeleton } from "@/components/ui";
import { api, resolveMedia } from "@/lib/api";
import { parseStoredStream } from "@/lib/sam/replay";

/**
 * Frame-accurate playback with the SAM 3.1 mask drawn on top, using Meta's own
 * @meta-sam/react `Video` and @meta-sam/parser (build-plan Part 24). The stored event stream
 * is replayed into the parser. This chunk is only loaded when the user turns on the mask view.
 */
export default function SamMaskVideo({ videoUrl, eventsUrl }: { videoUrl: string; eventsUrl: string }) {
  const [result, setResult] = useState<VideoSegmentationResult | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    api
      .getSamEvents(eventsUrl)
      .then((events) => parseStoredStream(events))
      .then(
        (r) => alive && setResult(r),
        (e) => alive && setError(e),
      );
    return () => {
      alive = false;
    };
  }, [eventsUrl, nonce]);

  if (error) {
    return (
      <ErrorState
        error={error}
        title="The mask overlay couldn't be loaded."
        onRetry={() => {
          setError(null);
          setNonce((n) => n + 1);
        }}
      />
    );
  }
  if (!result) return <Skeleton className="aspect-video w-full" />;
  return (
    <Video
      src={resolveMedia(videoUrl)!}
      result={result}
      objectFit="contain"
      devicePixelRatio={() => window.devicePixelRatio || 1}
      canvasProps={{ "aria-label": "Clip with the SAM 3.1 mask overlaid" }}
      style={{ width: "100%" }}
      onError={(e) => setError(e)}
    />
  );
}
