import { parseVideoStream, type ResponsesEvent, type VideoSegmentationResult } from "@meta-sam/parser";

/**
 * Replays a stored SAM 3.1 event stream (`sam/{object}.events.jsonl` on the Volume, build-plan
 * Parts 7 and 24) through Meta's own parser, exactly as if it were arriving from the API.
 * The result goes to the `Video` component of @meta-sam/react for frame-accurate playback
 * with the mask drawn on top.
 */
export async function* replayEvents(events: readonly unknown[]): AsyncIterable<ResponsesEvent> {
  for (const e of events) {
    if (e && typeof e === "object" && typeof (e as { type?: unknown }).type === "string") {
      yield e as ResponsesEvent;
    }
  }
}

export async function parseStoredStream(events: readonly unknown[]): Promise<VideoSegmentationResult> {
  const parsed = parseVideoStream(replayEvents(events));
  // Drain the snapshots; the final result carries everything.
  for await (const snapshot of parsed) void snapshot;
  return parsed.finalResult;
}
