import { describe, expect, it } from "vitest";
import { frameIndexOf, recordsOfKind } from "@meta-sam/parser";

import { parseStoredStream } from "@/lib/sam/replay";

// Frame 0 of a 320x334 clip with two objects, as the SAM 3.1 API emits it (the example in the
// @meta-sam/parser README), split mid-payload the way streaming deltas split.
const outputText =
  '<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>' +
  "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>" +
  ',1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>' +
  "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n";

const split = outputText.indexOf(",1<|box");
const stored = [
  { type: "response.output_text.delta", item_id: "m1", output_index: 0, content_index: 0, delta: outputText.slice(0, split) },
  { type: "response.output_text.delta", item_id: "m1", output_index: 0, content_index: 0, delta: outputText.slice(split) },
  { type: "response.output_text.done", item_id: "m1", output_index: 0, content_index: 0, text: outputText },
  { type: "response.completed" },
];

describe("replaying a stored SAM event stream", () => {
  it("rebuilds the same records the live stream would give", async () => {
    // Stored lines come back from JSONL, so go through JSON first.
    const events = JSON.parse(JSON.stringify(stored)) as unknown[];
    const result = await parseStoredStream(events);
    expect(result.outcome.status).toBe("completed");
    expect(result.diagnostics).toEqual([]);
    expect(recordsOfKind(result.records, "box")).toHaveLength(2);
    const masks = recordsOfKind(result.records, "mask");
    expect(masks.map((m) => m.objectId)).toEqual(["0", "1"]);
    expect(frameIndexOf(masks[0])).toBe(0);
  });

  it("skips lines that are not events", async () => {
    const result = await parseStoredStream([null, 42, { nope: true }, ...stored]);
    expect(recordsOfKind(result.records, "mask")).toHaveLength(2);
  });
});
