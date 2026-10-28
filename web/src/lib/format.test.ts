import { describe, expect, it } from "vitest";

import { ApiError, NetworkError, describeError } from "@/lib/errors";
import { fmtBytes, fmtDuration, fmtNum, fmtPm } from "@/lib/format";
import { phaseToStep } from "@/lib/phases";
import { PIPELINE_PHASES, PROVE_PHASES } from "@/lib/types";
import { resolveMedia } from "@/lib/api";

describe("number formatting", () => {
  it("rounds a value to the place of its uncertainty", () => {
    expect(fmtPm(9.8007, 0.1015)).toBe("9.80 ± 0.10");
    expect(fmtPm(0.9, 0.0014)).toBe("0.9000 ± 0.0014");
    expect(fmtPm(1234.5, 12)).toBe("1235 ± 12");
  });

  it("formats sizes, durations and plain numbers", () => {
    expect(fmtBytes(26 * 1024 * 1024)).toBe("26 MB");
    expect(fmtBytes(900)).toBe("1 KB");
    expect(fmtDuration(75)).toBe("1 min 15 s");
    expect(fmtNum(0.000012345, 3)).toBe("1.23e-5");
    expect(fmtNum(null)).toBe("–");
  });
});

describe("pipeline phases", () => {
  it("maps the orchestrator's state machine onto the seven user-facing steps", () => {
    expect(PIPELINE_PHASES.map((p) => phaseToStep(p, PIPELINE_PHASES))).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(phaseToStep("SCENE", PIPELINE_PHASES)).toBe(1);
    expect(phaseToStep("MEASURE", PIPELINE_PHASES)).toBe(2);
    expect(phaseToStep("ANALYZE", PIPELINE_PHASES)).toBe(5);
    expect(phaseToStep("RESPOND", PIPELINE_PHASES)).toBe(6);
    expect(phaseToStep("something new", PIPELINE_PHASES)).toBe(0);
    expect(phaseToStep("render", PROVE_PHASES)).toBe(4);
  });
});

describe("errors say what happened and what to do", () => {
  it("covers network, passcode and server errors", () => {
    expect(describeError(new NetworkError()).title).toContain("Can't reach");
    expect(describeError(new ApiError("nope", 401)).title).toContain("passcode");
    expect(describeError(new ApiError("boom", 503)).hint).toContain("Try again");
    expect(describeError(new ApiError("limit", 429)).title).toContain("busy");
  });
});

describe("media URLs", () => {
  it("keeps absolute and site paths and routes the rest through /api/media", () => {
    expect(resolveMedia("/mock/a.mp4")).toBe("/mock/a.mp4");
    expect(resolveMedia("https://cdn.example/x.mp4")).toBe("https://cdn.example/x.mp4");
    expect(resolveMedia("exp1/trials/t1/playback_720.mp4")).toMatch(/\/api\/media\/exp1\/trials\/t1\/playback_720\.mp4$/);
    expect(resolveMedia(null)).toBeUndefined();
  });
});
