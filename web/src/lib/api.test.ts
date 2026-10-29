import { describe, expect, it } from "vitest";

import { apiUrl, createBody, resolveMedia } from "@/lib/api";

describe("apiUrl", () => {
  it("joins the base and the path with one slash", () => {
    expect(apiUrl("/health", "https://ghostline.modal.run/")).toBe(
      "https://ghostline.modal.run/api/health",
    );
  });

  it("works without slashes on either side", () => {
    expect(apiUrl("health", "http://localhost:8000")).toBe("http://localhost:8000/api/health");
  });
});

describe("resolveMedia", () => {
  it("leaves absolute URLs and site paths alone", () => {
    expect(resolveMedia("https://cdn.example/x.mp4")).toBe("https://cdn.example/x.mp4");
    expect(resolveMedia("/mock/drop.mp4")).toBe("/mock/drop.mp4");
    expect(resolveMedia(null)).toBeUndefined();
  });

  it("serves files of the experiment folder from the media route", () => {
    expect(resolveMedia("e1/trials/t1/poster.jpg")).toBe(
      "http://localhost:8000/api/media/e1/trials/t1/poster.jpg",
    );
  });

  it("serves the computed trajectory from its own API route, not from media", () => {
    expect(resolveMedia("api/experiments/e1/trials/t1/trajectory")).toBe(
      "http://localhost:8000/api/experiments/e1/trials/t1/trajectory",
    );
  });
});

describe("createBody", () => {
  it("nests the setup the way POST /experiments takes it", () => {
    const body = createBody({
      title: "Drop",
      system_hint: "unknown",
      description: "",
      declared_capture: { mode: "slow_mo", fps: 240 },
      plane_offset_cm: 2.5,
      paper: "letter",
      pendulum_length_cm: null,
    });
    expect(body.system_hint).toBe("auto");
    expect(body.declared_capture).toEqual({ mode: "slow_mo", fps: 240 });
    expect(body.setup).toEqual({ plane_offset_cm: 2.5, paper: "letter", pendulum_length_cm: null });
  });

  it("passes a chosen system through", () => {
    const body = createBody({
      title: "x",
      system_hint: "pendulum",
      description: "",
      declared_capture: { mode: "normal", fps: 60 },
      plane_offset_cm: 2,
      paper: "a4",
      pendulum_length_cm: 60,
    });
    expect(body.system_hint).toBe("pendulum");
  });
});
