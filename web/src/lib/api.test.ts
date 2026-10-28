import { describe, expect, it } from "vitest";

import { apiUrl } from "@/lib/api";

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
