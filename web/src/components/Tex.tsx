"use client";

import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** LaTeX rendered with KaTeX (build-plan Part 24). `throwOnError: false` shows the source on a typo. */
export function Tex({ latex, display = false, className }: { latex: string; display?: boolean; className?: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(latex, {
        throwOnError: false,
        displayMode: display,
        output: "htmlAndMathml",
        strict: "ignore",
        trust: false,
      }),
    [latex, display],
  );
  return display ? (
    <div className={className} dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
  );
}
