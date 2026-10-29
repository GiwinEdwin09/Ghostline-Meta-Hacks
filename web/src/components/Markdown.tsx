"use client";

import { Fragment, type ReactNode } from "react";

import { Tex } from "@/components/Tex";

/**
 * A small Markdown renderer for notebook and narrative text (build-plan Part 20): paragraphs,
 * "- " lists, **bold**, *italic*, `code`, inline $math$ and $$display math$$. It builds React
 * elements instead of HTML strings, so model-written text can never inject markup.
 */

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\$[^$\n]+\$)/g;

function inline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{inline(part.slice(2, -2))}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("$") && part.endsWith("$") && part.length > 2) return <Tex key={i} latex={part.slice(1, -1)} />;
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) return <em key={i}>{part.slice(1, -1)}</em>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function Markdown({ text, className }: { text: string; className?: string }) {
  const blocks = text.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
  return (
    <div className={`prose-ghost text-sm leading-relaxed text-zinc-300 ${className ?? ""}`}>
      {blocks.map((block, i) => {
        if (block.startsWith("$$") && block.endsWith("$$")) {
          return <Tex key={i} latex={block.slice(2, -2).trim()} display />;
        }
        const lines = block.split("\n");
        if (lines.every((l) => /^\s*-\s+/.test(l))) {
          return (
            <ul key={i}>
              {lines.map((l, j) => (
                <li key={j}>{inline(l.replace(/^\s*-\s+/, ""))}</li>
              ))}
            </ul>
          );
        }
        return <p key={i}>{inline(lines.join(" "))}</p>;
      })}
    </div>
  );
}
