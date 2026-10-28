"use client";

import { useState } from "react";

import { Button, Dialog, ErrorState, Notice, Skeleton, cn } from "@/components/ui";
import { api } from "@/lib/api";
import { fmtNum } from "@/lib/format";
import { useResource } from "@/lib/hooks";
import type { Experiment, ReplayCheck } from "@/lib/types";

/** Export to simulator (build-plan Part 19): format, mass, the generated file and the replay check. */
export function ExportDialog({ exp, open, onClose }: { exp: Experiment; open: boolean; onClose: () => void }) {
  const formats = exp.export_formats;
  const [format, setFormat] = useState<"json" | "pybullet">(formats[0] ?? "json");
  const [massG, setMassG] = useState("");
  const [copied, setCopied] = useState(false);
  const [check, setCheck] = useState<{ busy: boolean; result: ReplayCheck | null; error: unknown }>({
    busy: false,
    result: null,
    error: null,
  });
  const mass = massG.trim() === "" ? null : Number(massG);
  const massOk = mass === null || (Number.isFinite(mass) && mass > 0);

  const file = useResource(
    `export:${exp.id}:${format}:${mass ?? ""}`,
    () => api.getExport(exp.id, format, mass),
    { enabled: open && formats.length > 0 && massOk },
  );

  const copy = async () => {
    if (!file.data) return;
    try {
      await navigator.clipboard.writeText(file.data.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard blocked: the text is selectable below */
    }
  };

  const download = () => {
    if (!file.data) return;
    const url = URL.createObjectURL(new Blob([file.data.text], { type: file.data.content_type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = file.data.filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const runCheck = async () => {
    setCheck({ busy: true, result: null, error: null });
    try {
      await api.verifyExport(exp.id);
      const json = await api.getExport(exp.id, "json", mass);
      const parsed = JSON.parse(json.text) as { replay_check: ReplayCheck | null };
      setCheck({ busy: false, result: parsed.replay_check, error: null });
    } catch (error) {
      setCheck({ busy: false, result: null, error });
    }
  };

  return (
    <Dialog open={open} onClose={onClose} title="Export to simulator">
      <p className="text-sm text-zinc-300">
        The measured parameters, with their 95% intervals, in a file a robotics simulator can load. Intervals cover measurement
        uncertainty only; part-to-part variation needs wider ranges.
      </p>

      <fieldset className="mt-4">
        <legend className="text-sm font-medium">Format</legend>
        <div className="mt-2 flex flex-wrap gap-2">
          {formats.map((f) => (
            <label
              key={f}
              className={cn(
                "inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-lg border px-3.5 text-sm",
                format === f ? "border-ghost bg-ghost/10" : "border-line",
              )}
            >
              <input type="radio" name="fmt" className="size-4 accent-cyan-400" checked={format === f} onChange={() => setFormat(f)} />
              {f === "json" ? "Ghostline JSON" : "PyBullet snippet"}
            </label>
          ))}
        </div>
      </fieldset>

      <div className="mt-4">
        <label htmlFor="mass" className="text-sm font-medium">
          Object mass in grams (optional)
        </label>
        <p className="text-xs text-muted">A kitchen scale is enough. Needed to turn rates into forces and torques.</p>
        <input
          id="mass"
          inputMode="decimal"
          value={massG}
          onChange={(e) => setMassG(e.target.value)}
          aria-invalid={!massOk}
          className="mt-1.5 h-11 w-36 rounded-lg border border-line-strong bg-background px-3 text-base"
        />
        {!massOk ? <p className="mt-1 text-xs text-red-300">Enter a number greater than zero.</p> : null}
      </div>

      <div className="mt-4">
        {file.error ? (
          <ErrorState error={file.error} onRetry={file.reload} />
        ) : !file.data ? (
          <Skeleton className="h-40" />
        ) : (
          <>
            <pre
              tabIndex={0}
              aria-label="Generated file"
              className="max-h-64 overflow-auto rounded-lg border border-line bg-background p-3 font-mono text-xs leading-relaxed text-zinc-200"
            >
              {file.data.text}
            </pre>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button tone="primary" onClick={download}>
                Download {file.data.filename}
              </Button>
              <Button onClick={copy}>{copied ? "Copied" : "Copy"}</Button>
            </div>
          </>
        )}
      </div>

      {formats.includes("pybullet") ? (
        <div className="mt-5 border-t border-line pt-4">
          <p className="text-sm font-medium">Replay check</p>
          <p className="text-xs text-muted">Runs the same experiment in PyBullet and compares it with what you measured.</p>
          <Button className="mt-2" onClick={runCheck} disabled={check.busy}>
            {check.busy ? "Running in PyBullet…" : "Run replay check"}
          </Button>
          {check.error ? (
            <div className="mt-3">
              <ErrorState error={check.error} />
            </div>
          ) : null}
          {check.result ? (
            <div className="mt-3">
              <Notice tone={check.result.passed ? "cyan" : "amber"} title={check.result.passed ? "Reproduced in PyBullet" : "Does not match"}>
                {check.result.quantity}: simulated {fmtNum(check.result.simulated)} {check.result.unit}, measured {fmtNum(check.result.measured)}{" "}
                {check.result.unit} (95% interval {fmtNum(check.result.ci95[0])} to {fmtNum(check.result.ci95[1])}).
              </Notice>
            </div>
          ) : null}
        </div>
      ) : null}
    </Dialog>
  );
}
