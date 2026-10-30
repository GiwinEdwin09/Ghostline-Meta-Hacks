"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { CHART_COLORS, LineChart } from "@/components/charts/LineChart";
import { Button, Card, ErrorState, Notice, SectionTitle, cn } from "@/components/ui";
import { api, resolveMedia } from "@/lib/api";
import { useElementWidth } from "@/lib/hooks";
import { fmtNum } from "@/lib/format";
import {
  applyH,
  buildScenario,
  computeReadouts,
  controlsFor,
  defaultControls,
  extrapolationWarnings,
  fitTransform,
  instantExplanations,
  poseAt,
  runScenario,
  type ControlSpec,
  type ControlValues,
  type Scenario,
  type SimResult,
} from "@/lib/sim";
import type { Experiment, WhatIfContext } from "@/lib/types";

/**
 * The what-if simulator (build-plan Part 18). The TypeScript twin of the fitted model runs in
 * the browser at a fixed 1/240 s step; two ghosts play side by side on a still frame from
 * the user's own trial: the fitted law (cyan) and the what-if (amber).
 */

const BASE_COLOR = "#22d3ee";
const WHAT_COLOR = "#fbbf24";

function stateAt(res: SimResult, t: number): number[] {
  const i = Math.min(res.states.length - 1, Math.max(0, Math.floor(t / res.dt)));
  return res.states[i];
}

export default function WhatIf({ exp }: { exp: Experiment }) {
  const ctx = exp.whatif!;
  if (!ctx.sim_supported) return <ServerSimulated exp={exp} ctx={ctx} />;
  return <LocalWhatIf exp={exp} ctx={ctx} />;
}

function LocalWhatIf({ exp, ctx }: { exp: Experiment; ctx: WhatIfContext }) {
  const controls = useMemo(() => controlsFor(ctx), [ctx]);
  const defaults = useMemo(() => defaultControls(ctx), [ctx]);
  const [values, setValues] = useState<ControlValues>(defaults);
  const [playing, setPlaying] = useState(true);
  const [explainText, setExplainText] = useState<string | null>(null);
  const [explainBusy, setExplainBusy] = useState(false);
  const [explainError, setExplainError] = useState<unknown>(null);

  const base = useMemo(() => {
    const sc = buildScenario(ctx, defaults);
    const res = runScenario(sc);
    return { sc, res, read: computeReadouts(sc, res) };
  }, [ctx, defaults]);
  const what = useMemo(() => {
    const sc = buildScenario(ctx, values);
    const res = runScenario(sc);
    return { sc, res, read: computeReadouts(sc, res) };
  }, [ctx, values]);

  const warnings = useMemo(() => extrapolationWarnings(ctx, values), [ctx, values]);
  const explanations = useMemo(
    () => instantExplanations(ctx, defaults, values, base.sc, what.sc, base.read, what.read),
    [ctx, defaults, values, base, what],
  );
  const isDefault = controls.every((c) => values[c.id] === defaults[c.id]);

  const set = (id: string, v: number | string) => {
    setValues((s) => ({ ...s, [id]: v }));
    setExplainText(null);
  };

  const explain = async () => {
    setExplainBusy(true);
    setExplainError(null);
    try {
      const numbers: Record<string, number | string> = {};
      for (const r of what.read) if (r.value !== null) numbers[r.label] = Number(r.value.toFixed(4));
      for (const c of controls) numbers[c.label] = values[c.id];
      const out = await api.explain(exp.id, { summary: explanations.join(" "), numbers });
      setExplainText(out.text);
    } catch (err) {
      setExplainError(err);
    } finally {
      setExplainBusy(false);
    }
  };

  const tMax = Math.max(base.res.t[base.res.t.length - 1], what.res.t[what.res.t.length - 1]);
  const obsName = base.sc.model.observableNames[0];
  const obsUnit = ctx.system === "spin_down" || ctx.system === "pendulum" ? "rad" : "m";

  return (
    <div className="space-y-5">
      <p className="text-sm text-zinc-300">
        Change the world and watch <span className="font-medium text-foreground">your own {ctx.object.name.toLowerCase()}</span> follow the law you measured.
        The cyan ghost is the fitted model; the amber one is your what-if.
      </p>

      <Stage ctx={ctx} exp={exp} base={base} what={what} playing={playing} />

      <div className="flex flex-wrap gap-2">
        <Button onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
          {playing ? "Pause" : "Play"}
        </Button>
        <Button tone="ghost" onClick={() => setValues(defaults)} disabled={isDefault}>
          Reset to measured
        </Button>
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card as="section">
          <SectionTitle>Controls</SectionTitle>
          <div className="space-y-5">
            {controls.map((c) => (
              <Control key={c.id} spec={c} value={values[c.id]} onChange={(v) => set(c.id, v)} />
            ))}
          </div>
          {warnings.map((w) => (
            <div key={w} className="mt-4">
              <Notice tone="amber">{w}</Notice>
            </div>
          ))}
        </Card>

        <div className="space-y-5">
          <Card as="section">
            <SectionTitle>What changes</SectionTitle>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-muted">
                  <th className="pb-1 font-medium" />
                  <th className="pb-1 font-medium" style={{ color: BASE_COLOR }}>Measured</th>
                  <th className="pb-1 font-medium" style={{ color: WHAT_COLOR }}>What if</th>
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {what.read.map((r, i) => {
                  const b = base.read[i];
                  return (
                    <tr key={r.id} className="border-t border-line">
                      <th scope="row" className="py-1.5 pr-2 text-left font-normal text-zinc-300">{r.label}</th>
                      <td className="py-1.5 pr-2">{b?.value === null || b === undefined ? "–" : `${fmtNum(b.value, 3)} ${b.unit}`}</td>
                      <td className="py-1.5 font-medium">{r.value === null ? "–" : `${fmtNum(r.value, 3)} ${r.unit}`}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <ul className="mt-3 space-y-1.5 text-sm text-zinc-300">
              {explanations.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
            {!isDefault ? (
              <div className="mt-3">
                <Button onClick={explain} disabled={explainBusy}>
                  {explainBusy ? "Asking Wonder…" : "Explain with Wonder"}
                </Button>
                {explainText ? <p className="mt-3 rounded-lg border border-line bg-background p-3 text-sm text-zinc-200">{explainText}</p> : null}
                {explainError ? <div className="mt-3"><ErrorState error={explainError} /></div> : null}
              </div>
            ) : null}
          </Card>

          <Card as="section">
            <LineChart
              ariaLabel={`${obsName} over time for the measured law and the what-if`}
              series={[
                { id: "base", label: "Measured law", color: BASE_COLOR, x: base.res.t, y: base.res.obs[0] },
                { id: "what", label: "What if", color: WHAT_COLOR, x: what.res.t, y: what.res.obs[0], dash: isDefault ? "5 4" : undefined },
              ]}
              xDomain={[0, tMax]}
              xLabel="time (s)"
              yLabel={`${obsName} (${obsUnit})`}
              height={200}
              zeroLine
            />
          </Card>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Controls                                                                              */
/* ------------------------------------------------------------------------------------ */

function Control({ spec, value, onChange }: { spec: ControlSpec; value: number | string; onChange: (v: number | string) => void }) {
  const id = `ctl-${spec.id}`;
  if (spec.type === "choice") {
    return (
      <div>
        <p className="text-sm font-medium" id={`${id}-label`}>{spec.label}</p>
        <div role="radiogroup" aria-labelledby={`${id}-label`} className="mt-2 flex flex-wrap gap-2">
          {spec.options!.map((o) => (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={value === o.value}
              onClick={() => onChange(o.value)}
              className={cn("min-h-11 rounded-lg border px-3.5 text-sm", value === o.value ? "border-ghost bg-ghost/10 text-ghost" : "border-line text-muted hover:text-foreground")}
            >
              {o.label}
            </button>
          ))}
        </div>
        {spec.help ? <p className="mt-1.5 text-xs text-muted">{spec.help}</p> : null}
      </div>
    );
  }
  const v = Number(value);
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium">{spec.label}</label>
        <output htmlFor={id} className="text-sm tabular-nums text-amber-300">
          {v.toFixed(spec.decimals)} {spec.unit}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={spec.min}
        max={spec.max}
        step={spec.step}
        value={Math.min(spec.max, Math.max(spec.min, v))}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 h-11 w-full accent-amber-400"
      />
      {spec.presets ? (
        <div className="flex flex-wrap gap-1.5">
          {spec.presets.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => onChange(p.value)}
              className={cn(
                "min-h-9 rounded-md border px-2.5 text-xs",
                Math.abs(v - p.value) < spec.step / 2 + 1e-9 ? "border-amber-400/60 bg-amber-400/10 text-amber-300" : "border-line text-muted hover:text-foreground",
              )}
            >
              {p.label} {p.value}
            </button>
          ))}
        </div>
      ) : null}
      {spec.help ? <p className="mt-1.5 text-xs text-muted">{spec.help}</p> : null}
    </div>
  );
}

/* ------------------------------------------------------------------------------------ */
/* The stage: a canvas over a still frame from the user's own trial                      */
/* ------------------------------------------------------------------------------------ */

interface Run {
  sc: Scenario;
  res: SimResult;
}

function Stage({ ctx, exp, base, what, playing }: { ctx: WhatIfContext; exp: Experiment; base: Run; what: Run; playing: boolean }) {
  const trial = exp.trials.find((t) => t.role === "fit") ?? exp.trials[0];
  const aspect = trial ? trial.width / trial.height : 16 / 9;
  const H = trial?.calibration?.world_to_px ?? null;
  const [wrapRef, width] = useElementWidth<HTMLDivElement>(640);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const clock = useRef(0);
  const image = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!ctx.still_url) return;
    const img = new Image();
    img.src = resolveMedia(ctx.still_url)!;
    image.current = img;
    return () => {
      image.current = null;
    };
  }, [ctx.still_url]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const g = canvas.getContext("2d");
    if (!g) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cw = Math.round(width);
    const ch = Math.round(width / aspect);
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    const kx = cw / (trial?.width ?? 960);

    // World to canvas pixels: the trial's calibration if we have it, else fit the motion box.
    const sampleWorld: [number, number][] = [];
    if (!H) {
      for (let i = 0; i < base.res.states.length; i += 12) {
        const p = poseAt(ctx.system, base.sc, base.res.states[i], ctx.anchor_m);
        sampleWorld.push([p.x, p.y]);
      }
      sampleWorld.push([ctx.anchor_m[0], ctx.anchor_m[1]]);
    }
    const fallback = H ? null : fitTransform(sampleWorld, cw, ch);
    const toCanvas = (x: number, y: number): [number, number] => {
      if (fallback) return fallback(x, y);
      const [px, py] = applyH(H!, x, y);
      return [px * kx, py * kx];
    };
    const radiusPx = (x: number, y: number) => {
      const [a, b] = toCanvas(x, y);
      const [c, d] = toCanvas(x + ctx.object.radius_m, y);
      return Math.max(5, Math.hypot(c - a, d - b));
    };

    const duration = Math.max(base.res.t[base.res.t.length - 1], what.res.t[what.res.t.length - 1]) + 1;
    let raf = 0;
    let last = performance.now();

    const draw = (t: number) => {
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, cw, ch);
      const img = image.current;
      if (img && img.complete && img.naturalWidth > 0) {
        g.drawImage(img, 0, 0, cw, ch);
        g.fillStyle = "rgba(7,9,12,0.35)";
        g.fillRect(0, 0, cw, ch);
      } else {
        g.fillStyle = "#0e1319";
        g.fillRect(0, 0, cw, ch);
      }
      const [ax, ay] = toCanvas(ctx.anchor_m[0], ctx.anchor_m[1]);

      // Scenery for each system.
      g.lineWidth = 1.5;
      g.strokeStyle = "rgba(255,255,255,0.35)";
      if (ctx.system === "free_fall" || ctx.system === "bounce") {
        g.beginPath();
        g.moveTo(ax - cw * 0.15, ay);
        g.lineTo(ax + cw * 0.15, ay);
        g.stroke();
      } else if (ctx.system === "ramp") {
        const a = what.sc.params.alpha;
        const dir = what.sc.params.dir_x ?? 1; // -1 when the object slid to the left in the clip
        const len = (what.sc.geometry.sMax ?? 1.2) + 0.1;
        const [bx, by] = toCanvas(ctx.anchor_m[0] + dir * len * Math.cos(a), ctx.anchor_m[1] - len * Math.sin(a));
        g.beginPath();
        g.moveTo(ax, ay);
        g.lineTo(bx, by);
        g.stroke();
      } else if (ctx.system === "spin_down") {
        g.beginPath();
        g.arc(ax, ay, radiusPx(ctx.anchor_m[0], ctx.anchor_m[1]) * 0.6, 0, Math.PI * 2);
        g.stroke();
      }

      const drawBody = (run: Run, color: string, filled: boolean) => {
        const st = stateAt(run.res, t);
        const pose = poseAt(ctx.system, run.sc, st, ctx.anchor_m);
        const [x, y] = toCanvas(pose.x, pose.y);
        const r = radiusPx(pose.x, pose.y);
        if (ctx.system === "pendulum" || ctx.system === "spin_down") {
          g.strokeStyle = color;
          g.globalAlpha = filled ? 0.45 : 0.3;
          g.lineWidth = 1.5;
          g.beginPath();
          g.moveTo(ax, ay);
          g.lineTo(x, y);
          g.stroke();
        }
        g.globalAlpha = 1;
        if (!filled) {
          const grad = g.createRadialGradient(x, y, 0, x, y, r * 2.4);
          grad.addColorStop(0, "rgba(34,211,238,0.35)");
          grad.addColorStop(1, "rgba(34,211,238,0)");
          g.fillStyle = grad;
          g.beginPath();
          g.arc(x, y, r * 2.4, 0, Math.PI * 2);
          g.fill();
        }
        g.fillStyle = color;
        g.globalAlpha = filled ? 0.95 : 0.55;
        g.beginPath();
        g.arc(x, y, r, 0, Math.PI * 2);
        g.fill();
        g.globalAlpha = 1;
        g.strokeStyle = filled ? "#fff" : color;
        g.lineWidth = 1.5;
        g.stroke();
      };
      drawBody(base, BASE_COLOR, false);
      drawBody(what, WHAT_COLOR, true);

      g.font = "12px system-ui, sans-serif";
      g.fillStyle = "#e7ecf2";
      g.fillText(`t = ${t.toFixed(2)} s`, 10, ch - 10);
    };

    const frame = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (playing) clock.current = (clock.current + dt) % duration;
      draw(clock.current);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [ctx, base, what, playing, width, aspect, H, trial]);

  return (
    <div ref={wrapRef} className="w-full">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Your ${ctx.object.name.toLowerCase()} in the measured law (cyan ghost) and in the what-if (amber)`}
        className="block w-full rounded-xl border border-line bg-black"
        style={{ aspectRatio: String(aspect) }}
      />
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-full" style={{ background: BASE_COLOR }} /> Measured law
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-full" style={{ background: WHAT_COLOR }} /> What if
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------------------ */
/* Wonder-written models are simulated on the server                                     */
/* ------------------------------------------------------------------------------------ */

function ServerSimulated({ exp, ctx }: { exp: Experiment; ctx: WhatIfContext }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [out, setOut] = useState<{ t: number[]; y: number[] } | null>(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.simulate(exp.id, { params: ctx.params, initial: ctx.initial, t_end: 12, dt: 1 / 60 });
      setOut({ t: r.t, y: r.observables[0] });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card className="space-y-4">
      <Notice tone="cyan" title="This law was written by Wonder">
        Custom models run in the server&apos;s sandbox instead of your browser, so each run takes a moment.
      </Notice>
      <Button tone="primary" onClick={run} disabled={busy}>
        {busy ? "Simulating…" : "Simulate"}
      </Button>
      {error ? <ErrorState error={error} /> : null}
      {out ? (
        <LineChart ariaLabel="Simulated motion" series={[{ id: "sim", label: "Simulation", color: WHAT_COLOR, x: out.t, y: out.y }]} xLabel="time (s)" height={200} />
      ) : null}
      <p className="text-xs" style={{ color: CHART_COLORS.muted }}>Sliders for custom models arrive with the server-side simulator.</p>
    </Card>
  );
}
