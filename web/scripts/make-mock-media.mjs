// Renders the small synthetic clips that the mock experiments play (npm run mock:media).
//
// The motion comes from the same equations the mocks use (RK4 for the pendulum, closed
// forms for the spinner and the drop) with the parameters in src/lib/mocks/scenarios.json,
// so the picture agrees with the charts. There is no model call and no camera involved:
// a tiny software rasterizer draws the frames and pipes them to ffmpeg (H.264, faststart,
// 480x270, a few hundred KB each). Requires ffmpeg on PATH.
//
//   node scripts/make-mock-media.mjs

import { spawn } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const pub = path.join(here, "..", "public");
const mock = path.join(pub, "mock");
mkdirSync(mock, { recursive: true });
const sc = JSON.parse(readFileSync(path.join(here, "..", "src", "lib", "mocks", "scenarios.json"), "utf8"));

const W = 480;
const H = 270;
const S = W / 960; // scene coordinates are 960x540
const DEG = Math.PI / 180;

/* ------------------------------- rasterizer -------------------------------- */

class Raster {
  constructor(bg) {
    this.bg = bg;
    this.px = new Uint8ClampedArray(W * H * 3);
  }
  reset() {
    this.px.set(this.bg);
  }
  blend(i, r, g, b, a) {
    const p = this.px;
    p[i] = p[i] + (r - p[i]) * a;
    p[i + 1] = p[i + 1] + (g - p[i + 1]) * a;
    p[i + 2] = p[i + 2] + (b - p[i + 2]) * a;
  }
  disc(cx, cy, r, [cr, cg, cb], alpha = 1, shade = false) {
    cx *= S;
    cy *= S;
    r *= S;
    const x0 = Math.max(0, Math.floor(cx - r - 1));
    const x1 = Math.min(W - 1, Math.ceil(cx + r + 1));
    const y0 = Math.max(0, Math.floor(cy - r - 1));
    const y1 = Math.min(H - 1, Math.ceil(cy + r + 1));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy);
        const cov = Math.min(1, Math.max(0, r + 0.5 - d));
        if (cov <= 0) continue;
        let k = 1;
        if (shade) k = 1.25 - 0.55 * Math.hypot((x + 0.5 - cx + r * 0.35) / r, (y + 0.5 - cy + r * 0.35) / r) * 0.7;
        this.blend((y * W + x) * 3, cr * k, cg * k, cb * k, cov * alpha);
      }
    }
  }
  glow(cx, cy, r, [cr, cg, cb], strength) {
    cx *= S;
    cy *= S;
    r *= S;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(W - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(H - 1, Math.ceil(cy + r));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / r;
        if (d >= 1) continue;
        this.blend((y * W + x) * 3, cr, cg, cb, strength * (1 - d) ** 2);
      }
    }
  }
  line(x1, y1, x2, y2, width, [cr, cg, cb], alpha = 1) {
    x1 *= S; y1 *= S; x2 *= S; y2 *= S; width *= S;
    const minX = Math.max(0, Math.floor(Math.min(x1, x2) - width - 1));
    const maxX = Math.min(W - 1, Math.ceil(Math.max(x1, x2) + width + 1));
    const minY = Math.max(0, Math.floor(Math.min(y1, y2) - width - 1));
    const maxY = Math.min(H - 1, Math.ceil(Math.max(y1, y2) + width + 1));
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len2 = dx * dx + dy * dy || 1;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const t = Math.min(1, Math.max(0, ((x + 0.5 - x1) * dx + (y + 0.5 - y1) * dy) / len2));
        const d = Math.hypot(x + 0.5 - (x1 + t * dx), y + 0.5 - (y1 + t * dy));
        const cov = Math.min(1, Math.max(0, width / 2 + 0.5 - d));
        if (cov > 0) this.blend((y * W + x) * 3, cr, cg, cb, cov * alpha);
      }
    }
  }
  rect(x, y, w, h, [cr, cg, cb], alpha = 1) {
    const x0 = Math.max(0, Math.round(x * S));
    const x1 = Math.min(W, Math.round((x + w) * S));
    const y0 = Math.max(0, Math.round(y * S));
    const y1 = Math.min(H, Math.round((y + h) * S));
    for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) this.blend((yy * W + xx) * 3, cr, cg, cb, alpha);
  }
}

function makeBackground(kind) {
  const bg = new Uint8ClampedArray(W * H * 3);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 3;
      let n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
      n = (n - Math.floor(n) - 0.5) * 5;
      const g = kind === "table" ? 30 + (y / H) * 6 : 24 + (y / H) * 8;
      bg[i] = g - 2 + n;
      bg[i + 1] = g + 3 + n;
      bg[i + 2] = g + 9 + n;
    }
  }
  return bg;
}

/* --------------------------------- ffmpeg ---------------------------------- */

function encoder(file, fps, crf = 30) {
  const proc = spawn(
    "ffmpeg",
    ["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", `${W}x${H}`, "-r", String(fps), "-i", "-",
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", String(crf), "-preset", "slow", "-movflags", "+faststart", "-an", file],
    { stdio: ["pipe", "inherit", "inherit"] },
  );
  const done = new Promise((resolve, reject) => {
    proc.on("close", (c) => (c === 0 ? resolve() : reject(new Error(`ffmpeg exited with ${c} for ${file}`))));
  });
  return {
    async write(buf) {
      if (!proc.stdin.write(buf)) await new Promise((r) => proc.stdin.once("drain", r));
    },
    async end() {
      proc.stdin.end();
      await done;
    },
  };
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: "inherit" });
    p.on("close", (c) => (c === 0 ? resolve() : reject(new Error(`${cmd} failed`))));
  });
}

async function render(name, { fps, seconds, bgKind, drawScene, stillOnly }) {
  const r = new Raster(makeBackground(bgKind));
  const enc = encoder(path.join(mock, `${name}.mp4`), fps);
  const n = Math.round(seconds * fps);
  for (let i = 0; i < n; i++) {
    r.reset();
    drawScene(r, i / fps, i);
    await enc.write(Buffer.from(r.px.buffer.slice(0)));
  }
  await enc.end();
  void stillOnly;
}

async function posters(name, { bgKind, drawScene, t = 0, fps = 30, file = `${name}-poster.jpg`, dir = mock }) {
  const r = new Raster(makeBackground(bgKind));
  r.reset();
  drawScene(r, t, Math.round(t * fps));
  // Write a raw frame to a temp file and let ffmpeg turn it into a JPEG.
  const tmp = path.join(mock, `.${name}.rgb`);
  const { writeFileSync, unlinkSync } = await import("node:fs");
  writeFileSync(tmp, Buffer.from(r.px.buffer.slice(0)));
  await run("ffmpeg", ["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", `${W}x${H}`, "-i", tmp, "-frames:v", "1", "-q:v", "4", path.join(dir, file)]);
  unlinkSync(tmp);
}

/* --------------------------------- physics --------------------------------- */

function pendulumPath(theta0, gamma, L, g, seconds, fps) {
  // RK4 at 1/240 s; returns theta at each video frame.
  const dt = 1 / 240;
  const out = [];
  let th = theta0;
  let om = 0;
  const f = (t, o) => [o, -(g / L) * Math.sin(t) - gamma * o];
  const steps = Math.round(seconds / dt);
  const every = 240 / fps;
  for (let i = 0; i <= steps; i++) {
    if (i % every === 0) out.push(th);
    const k1 = f(th, om);
    const k2 = f(th + 0.5 * dt * k1[0], om + 0.5 * dt * k1[1]);
    const k3 = f(th + 0.5 * dt * k2[0], om + 0.5 * dt * k2[1]);
    const k4 = f(th + dt * k3[0], om + dt * k3[1]);
    th += (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    om += (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
  }
  return out;
}

/* ---------------------------------- scenes --------------------------------- */

const STEEL = [150, 168, 186];
const CYAN = [34, 211, 238];
const SHEET = [226, 222, 208];
const PIVOT = [480, 70];
const PPM = 600; // pixels per meter in the pendulum scenes

function pendulumScene({ L, thetas, ghost, trail = true, bobR = 13, still = false }) {
  return (r, t, frame) => {
    // Sheet taped to the wall behind the swing, then the clamp the string hangs from.
    r.rect(70, 330, 126, 178, SHEET, 0.92);
    r.rect(445, 40, 70, 22, [70, 78, 90]);
    r.rect(470, 56, 20, 10, [100, 110, 124]);
    if (still) return;
    const i = Math.min(thetas.length - 1, frame);
    const pos = (th) => [PIVOT[0] + PPM * L * Math.sin(th), PIVOT[1] + PPM * L * Math.cos(th)];
    if (trail) {
      for (let k = Math.max(0, i - 15); k < i; k++) {
        const a = pos(thetas[k]);
        const b = pos(thetas[k + 1]);
        r.line(a[0], a[1], b[0], b[1], 2.2, [235, 240, 245], 0.15 + 0.5 * ((k - (i - 15)) / 15));
      }
    }
    const [x, y] = pos(thetas[i]);
    r.line(PIVOT[0], PIVOT[1] + 5, x, y, 2, [175, 182, 190], 0.9);
    r.disc(x, y, bobR, STEEL, 1, true);
    if (ghost) {
      const gi = Math.min(ghost.length - 1, frame);
      const [gx, gy] = pos(ghost[gi]);
      r.line(PIVOT[0], PIVOT[1] + 5, gx, gy, 1.5, CYAN, 0.28);
      r.glow(gx, gy, bobR * 2.8, CYAN, 0.4);
      r.disc(gx, gy, bobR, CYAN, 0.55);
    }
  };
}

function spinnerScene(phis, ghost) {
  const hub = [480, 270];
  return (r, t, frame) => {
    r.rect(690, 350, 220, 156, SHEET, 0.9);
    const phi = phis[Math.min(phis.length - 1, frame)];
    for (let k = 0; k < 3; k++) {
      const a = phi + (k * 2 * Math.PI) / 3;
      const lx = hub[0] + 70 * Math.cos(a);
      const ly = hub[1] + 70 * Math.sin(a);
      r.line(hub[0], hub[1], lx, ly, 26, [66, 72, 82], 1);
      r.disc(lx, ly, 34, [74, 80, 92], 1, true);
    }
    r.disc(hub[0], hub[1], 26, [96, 102, 112], 1, true);
    r.disc(hub[0], hub[1], 9, [30, 34, 40], 1);
    const mx = hub[0] + 96 * Math.cos(phi);
    const my = hub[1] + 96 * Math.sin(phi);
    r.disc(mx, my, 10, [60, 140, 255], 1, true);
    void ghost;
  };
}

function dropScene({ ys, ghostYs, ballR = 11 }) {
  const X = 640;
  const floorY = 500;
  const sy = (y) => floorY - 350 * y;
  return (r, t, frame) => {
    r.rect(500, 396, 73, 104, SHEET, 0.92);
    r.rect(0, floorY, 960, 40, [44, 50, 58]);
    r.line(0, floorY, 960, floorY, 2, [90, 100, 112], 1);
    const i = Math.min(ys.length - 1, frame);
    if (ghostYs) {
      const gi = Math.min(ghostYs.length - 1, frame);
      r.glow(X, sy(ghostYs[gi]), ballR * 3.2, CYAN, 0.45);
      r.disc(X, sy(ghostYs[gi]), ballR + 3, CYAN, 0.5);
    }
    r.disc(X, sy(ys[i]), ballR, STEEL, 1, true);
  };
}

/* ----------------------------------- jobs ----------------------------------- */

const p = sc.pendulum;
const L = p.truth.L_eff;
const g = p.truth.g;
const pend1 = pendulumPath(p.truth.theta0_deg * DEG, p.truth.gamma, L, g, 6.5, 30);
const pend2Truth = pendulumPath(p.trial2.theta0_deg * DEG, p.truth.gamma, L, g, p.trial2.window_s, 30);
const pend2V1 = pendulumPath(p.trial2.theta0_deg * DEG, 0, L, g, p.trial2.window_s, 30);
const pend2V2 = pendulumPath(p.trial2.theta0_deg * DEG, p.truth.gamma * 1.14, L, g, p.trial2.window_s, 30);
const heroTruth = pendulumPath(40 * DEG, 0.03, L, g, 6, 30);
const heroGhost = pendulumPath(40 * DEG, 0.0, L, g, 6, 30);

const spin = sc.spin;
const spinFps = 30;
const spinPhi = Array.from({ length: Math.round(spin.window_s * spinFps) + 1 }, (_, i) => {
  const t = i / spinFps;
  const { c, b, omega0 } = spin.truth;
  const tStop = Math.log(1 + (b * omega0) / c) / b;
  const tt = Math.min(t, tStop);
  const r = c / b;
  return ((omega0 + r) / b) * (1 - Math.exp(-b * tt)) - r * tt;
});

const ff = sc.free_fall;
const dropAt = (t, y0, bm) => {
  const lam = bm;
  const vt = g / lam;
  return y0 - vt * t + ((vt) * (1 - Math.exp(-lam * t))) / lam;
};
const dropReal = Array.from({ length: Math.round(0.6 * 60) + 1 }, (_, i) => Math.max(0.02, dropAt(i / 60, ff.truth.y0, ff.truth.b_over_m)));
const dropT2 = Array.from({ length: Math.round(0.6 * 240) + 1 }, (_, i) => Math.max(0.02, dropAt(i / 240, ff.trial2.y0, ff.truth.b_over_m)));
const dropGhost = Array.from({ length: Math.round(0.6 * 240) + 1 }, (_, i) => Math.max(0.02, ff.trial2.y0 - 0.5 * g * (i / 240) ** 2));
const slowPick = (arr, ratio) => arr.filter((_, i) => i % ratio === 0);

const jobs = [
  ["pendulum-t1", { fps: 30, seconds: 6.5, bgKind: "wall", drawScene: pendulumScene({ L, thetas: pend1 }) }],
  ["pendulum-ghost-v1", { fps: 30, seconds: p.trial2.window_s, bgKind: "wall", drawScene: pendulumScene({ L, thetas: pend2Truth, ghost: pend2V1 }) }],
  ["pendulum-ghost-v2", { fps: 30, seconds: p.trial2.window_s, bgKind: "wall", drawScene: pendulumScene({ L, thetas: pend2Truth, ghost: pend2V2 }) }],
  ["spin-t1", { fps: spinFps, seconds: spin.window_s, bgKind: "table", drawScene: spinnerScene(spinPhi) }],
  ["drop-t1", { fps: 60, seconds: 0.6, bgKind: "wall", drawScene: dropScene({ ys: dropReal }) }],
  ["drop-ghost", { fps: 60, seconds: 0.6, bgKind: "wall", drawScene: dropScene({ ys: slowPick(dropT2, 4), ghostYs: slowPick(dropGhost, 4) }) }],
  ["drop-ghost-slow", { fps: 30, seconds: 4.8, bgKind: "wall", drawScene: dropScene({ ys: dropT2, ghostYs: dropGhost }) }],
];

for (const [name, spec] of jobs) {
  console.log("rendering", name);
  await render(name, spec);
}

// The landing page hero loop and its poster live at the site root.
console.log("rendering hero");
const hero = new Raster(makeBackground("wall"));
const heroScene = pendulumScene({ L, thetas: heroTruth, ghost: heroGhost });
{
  const enc = encoder(path.join(pub, "hero-ghost.mp4"), 30);
  for (let i = 0; i < 180; i++) {
    hero.reset();
    heroScene(hero, i / 30, i);
    await enc.write(Buffer.from(hero.px.buffer.slice(0)));
  }
  await enc.end();
}

// Posters: the object in its starting position (thumbnails) and the empty scene (the what-if backdrop).
await posters("pendulum", { bgKind: "wall", drawScene: pendulumScene({ L, thetas: pend1, ghost: pend2V1.map((th, i) => th * 0.9 + 0.0 * i) }), t: 0.3 });
await posters("pendulum-still", { bgKind: "wall", drawScene: pendulumScene({ L, thetas: pend1, still: true }), file: "pendulum-still.jpg" });
await posters("spin", { bgKind: "table", drawScene: spinnerScene(spinPhi), t: 0.7, fps: 30 });
await posters("spin-still", { bgKind: "table", drawScene: (r) => { r.rect(690, 350, 220, 156, SHEET, 0.9); r.disc(480, 270, 9, [30, 34, 40], 1); }, file: "spin-still.jpg" });
await posters("drop", { bgKind: "wall", drawScene: dropScene({ ys: dropReal, ghostYs: dropReal.map((y) => y + 0.02) }), t: 0.3, fps: 60 });
await posters("drop-still", { bgKind: "wall", drawScene: dropScene({ ys: [-10] }), file: "drop-still.jpg" });
await posters("hero", { bgKind: "wall", drawScene: heroScene, t: 1.2, dir: pub, file: "hero-poster.jpg" });
console.log("done");
