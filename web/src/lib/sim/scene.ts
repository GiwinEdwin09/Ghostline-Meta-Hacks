import type { Scenario } from "./scenario";
import type { SimSystem } from "./types";

/** A row-major 3x3 matrix. */
export type Mat3 = readonly number[];

/** Apply a homography to (x, y). */
export function applyH(H: Mat3, x: number, y: number): [number, number] {
  const u = H[0] * x + H[1] * y + H[2];
  const v = H[3] * x + H[4] * y + H[5];
  const w = H[6] * x + H[7] * y + H[8];
  return [u / w, v / w];
}

export interface BodyPose {
  /** World position in meters (x right, y up). */
  x: number;
  y: number;
  /** Rotation of the body (the swing angle, the rotor angle, ...). */
  angle: number;
}

/**
 * Where the simulated body is in the world, given a state vector and the anchor
 * (pivot, hub, floor point or top of the slope; see `WhatIfContext.anchor_m`).
 */
export function poseAt(
  system: SimSystem,
  sc: Scenario,
  state: readonly number[],
  anchor: readonly [number, number],
): BodyPose {
  const [ax, ay] = anchor;
  switch (system) {
    case "pendulum": {
      const L = sc.params.L_eff;
      return { x: ax + L * Math.sin(state[0]), y: ay - L * Math.cos(state[0]), angle: state[0] };
    }
    case "free_fall":
    case "bounce":
      return { x: ax, y: ay + state[0], angle: 0 };
    case "ramp": {
      const a = sc.params.alpha;
      return { x: ax + state[0] * Math.cos(a), y: ay - state[0] * Math.sin(a), angle: -a };
    }
    case "spin_down": {
      const r = sc.geometry.markerRadius ?? 0.03;
      return { x: ax + r * Math.cos(state[0]), y: ay + r * Math.sin(state[0]), angle: state[0] };
    }
  }
}

/** Fallback when the experiment has no calibration: fit the world box into the canvas. */
export function fitTransform(
  points: readonly [number, number][],
  width: number,
  height: number,
  margin = 0.12,
): (x: number, y: number) => [number, number] {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const w = Math.max(maxX - minX, 1e-3);
  const h = Math.max(maxY - minY, 1e-3);
  const scale = Math.min((width * (1 - 2 * margin)) / w, (height * (1 - 2 * margin)) / h);
  const ox = (width - w * scale) / 2 - minX * scale;
  const oy = (height - h * scale) / 2 + maxY * scale;
  return (x, y) => [ox + x * scale, oy - y * scale];
}
