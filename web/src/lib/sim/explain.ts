import type { ControlValues, Readout, Scenario } from "./scenario";
import type { SimContext } from "./types";

/**
 * Instant explanations from formulas (build-plan Part 18): no model call, just the algebra
 * of the equations that were fitted. Each string explains one thing the user changed.
 */

function ratioWords(r: number, more: string, less: string): string {
  if (!Number.isFinite(r) || r <= 0) return "";
  if (Math.abs(r - 1) < 0.005) return "unchanged";
  return r > 1 ? `${r.toFixed(2)}× ${more}` : `${(1 / r).toFixed(2)}× ${less}`;
}

function changed(a: ControlValues, b: ControlValues, id: string, tol = 1e-6): boolean {
  const x = a[id];
  const y = b[id];
  if (x === undefined || y === undefined) return false;
  if (typeof x === "number" && typeof y === "number") return Math.abs(x - y) > tol * Math.max(1, Math.abs(x));
  return x !== y;
}

function readout(list: Readout[], id: string): number | null {
  return list.find((r) => r.id === id)?.value ?? null;
}

function sig(x: number, d = 3): string {
  return Number(x.toPrecision(d)).toString();
}

export function instantExplanations(
  ctx: SimContext,
  base: ControlValues,
  what: ControlValues,
  baseSc: Scenario,
  whatSc: Scenario,
  baseRead: Readout[],
  whatRead: Readout[],
): string[] {
  const out: string[] = [];
  const gB = baseSc.params.g;
  const gW = whatSc.params.g;

  switch (ctx.system) {
    case "pendulum": {
      const LB = baseSc.params.L_eff;
      const LW = whatSc.params.L_eff;
      if (changed(base, what, "g")) {
        const r = Math.sqrt(gB / gW);
        out.push(
          `T ∝ 1/√g, so with g = ${sig(gW)} m/s² the period is ${ratioWords(r, "longer", "shorter")}.`,
        );
      }
      if (changed(base, what, "length_cm")) {
        const r = Math.sqrt(LW / LB);
        out.push(`T ∝ √L, so a ${(LW * 100).toFixed(0)} cm string gives a period ${ratioWords(r, "longer", "shorter")}.`);
      }
      if (changed(base, what, "mass")) {
        const m = Number(what.mass);
        out.push(
          `Mass does not change the period, because it cancels out of the equation of motion. It only changes how fast the swing dies away: damping is b/m, so a bob ${sig(m, 2)}× as heavy loses energy ${sig(1 / m, 2)}× as fast.`,
        );
      }
      if (changed(base, what, "amplitude_deg")) {
        out.push(
          "Bigger swings take slightly longer, because sin θ is a little smaller than θ. The small-angle formula T = 2π√(L/g) is only exact as the amplitude shrinks to zero.",
        );
      }
      if (changed(base, what, "air")) {
        const hb = readout(baseRead, "half_life");
        const hw = readout(whatRead, "half_life");
        const tail = hb && hw ? ` The swing halves in ${sig(hw, 2)} s instead of ${sig(hb, 2)} s.` : "";
        out.push(`Damping scales with air density, so the swing decays ${Number(what.air) < Number(base.air) ? "more slowly" : "faster"}.${tail}`);
      }
      break;
    }
    case "free_fall": {
      if (changed(base, what, "g")) {
        out.push(`With no drag the fall time is √(2h/g), so it is ${ratioWords(Math.sqrt(gB / gW), "longer", "shorter")}.`);
      }
      if (changed(base, what, "height_m")) {
        out.push("Fall time grows as √h: four times the height takes only twice as long.");
      }
      if (changed(base, what, "air")) {
        out.push(
          "With quadratic drag the terminal velocity is √(g / (k/m)), and k scales with air density, so thinner air raises it. In a vacuum every object falls at g.",
        );
      }
      if (changed(base, what, "mass")) {
        out.push("A heavier object of the same size feels the same drag force but has more inertia, so its terminal velocity rises.");
      }
      break;
    }
    case "ramp": {
      const aB = readout(baseRead, "accel");
      const aW = readout(whatRead, "accel");
      if (aB !== null && aW !== null && aB > 0 && changed(base, what, "angle_deg")) {
        out.push(`The pull along the slope is g sin α, so the acceleration is ${ratioWords(aW / aB, "larger", "smaller")} (${sig(aW, 3)} m/s²).`);
      }
      if (changed(base, what, "g")) {
        out.push(`Acceleration is proportional to g, so it is ${ratioWords(gW / gB, "larger", "smaller")}.`);
      }
      if (changed(base, what, "mu_k")) {
        out.push(
          aW !== null && aW <= 0
            ? "Friction now beats the pull of gravity along the slope (μ > tan α), so the block never starts to slide."
            : "Kinetic friction takes μ g cos α off the acceleration, so more friction means a slower slide.",
        );
      }
      if (changed(base, what, "shape")) {
        out.push(
          "A rolling body spends part of its energy spinning, so a = g sin α / (1 + β). β is 2/5 for a solid ball, so it gets 5/7 of the sliding acceleration. A hollow ball (β = 2/3) is slower still.",
        );
      }
      break;
    }
    case "bounce": {
      if (changed(base, what, "e")) {
        out.push(`Each bounce keeps a fraction e of the speed, so the rebound height is e² times the previous one: e = ${sig(Number(what.e), 2)} gives ${sig(Number(what.e) ** 2, 2)}× per bounce.`);
      }
      if (changed(base, what, "height_m")) {
        out.push("Bounce heights are in proportion to the drop height; the number of bounces before rest barely changes.");
      }
      if (changed(base, what, "g")) {
        out.push("Gravity sets the time scale (√(h/g)) but not the ratio of one bounce height to the next.");
      }
      break;
    }
    case "spin_down": {
      if (changed(base, what, "g")) {
        out.push(
          `Bearing friction scales with g, so the Coulomb term is ${ratioWords(gW / gB, "larger", "smaller")}. This assumes the load on the bearing is the spinner's own weight.`,
        );
      }
      if (changed(base, what, "spin_rev_s")) {
        const sb = readout(baseRead, "stop");
        const sw = readout(whatRead, "stop");
        out.push(
          sb && sw
            ? `A faster start takes longer to stop (${sig(sw, 3)} s instead of ${sig(sb, 3)} s), but not in proportion: friction that is proportional to speed removes more per second at high spin.`
            : "A faster start takes longer to stop.",
        );
      }
      if (changed(base, what, "air")) {
        out.push("Air drag falls with density, so in thinner air the spinner coasts a little longer.");
      }
      break;
    }
  }
  if (out.length === 0) out.push("Move a slider to see how the law you measured reacts.");
  return out;
}
