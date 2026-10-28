import { describe, expect, it } from "vitest";

import { getModel, integrate } from "@/lib/sim";
import fixtures from "./__fixtures__/parity.json";

/**
 * Parity with an independent integration (build-plan Part 18, "parity tests").
 *
 * `scripts/gen_sim_parity.py` integrates the same equations with scipy's DOP853 at 1e-12
 * tolerance and samples them on the simulator's grid. The TypeScript RK4 output has to agree
 * within 1e-6 relative to the size of the signal (relative to each sample would blow up at
 * zero crossings, which says nothing about accuracy).
 */

interface Fixture {
  model: string;
  params: Record<string, number>;
  init: Record<string, number>;
  t_end: number;
  dt: number;
  stride: number;
  obs: number[];
}

describe("parity with scipy DOP853", () => {
  for (const fx of fixtures as Fixture[]) {
    const label = `${fx.model} ${JSON.stringify(fx.params)}`;
    it(`${label} agrees within 1e-6`, () => {
      const model = getModel(fx.model);
      expect(model).toBeDefined();
      const res = integrate(model!, fx.params, fx.init, { tEnd: fx.t_end, dt: fx.dt });
      const scale = Math.max(...fx.obs.map((v) => Math.abs(v)), 1e-9);
      let worst = 0;
      fx.obs.forEach((expected, i) => {
        const got = res.obs[0][i * fx.stride];
        worst = Math.max(worst, Math.abs(got - expected) / scale);
      });
      expect(worst).toBeLessThan(1e-6);
    });
  }
});
