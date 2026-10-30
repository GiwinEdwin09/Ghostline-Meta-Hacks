import type { Params, SimContext, SimSystem } from "./types";

/**
 * Names the mock data in `lib/mocks` used before the twins took the Python library's names
 * (build-plan Part 18). They are mapped to the library names at the entry points of this
 * folder, so mock mode keeps working. Delete this file once the mocks use `L`, `b`, `k` and
 * `vy0`.
 */
const RENAMES: Record<string, string> = { L_eff: "L", b_over_m: "b", k_over_m: "k" };

function hasLegacy(p: Params): boolean {
  return Object.keys(RENAMES).some((old) => old in p);
}

/** The parameters with legacy names replaced by the library's (the new name wins if both). */
export function modernParams(p: Params): Params {
  if (!hasLegacy(p)) return p;
  const out: Params = { ...p };
  for (const [old, name] of Object.entries(RENAMES)) {
    if (old in out) {
      if (!(name in out)) out[name] = out[old];
      delete out[old];
    }
  }
  return out;
}

/** Initial conditions with a free fall's `v0` renamed to the library's `vy0`. */
export function modernInit(system: SimSystem, init: Params): Params {
  if (system !== "free_fall" || !("v0" in init)) return init;
  const out: Params = { ...init };
  if (!("vy0" in out)) out.vy0 = out.v0;
  delete out.v0;
  return out;
}

export function modernContext(ctx: SimContext): SimContext {
  const params = modernParams(ctx.params);
  const fixed = modernParams(ctx.fixed);
  const initial = modernInit(ctx.system, ctx.initial);
  if (params === ctx.params && fixed === ctx.fixed && initial === ctx.initial) return ctx;
  return { ...ctx, params, fixed, initial };
}
