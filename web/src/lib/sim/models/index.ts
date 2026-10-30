import { modernInit, modernParams } from "../legacy";
import type { SimModel, SimSystem } from "../types";
import { B0, B1, B2 } from "./bounce";
import { FF0, FF1, FF2 } from "./free-fall";
import { P0, P1, P2, P3, P4 } from "./pendulum";
import { R0, R1, R2, R3 } from "./ramp";
import { S0, S1, S2, S3 } from "./spin";

/** The model, also accepting the legacy names of the mock data (see `legacy.ts`). */
function withLegacyNames(model: SimModel): SimModel {
  return {
    ...model,
    initialState: (init) => model.initialState(modernInit(model.system, init)),
    closedForm: model.closedForm
      ? (t, p, init) => model.closedForm!(t, modernParams(p), modernInit(model.system, init))
      : undefined,
  };
}

const TWINS: Record<string, SimModel> = {
  FF0, FF1, FF2,
  P0, P1, P2, P3, P4,
  R0, R1, R2, R3,
  B0, B1, B2,
  S0, S1, S2, S3,
};

/** Every library model that has a TypeScript twin, keyed by its library id. */
export const MODELS: Record<string, SimModel> = Object.fromEntries(
  Object.entries(TWINS).map(([id, model]) => [id, withLegacyNames(model)]),
);

export function getModel(id: string): SimModel | undefined {
  return MODELS[id];
}

export function modelsFor(system: SimSystem): SimModel[] {
  return Object.values(MODELS).filter((m) => m.system === system);
}

export { B0, B1, B2, FF0, FF1, FF2, P0, P1, P2, P3, P4, R0, R1, R2, R3, S0, S1, S2, S3 };
