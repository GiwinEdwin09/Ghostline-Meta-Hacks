/**
 * Maps whatever phase string the back end reports onto a step of the stepper. The plan names
 * the user-facing steps (Normalizing, Watching, Tracking, Calibrating, Hypothesizing, Fitting,
 * Peer review) but the orchestrator's state machine uses its own words (SCENE, MEASURE,
 * HYPOTHESIZE, FIT, ANALYZE, REVISE, CONCLUDE, REVIEW, RESPOND, FINAL), so accept both.
 */
const ALIASES: Record<string, string> = {
  ingest: "normalize",
  normalizing: "normalize",
  scene: "watch",
  watching: "watch",
  tracking: "track",
  measure: "track",
  sam: "track",
  calibrating: "calibrate",
  hypothesizing: "hypothesize",
  fitting: "fit",
  analyze: "fit",
  revise: "fit",
  conclude: "fit",
  respond: "review",
  final: "review",
  reviewing: "review",
  prediction: "predict",
  ghost: "render",
  rendering: "render",
};

export function phaseToStep(phase: string, phases: readonly string[]): number {
  const key = phase.toLowerCase();
  const name = ALIASES[key] ?? key;
  const i = phases.indexOf(name);
  return i >= 0 ? i : 0;
}
