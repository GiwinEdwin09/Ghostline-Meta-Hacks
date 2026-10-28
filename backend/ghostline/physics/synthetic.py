"""Synthetic datasets with known truth, for tests and the recovery suite (build plan Part 26)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .harness.schema import FitJob, ModelSpec, SeriesData
from .library import get_model


@dataclass
class SyntheticCase:
    """A noisy series from a library model with known parameters."""

    system: str
    model_id: str
    params: dict[str, float]
    constants: dict[str, float]
    t: np.ndarray
    observed: dict[str, np.ndarray]
    sigma: dict[str, float]
    truth: dict[str, float] = field(default_factory=dict)
    segment_id: np.ndarray | None = None

    def series(self) -> SeriesData:
        return SeriesData(
            t=self.t.tolist(),
            series={k: v.tolist() for k, v in self.observed.items()},
            sigma=self.sigma,
            segment_id=None if self.segment_id is None else self.segment_id.tolist(),
        )

    def job(self, model_ids: list[str], mode: str = "select", **kwargs) -> FitJob:
        return FitJob(
            mode=mode,  # type: ignore[arg-type]
            system=self.system,
            data=self.series(),
            constants=self.constants,
            models=[ModelSpec(id=i, library_ref=i) for i in model_ids],
            **kwargs,
        )


def make_case(
    system: str,
    model_id: str,
    params: dict[str, float],
    constants: dict[str, float],
    t: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    *,
    impact_gap: int = 2,
) -> SyntheticCase:
    """Simulate `model_id` at times t (starting at t[0]) and add Gaussian noise."""
    model = get_model(model_id, system)
    p = {**constants, **params}
    clean = model.predict(t - t[0], p)
    noisy = clean + rng.normal(0.0, sigma, clean.shape)
    observed = {o: noisy[i] for i, o in enumerate(model.observables)}
    segment = None
    if model.by_flight:
        # Flights are cut at the floor, using the noise-free curve, like the trajectory stage.
        floor = p["y_floor"]
        near = clean[0] < floor + 0.02
        labels = np.cumsum(np.concatenate([[0], (near[1:] & ~near[:-1]).astype(int)]))
        keep = np.ones(t.size, dtype=bool)
        for i in np.where(near)[0]:
            keep[max(i - impact_gap, 0) : i + impact_gap + 1] = False
        t, observed = t[keep], {k: v[keep] for k, v in observed.items()}
        segment = labels[keep]
    return SyntheticCase(
        system=system,
        model_id=model_id,
        params=params,
        constants=constants,
        t=t,
        observed=observed,
        sigma={o: sigma for o in model.observables},
        truth={**params, **model.derived(p)},
        segment_id=segment,
    )
