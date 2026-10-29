"""Job and result JSON for the fitting harness (build plan Part 11).

`job.json` goes in, `result.json` comes out. The same models run in-process for tests and inside
the Modal Sandbox, so everything here is plain JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..library.base import ParamSpec

MAX_RESULT_BYTES = 5_000_000
MAX_PLOTS_PER_MODEL = 4


class SeriesData(BaseModel):
    """The motion window: times, one series per observable, noise and quality weights."""

    t: list[float]
    series: dict[str, list[float | None]]
    sigma: dict[str, float]
    weights: list[float] | None = None
    segment_id: list[int] | None = None

    @model_validator(mode="after")
    def _aligned(self) -> SeriesData:
        n = len(self.t)
        for name, values in self.series.items():
            if len(values) != n:
                raise ValueError(f"series {name!r} has {len(values)} values, expected {n}")
            if name not in self.sigma or not self.sigma[name] > 0:
                raise ValueError(f"sigma for {name!r} must be positive")
        for name, extra in (("weights", self.weights), ("segment_id", self.segment_id)):
            if extra is not None and len(extra) != n:
                raise ValueError(f"{name} has {len(extra)} values, expected {n}")
        return self


class ModelSpec(BaseModel):
    """A library reference, or Wonder-written code with its parameter spec."""

    id: str
    library_ref: str | None = None
    code: str | None = None
    name: str = ""
    mechanism: str = ""
    equations_latex: str = ""
    params: dict[str, ParamSpec] | None = None
    observables: list[str] | None = None
    state: list[str] | None = None
    constants: list[str] = Field(default_factory=list)
    fixed: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_source(self) -> ModelSpec:
        if (self.library_ref is None) == (self.code is None):
            raise ValueError("give exactly one of library_ref or code")
        if self.code is not None and not self.params:
            raise ValueError("custom models need a params spec")
        return self


class Perturbation(BaseModel):
    """One calibration/timing draw: lengths x scale, times x time_scale, constants overridden.

    `eps` echoes the draw's components for the variance decomposition; the harness ignores it.
    """

    scale: float = 1.0
    time_scale: float = 1.0
    constants: dict[str, float] = Field(default_factory=dict)
    eps: dict[str, float] = Field(default_factory=dict)


class BootstrapSpec(BaseModel):
    n: int = 0
    block: int = 10
    seed: int = 0


class SimulateSpec(BaseModel):
    times: list[float]
    params: dict[str, dict[str, float]]


class FitJob(BaseModel):
    job_id: str = "job"
    mode: Literal["validate", "select", "final", "simulate"]
    system: str
    data: SeriesData | None = None
    constants: dict[str, float] = Field(default_factory=dict)
    models: list[ModelSpec]
    splits: list[float] = Field(default_factory=lambda: [0.65, 0.5, 0.8])
    primary_split: float = 0.65
    n_starts: int = 8
    forecast_draws: int = 16
    bootstrap: BootstrapSpec = Field(default_factory=BootstrapSpec)
    perturbations: list[Perturbation] = Field(default_factory=list)
    simulate: SimulateSpec | None = None
    plots: bool = True
    time_limit_s: float = 30.0
    workers: int = 1


class ParamEstimate(BaseModel):
    name: str
    value: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    unit: str = ""
    kind: str = "physical"
    at_bound: bool = False
    derived: bool = False


class Diagnostics(BaseModel):
    n: int
    n_params: int
    rmse: float
    chi2: float
    chi2_red: float
    lag1: float | None = None
    durbin_watson: float | None = None
    runs_p: float | None = None
    systematic: bool = False
    systematic_r2: float = 0.0
    condition: float | None = None


class SplitFit(BaseModel):
    split: float
    t_split: float
    n_train: int
    n_test: int
    converged: bool
    params: list[ParamEstimate]
    train: Diagnostics
    heldout_rmse: float | None = None
    heldout_errors: list[float] | None = None
    heldout_draws: list[list[float]] | None = None  # forecast errors under parameter draws
    t_test: list[float] | None = None


class PlotRef(BaseModel):
    id: str
    path: str
    kind: Literal["fit", "resid", "fit_train", "resid_train"]
    audience: Literal["agent", "reviewer"]


class ModelResult(BaseModel):
    id: str
    status: Literal["ok", "failed", "timeout", "invalid"]
    error: str | None = None
    library_ref: str | None = None
    n_physical: int = 0
    splits: list[SplitFit] = Field(default_factory=list)
    full: SplitFit | None = None
    aic: float | None = None
    bic: float | None = None
    bootstrap_requested: int = 0
    bootstrap_done: int = 0
    perturbations_requested: int = 0
    replicates: list[dict[str, float]] = Field(default_factory=list)
    replicate_eps: list[dict[str, float]] = Field(default_factory=list)
    plots: list[PlotRef] = Field(default_factory=list)
    elapsed_s: float = 0.0


class ValidationReport(BaseModel):
    model_id: str
    ok: bool
    errors: list[str] = Field(default_factory=list)


class FitResult(BaseModel):
    job_id: str
    mode: str
    system: str
    models: list[ModelResult] = Field(default_factory=list)
    validation: list[ValidationReport] = Field(default_factory=list)
    simulations: dict[str, dict[str, list[float]]] = Field(default_factory=dict)
    elapsed_s: float = 0.0
    versions: dict[str, str] = Field(default_factory=dict)

    def model(self, model_id: str) -> ModelResult:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)
