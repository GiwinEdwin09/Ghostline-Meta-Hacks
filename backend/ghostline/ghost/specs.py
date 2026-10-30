"""Model specs for the ghost: library models and Wonder-written ones, and what the ghost needs to
know about them (observables, parameter kinds) without running untrusted code in this process.
"""

from __future__ import annotations

from dataclasses import dataclass

from ghostline.physics.harness.schema import ModelSpec
from ghostline.physics.library import ModelDef, ParamSpec, get_model
from ghostline.store.records import ModelRecord


@dataclass
class ModelInfo:
    """What the ghost reads off a model. `instance` is set for library models only: custom code
    is never built outside the Sandbox, so its `prepare` step is taken to be the identity."""

    id: str
    observables: tuple[str, ...]
    params: dict[str, ParamSpec]
    instance: ModelDef | None

    @property
    def initial_names(self) -> list[str]:
        return [n for n, s in self.params.items() if s.kind == "initial_condition"]

    @property
    def physical_names(self) -> list[str]:
        return [n for n, s in self.params.items() if s.kind == "physical"]

    @property
    def frozen_names(self) -> list[str]:
        """Parameters held fixed for trial 2: the law, and the setup's nuisance offsets."""
        return [n for n, s in self.params.items() if s.kind != "initial_condition"]


def library_spec(model_id: str) -> ModelSpec:
    return ModelSpec(id=model_id, library_ref=model_id)


def spec_from_record(record: ModelRecord) -> ModelSpec:
    """A harness spec for a model row: a library reference, or its code with its parameter spec."""
    if record.library_ref:
        return ModelSpec(
            id=record.id, library_ref=record.library_ref, name=record.name,
            mechanism=record.mechanism, equations_latex=record.equations_latex,
        )  # fmt: skip
    params = {k: ParamSpec(**v.model_dump()) for k, v in (record.params_spec or {}).items()}
    return ModelSpec(
        id=record.id, code=record.code, name=record.name, mechanism=record.mechanism,
        equations_latex=record.equations_latex, params=params,
    )  # fmt: skip


def model_info(spec: ModelSpec, system: str, default_observables: tuple[str, ...]) -> ModelInfo:
    if spec.library_ref is not None:
        model = get_model(spec.library_ref, system)
        return ModelInfo(spec.id, tuple(model.observables), dict(model.params), model)
    observables = tuple(spec.observables or default_observables)
    return ModelInfo(spec.id, observables, dict(spec.params or {}), None)


def with_observables(spec: ModelSpec, observables: tuple[str, ...]) -> ModelSpec:
    """A custom spec with its observables pinned, so a harness job without data can build it."""
    if spec.code is None or spec.observables:
        return spec
    return spec.model_copy(update={"observables": list(observables)})
