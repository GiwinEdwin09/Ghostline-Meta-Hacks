"""Turn Wonder-written code into a ModelDef, and dry-run it (build plan Part 11)."""

from __future__ import annotations

import builtins
import contextlib
import math
import signal
import threading
from collections.abc import Iterator

import numpy as np

from ..library import get_model
from ..library.base import ModelDef
from .safety import ALLOWED_MODULES, FORBIDDEN_ATTRS, check_code
from .schema import ModelSpec

CUSTOM_MODEL_CONTRACT = """\
A custom model is Python source that defines these module-level functions. `p` is a dict of
floats holding every parameter in the spec and every constant the model declares. `s` is the
state vector, ordered as in `state`. Time starts at 0 at the first sample of the motion window.

    def initial_state(p): ...     # state at t = 0, a sequence of len(state) floats
    def rhs(t, s, p): ...         # ds/dt, a sequence of len(state) floats
    def observe(s): ...           # optional; observables from a state, default s[:len(observables)]
    def closed_form(t, p): ...    # optional; array of shape (len(observables), len(t)), skips rhs

Only `import math`, `import numpy as np` and `from scipy.special import ...` are allowed. No file
or network access, classes, `with` blocks, or names such as open, exec, eval, os and sys.
"""

_SAFE_BUILTIN_NAMES = (
    "abs all any bool complex dict divmod enumerate filter float int isinstance len list map max "
    "min pow range reversed round set slice sorted sum tuple zip ArithmeticError "
    "ZeroDivisionError ValueError Exception True False None"
).split()

DEFAULT_UNITS = {"x": "m", "y": "m", "s": "m", "theta": "rad", "phi": "rad"}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    scipy_special = name == "scipy" and set(fromlist or ()) <= {"special"}
    if level or (name not in ALLOWED_MODULES and not scipy_special):
        raise ImportError(f"import of {name!r} is not allowed")
    for item in fromlist or ():
        if item in FORBIDDEN_ATTRS or item.startswith("__"):
            raise ImportError(f"importing {item!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _namespace() -> dict:
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    safe["__import__"] = _safe_import
    return {"__builtins__": safe, "__name__": "wonder_model", "math": math, "np": np}


@contextlib.contextmanager
def time_limit(seconds: float) -> Iterator[None]:
    """Raise TimeoutError in the calling thread after `seconds`, where SIGALRM is available."""
    usable = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError(f"model code ran longer than {seconds:g} s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class CustomModel(ModelDef):
    """A model whose equations come from untrusted code that passed `check_code`."""

    def __init__(self, spec: ModelSpec, system: str, default_observables: tuple[str, ...]):
        errors = check_code(spec.code or "")
        if errors:
            raise ValueError("; ".join(errors))
        self.id = spec.id
        self.name = spec.name or spec.id
        self.system = system
        self.mechanism = spec.mechanism
        self.equations_latex = spec.equations_latex
        self.observables = tuple(spec.observables or default_observables)
        self.observable_units = {o: DEFAULT_UNITS.get(o, "") for o in self.observables}
        self.state = tuple(spec.state or self.observables)
        self.params = {k: v.model_copy() for k, v in (spec.params or {}).items()}
        self.constants = tuple(spec.constants)
        namespace = _namespace()
        with time_limit(5.0):
            exec(compile(spec.code or "", f"<model {spec.id}>", "exec"), namespace)  # noqa: S102
        self._initial = namespace.get("initial_state")
        self._rhs = namespace.get("rhs")
        self._closed = namespace.get("closed_form")
        self._observe = namespace.get("observe")

    def initial_state(self, p):
        return np.asarray(self._initial(p), dtype=float)

    def rhs(self, t, s, p):
        return self._rhs(t, s, p)

    def closed_form(self, tau, p):
        if self._closed is None:
            return None
        out = np.asarray(self._closed(tau, p), dtype=float)
        return out[None, :] if out.ndim == 1 else out

    def observe(self, s):
        s = np.asarray(s, dtype=float)
        if self._observe is None:
            return s[: len(self.observables)]
        if s.ndim == 1:
            return np.asarray(self._observe(s), dtype=float)
        columns = [np.asarray(self._observe(s[:, i]), dtype=float) for i in range(s.shape[1])]
        return np.array(columns).T

    def simulate(self, tau, p):
        if self._rhs is None or self._initial is None:
            raise ValueError("model has no rhs/initial_state and no closed_form")
        return super().simulate(tau, p)


def build_model(spec: ModelSpec, system: str, default_observables: tuple[str, ...]) -> ModelDef:
    if spec.library_ref is not None:
        return get_model(spec.library_ref, system)
    return CustomModel(spec, system, default_observables)


def dry_run(model: ModelDef, constants: dict[str, float], limit_s: float = 10.0) -> list[str]:
    """Run the model once on its initial parameters and report shape, NaN and error problems."""
    missing = model.missing_constants(constants)
    if missing:
        return [f"constants {missing} are used by the model but not provided"]
    p = {**constants, **{k: float(v.init) for k, v in model.params.items()}}
    n_obs, n_state = len(model.observables), len(model.state)
    tau = np.linspace(0.0, 1.0, 21)
    errors: list[str] = []

    def check(label: str, value, shape: tuple[int, ...]) -> None:
        arr = np.asarray(value, dtype=float)
        if arr.shape != shape:
            errors.append(f"{label} returned shape {arr.shape}, expected {shape}")
        elif not np.all(np.isfinite(arr)):
            errors.append(f"{label} returned NaN or inf")

    try:
        with time_limit(limit_s):
            if isinstance(model, CustomModel):
                if model._initial is not None:
                    s0 = model.initial_state(p)
                    check("initial_state", s0, (n_state,))
                    if not errors and model._rhs is not None:
                        check("rhs", model.rhs(0.0, s0, p), (n_state,))
                if model._closed is not None:
                    check("closed_form", model.closed_form(tau, p), (n_obs, tau.size))
            if not errors:
                check("the prediction", model.predict(tau, p), (n_obs, tau.size))
    except TimeoutError as exc:
        errors.append(str(exc))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def spec_issues(spec: ModelSpec) -> list[str]:
    """Static problems with a custom spec: code checks plus parameter-name clashes."""
    if spec.code is None:
        return []
    issues = check_code(spec.code)
    clash = set(spec.params or {}) & set(spec.constants)
    if clash:
        issues.append(f"names used as both parameter and constant: {sorted(clash)}")
    return issues
