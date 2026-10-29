"""Choosing where a fit job runs (build plan Part 11).

The plan runs every fit in a Modal Sandbox: no network, CPU and memory limits, one per analysis
job. Wonder-written code is only AST-checked on the way in, so the Sandbox is what makes it safe.
`physics.runner` and `physics.sandbox` give three ways to run a `FitJob`:

- `FitSandbox` (Modal): the real thing, used for every fit when this code runs inside Modal.
- `SubprocessRunner`: the same contract in a child process; no isolation, so only for local
  development of custom models.
- `InProcessRunner`: fast, for library models (trusted code) and tests.

`RunnerPool.run` picks per job. `GHOSTLINE_FIT_RUNNER=inprocess|subprocess|sandbox` forces one;
otherwise (auto) a job with custom code never runs in-process, and everything goes to the Sandbox
when running inside Modal. If the Sandbox can't start, a library-only job falls back to
in-process rather than failing the analysis; a job with custom code fails instead (it must not run
outside an isolated process). Tests install a factory with `use_runner_factory`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ghostline.physics.harness.schema import FitJob, FitResult
from ghostline.physics.runner import InProcessRunner, Runner, SubprocessRunner

log = logging.getLogger("ghostline.agent")

RunnerFactory = Callable[[Path], Runner]
_factory: RunnerFactory | None = None


@contextmanager
def use_runner_factory(factory: RunnerFactory | None) -> Iterator[None]:
    """Make every `RunnerPool` build its runner with `factory(plots_dir)`, regardless of the job.
    For tests (a runner that replays recorded results) and for pinning a runner locally."""
    global _factory
    previous, _factory = _factory, factory
    try:
        yield
    finally:
        _factory = previous


def inside_modal() -> bool:
    return bool(os.environ.get("MODAL_IS_REMOTE") or os.environ.get("MODAL_TASK_ID"))


class RunnerPool:
    """Runs fit jobs for one stage call and cleans up after itself. Use as a context manager."""

    def __init__(self, kind: str | None = None) -> None:
        self.kind = (kind or os.environ.get("GHOSTLINE_FIT_RUNNER") or "auto").lower()
        self._tmp = Path(tempfile.mkdtemp(prefix="ghostline-fits-"))
        self._runners: dict[str, Runner] = {}
        self._sandbox = None

    def __enter__(self) -> RunnerPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.close()
            except Exception:
                log.warning("could not terminate the fit sandbox", exc_info=True)
            self._sandbox = None
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _choose(self, job: FitJob) -> str:
        custom = any(spec.code is not None for spec in job.models)
        if self.kind != "auto":
            if self.kind == "inprocess" and custom:
                return "subprocess"  # untrusted code never runs in this process
            return self.kind
        if inside_modal():
            return "sandbox"
        return "subprocess" if custom else "inprocess"

    def _runner(self, kind: str) -> Runner:
        if kind not in self._runners:
            plots = self._tmp / kind
            plots.mkdir(parents=True, exist_ok=True)
            if _factory is not None:
                self._runners[kind] = _factory(plots)
            elif kind == "inprocess":
                self._runners[kind] = InProcessRunner(plots)
            elif kind == "subprocess":
                self._runners[kind] = SubprocessRunner(plots)
            elif kind == "sandbox":
                from ghostline.physics.sandbox import FitSandbox

                sandbox = FitSandbox(plots_dir=plots).start()
                self._sandbox = sandbox
                self._runners[kind] = sandbox
            else:
                raise ValueError(f"unknown fit runner {kind!r}")
        return self._runners[kind]

    def run(self, job: FitJob) -> FitResult:
        kind = self._choose(job)
        try:
            return self._runner(kind).run(job)
        except Exception:
            custom = any(spec.code is not None for spec in job.models)
            if kind != "sandbox" or custom or _factory is not None:
                raise
            log.warning("the fit sandbox failed; running library models in-process", exc_info=True)
            if self._sandbox is not None:
                with contextlib.suppress(Exception):
                    self._sandbox.close()
                self._sandbox = None
            self._runners.pop("sandbox", None)
            return self._runner("inprocess").run(job)
