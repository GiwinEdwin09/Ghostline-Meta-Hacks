"""Ways to run a fit job: in this process, or in a subprocess that follows the Sandbox contract.

Every runner takes a `FitJob` and returns a `FitResult` whose plot paths are local files. The
Modal runner in `sandbox.py` has the same `run` method.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol

from .harness import run_fit
from .harness.schema import FitJob, FitResult


class Runner(Protocol):
    def run(self, job: FitJob) -> FitResult: ...


def _collect_plots(result: FitResult, workdir: Path, plots_dir: Path) -> None:
    """Copy plot files out of the job's work directory and point each PlotRef at the copy."""
    target = plots_dir / result.job_id
    for model in result.models:
        for ref in model.plots:
            source = workdir / "plots" / model.id / ref.path
            if source.exists():
                target.mkdir(parents=True, exist_ok=True)
                dest = target / f"{ref.id}.png"
                shutil.copyfile(source, dest)
                ref.path = str(dest)


class _PlotsDir:
    def __init__(self, plots_dir: Path | None = None):
        self._tmp = None if plots_dir else tempfile.TemporaryDirectory(prefix="ghostline-plots-")
        self.path = plots_dir or Path(self._tmp.name)  # type: ignore[union-attr]


class InProcessRunner(_PlotsDir):
    """Run the harness in this process. Fine for library models; custom code is only AST-checked."""

    def run(self, job: FitJob) -> FitResult:
        with tempfile.TemporaryDirectory(prefix="ghostline-job-") as tmp:
            result = run_fit.run_job(job, Path(tmp))
            _collect_plots(result, Path(tmp), self.path)
        return result


class SubprocessRunner(_PlotsDir):
    """Run `run_fit.py` as a script in a child process, the way the Sandbox does."""

    def __init__(self, plots_dir: Path | None = None, timeout: float = 180.0):
        super().__init__(plots_dir)
        self.timeout = timeout

    def run(self, job: FitJob) -> FitResult:
        script = Path(run_fit.__file__).resolve()
        with tempfile.TemporaryDirectory(prefix="ghostline-job-") as tmp:
            work = Path(tmp)
            (work / "job.json").write_text(job.model_dump_json())
            proc = subprocess.run(
                [sys.executable, str(script), str(work / "job.json")],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tmp,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"harness exited with {proc.returncode}: {proc.stderr[-2000:]}")
            result = FitResult.model_validate_json((work / "result.json").read_text())
            _collect_plots(result, work, self.path)
        return result
