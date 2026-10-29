"""The Modal Sandbox that runs the fitting harness (build plan Part 11).

One sandbox per analysis job, reused for every fit in the job, with no network. The image holds
the scientific stack and a copy of `ghostline/physics`; PyBullet for the export replay check
(Part 19) is left out for now.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import modal

from .harness.schema import FitJob, FitResult

PHYSICS_DIR = Path(__file__).resolve().parent
HARNESS_ROOT = "/harness"
DEFAULT_APP = "ghostline-fit-dev"

# Pinned to the versions in uv.lock so the Sandbox and the tests fit the same way.
fit_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.4",
        "scipy==1.18.1",
        "pandas==3.1.0",
        "matplotlib==3.11.2",
        "pydantic==2.14.1",
        "sympy==1.14.0",
    )
    .env(
        {
            "PYTHONPATH": HARNESS_ROOT,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        }
    )
    .add_local_file(
        PHYSICS_DIR.parent / "__init__.py", f"{HARNESS_ROOT}/ghostline/__init__.py", copy=True
    )
    .add_local_dir(
        PHYSICS_DIR,
        f"{HARNESS_ROOT}/ghostline/physics",
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)


class HarnessError(RuntimeError):
    """The harness process failed inside the Sandbox."""


class FitSandbox:
    """A reusable no-network Sandbox. Use as a context manager; `run` takes a FitJob."""

    def __init__(
        self,
        *,
        environment: str | None = None,
        app_name: str = DEFAULT_APP,
        plots_dir: Path | None = None,
        cpu: int = 2,
        memory: int = 2048,
        timeout: int = 600,
        idle_timeout: int = 120,
    ):
        self.environment = environment or os.environ.get("MODAL_ENVIRONMENT", "dev")
        self.app_name = app_name
        self.cpu = cpu
        self.memory = memory
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.plots_dir = plots_dir
        self._sandbox: modal.Sandbox | None = None

    def start(self) -> FitSandbox:
        app = modal.App.lookup(
            self.app_name, create_if_missing=True, environment_name=self.environment
        )
        self._sandbox = modal.Sandbox.create(
            app=app,
            image=fit_image,
            block_network=True,
            timeout=self.timeout,
            idle_timeout=self.idle_timeout,
            cpu=self.cpu,
            memory=self.memory,
        )
        return self

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.terminate()
            self._sandbox = None

    def __enter__(self) -> FitSandbox:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def _fetch_plots(self, work: str, result: FitResult) -> None:
        """Copy every plot in one tar file (one round trip instead of one per PNG)."""
        sandbox = self._sandbox
        assert sandbox is not None and self.plots_dir is not None
        proc = sandbox.exec("tar", "cf", f"{work}/plots.tar", "-C", f"{work}/plots", ".")
        proc.wait()
        target = self.plots_dir / result.job_id
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "plots.tar"
        sandbox.filesystem.copy_to_local(f"{work}/plots.tar", str(archive))
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        for model in result.models:
            for ref in model.plots:
                ref.path = str(target / model.id / ref.path)

    def exec_python(self, code: str, timeout: int = 60) -> tuple[int, str, str]:
        """Run a snippet in the Sandbox. Used for health and isolation checks."""
        assert self._sandbox is not None, "start the sandbox first"
        proc = self._sandbox.exec("python", "-c", code, timeout=timeout)
        proc.wait()
        return proc.returncode, proc.stdout.read(), proc.stderr.read()

    def run(self, job: FitJob, exec_timeout: int = 180) -> FitResult:
        sandbox = self._sandbox
        assert sandbox is not None, "start the sandbox first"
        job = job.model_copy(update={"workers": max(job.workers, self.cpu)})
        work = f"/work/{job.job_id}"
        sandbox.filesystem.make_directory(work)
        sandbox.filesystem.write_text(job.model_dump_json(), f"{work}/job.json")
        proc = sandbox.exec(
            "python", "-m", "ghostline.physics.harness.run_fit", f"{work}/job.json",
            timeout=exec_timeout,
        )  # fmt: skip
        proc.wait()
        if proc.returncode != 0:
            raise HarnessError(f"exit {proc.returncode}: {proc.stderr.read()[-2000:]}")
        try:
            text = sandbox.filesystem.read_text(f"{work}/result.json")
        except modal.exception.SandboxFilesystemNotFoundError as exc:
            out, err = proc.stdout.read(), proc.stderr.read()
            raise HarnessError(
                f"no result.json; stdout {out[-500:]!r} stderr {err[-1500:]!r}"
            ) from exc
        result = FitResult.model_validate_json(text)
        if self.plots_dir is not None and any(m.plots for m in result.models):
            self._fetch_plots(work, result)
        return result
