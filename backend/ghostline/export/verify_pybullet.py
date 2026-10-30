"""The PyBullet replay check (build plan Part 19).

The check runs the exported PyBullet script, the same text a user downloads, inside a no-network
Modal Sandbox whose image has PyBullet. The script rebuilds the experiment from the measured
geometry (a box on a tilted board, a sphere drop, a falling sphere, a pendulum on a revolute
joint), applies the exported parameters and steps at 240 Hz. This module then compares what it
reports with the measured 95% intervals and returns the `replay_check` dict.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ghostline.jobs.errors import StageError, TransientStageError

from .expectations import PYBULLET_VERSION, TIMESTEP_HZ, Check, expected_checks
from .measured import Measurement
from .snippet import render_pybullet

RESULT_MARKER = "GHOSTLINE_RESULT "
SCRIPT_NAME = "ghostline_export.py"
DEFAULT_APP = "ghostline-fit-dev"
_RUNNER = (
    "import json, sys; sys.path.insert(0, '/work'); import ghostline_export as m; "
    f"print('{RESULT_MARKER}' + json.dumps(m.simulate()))"
)


def parse_result(stdout: str) -> dict[str, Any]:
    """The `simulate()` dict from a run's output. PyBullet prints its own banner and warnings
    first, some without a trailing newline, so look for the marker anywhere."""
    at = stdout.rfind(RESULT_MARKER)
    if at < 0:
        raise StageError(f"The PyBullet script printed no result: {stdout[-400:]!r}")
    return json.loads(stdout[at + len(RESULT_MARKER) :].splitlines()[0])


def run_locally(source: str) -> dict[str, Any]:
    """Run the script in this process. For development machines that have PyBullet."""
    namespace: dict[str, Any] = {"__name__": "ghostline_export"}
    exec(compile(source, SCRIPT_NAME, "exec"), namespace)  # noqa: S102 - our own generated text
    return namespace["simulate"]()


def run_in_sandbox(
    source: str, *, environment: str | None = None, app_name: str = DEFAULT_APP, timeout: int = 240
) -> dict[str, Any]:
    """Run the script in a Modal Sandbox with PyBullet and no network."""
    import modal

    from .image import replay_image

    environment = environment or os.environ.get("MODAL_ENVIRONMENT", "dev")
    try:
        app = modal.App.lookup(app_name, create_if_missing=True, environment_name=environment)
        sandbox = modal.Sandbox.create(
            app=app,
            image=replay_image,
            block_network=True,
            timeout=timeout,
            idle_timeout=timeout,
            cpu=1,
            memory=1024,
        )
    except Exception as exc:  # the cloud, not the script
        raise TransientStageError(f"Couldn't start the PyBullet sandbox: {exc}") from exc
    try:
        sandbox.filesystem.make_directory("/work")
        sandbox.filesystem.write_text(source, f"/work/{SCRIPT_NAME}")
        proc = sandbox.exec("python", "-c", _RUNNER, timeout=timeout)
        proc.wait()
        out, err = proc.stdout.read(), proc.stderr.read()
        if proc.returncode != 0:
            raise StageError(f"The PyBullet script failed: {err[-800:]}", code="pybullet_failed")
        return parse_result(out)
    finally:
        sandbox.terminate()


def compare(checks: list[Check], simulated: dict[str, Any]) -> dict[str, Any]:
    """Build the `replay_check` dict. It passes when every simulated value is inside its
    measured 95% interval. The headline fields describe the first failing check, else the first."""
    rows = []
    for c in checks:
        value = simulated.get(c.key)
        ok = value is not None and c.low <= value <= c.high
        rows.append(
            {
                "quantity": c.quantity,
                "simulated": None if value is None else float(value),
                "measured": c.measured,
                "ci95": [c.low, c.high],
                "unit": c.unit,
                "passed": bool(ok),
            }
        )
    if not rows:
        raise StageError("There is nothing to compare the replay with.", code="missing_quantity")
    head = next((r for r in rows if not r["passed"]), rows[0])
    return {
        "simulator": "pybullet",
        **head,
        "passed": all(r["passed"] for r in rows),
        "checks": rows,
        "timestep_hz": TIMESTEP_HZ,
        "pybullet_version": PYBULLET_VERSION,
    }


def replay(m: Measurement, *, runner=run_in_sandbox) -> dict[str, Any]:
    """Run the export for this measurement through `runner` and compare it with the measurement."""
    checks = expected_checks(m)  # raises if this law has no replay
    source = render_pybullet(m, replay=True)
    return compare(checks, runner(source))


def main(argv: list[str]) -> int:
    """`python -m ghostline.export.verify_pybullet SCRIPT.py` runs a generated script locally."""
    if len(argv) != 2:
        print("usage: verify_pybullet.py SCRIPT.py", flush=True)
        return 2
    print(run_locally(Path(argv[1]).read_text()))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
