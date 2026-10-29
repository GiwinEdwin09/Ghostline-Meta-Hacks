import os
import subprocess
import sys
import time

import numpy as np
import pytest

from ghostline.physics import sandbox
from ghostline.physics.harness.schema import FitJob, ModelSpec, ParamSpec
from ghostline.physics.synthetic import make_case

live = pytest.mark.skipif(os.environ.get("GHOSTLINE_LIVE") != "1", reason="needs GHOSTLINE_LIVE=1")

ATTACKS = {
    "read_file": "def closed_form(t, p):\n    open('/etc/passwd').read()\n    return t\n",
    "network": "import urllib.request\ndef closed_form(t, p):\n    return t\n",
    "dunder": "def closed_form(t, p):\n    return ().__class__.__bases__\n",
}
SPECS = {"y0": ParamSpec(init=1.0, lower=-5, upper=5, unit="m", kind="initial_condition")}


def test_harness_imports_only_what_the_slim_fit_image_has():
    code = (
        "import sys, ghostline.physics.harness.run_fit\n"
        "stdlib = set(sys.stdlib_module_names)\n"
        "top = {m.split('.')[0] for m in sys.modules if not m.startswith('_')}\n"
        "print(sorted(top - stdlib))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    loaded = set(eval(out.stdout.strip()))  # noqa: S307 - our own subprocess output
    for forbidden in ("modal", "fastapi", "openai", "httpx", "cv2", "av", "psycopg", "pyarrow"):
        assert forbidden not in loaded
    assert {"numpy", "scipy", "pydantic", "ghostline"} <= loaded


def test_fit_image_is_built_without_contacting_modal():
    import modal

    assert isinstance(sandbox.fit_image, modal.Image)
    assert sandbox.HARNESS_ROOT == "/harness"
    for name in ("run_fit.py", "fitting.py", "schema.py", "safety.py", "custom.py"):
        assert (sandbox.PHYSICS_DIR / "harness" / name).exists()
    assert (sandbox.PHYSICS_DIR.parent / "__init__.py").exists()


@live
def test_five_model_comparison_runs_inside_the_sandbox_in_under_20_seconds(tmp_path):
    rng = np.random.default_rng(1)
    case = make_case(
        "pendulum", "P2", dict(theta0=0.7, omega0=0.0, g=9.81, gamma=0.08), {"L": 0.6},
        np.arange(600) / 60.0, 0.002, rng,
    )  # fmt: skip
    job = case.job(["P0", "P1", "P2", "P3", "P4"], job_id="live-pendulum")

    with sandbox.FitSandbox(plots_dir=tmp_path) as sb:
        started = time.monotonic()
        result = sb.run(job)
        elapsed = time.monotonic() - started
        print(
            f"five-model comparison: {elapsed:.1f} s wall, {result.elapsed_s:.1f} s in the harness"
        )

        assert result.elapsed_s < 20  # the comparison itself, inside the sandbox
        assert elapsed < 30  # plus upload, plots and download
        assert all(m.status == "ok" for m in result.models)
        g = next(p for p in result.model("P2").full.params if p.name == "g")
        assert abs(g.value - 9.81) < 3 * g.se
        assert all(os.path.exists(ref.path) for m in result.models for ref in m.plots)

        evil = FitJob(
            mode="validate",
            system="free_fall",
            models=[
                ModelSpec(id=name, code=code, params=SPECS, observables=["y"])
                for name, code in ATTACKS.items()
            ],
        )
        reports = {r.model_id: r for r in sb.run(evil).validation}
        assert not any(r.ok for r in reports.values())

        # Even code that skips the checks cannot reach the network from inside the sandbox.
        code, _, _ = sb.exec_python(
            "import urllib.request\nurllib.request.urlopen('https://example.com', timeout=5)"
        )
        assert code != 0
