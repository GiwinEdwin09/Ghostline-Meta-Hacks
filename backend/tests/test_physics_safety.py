import numpy as np
import pytest

from ghostline.physics.harness.custom import CustomModel, dry_run
from ghostline.physics.harness.safety import check_code
from ghostline.physics.harness.schema import FitJob, ModelSpec, ParamSpec, SeriesData
from ghostline.physics.runner import InProcessRunner

GOOD = """
import math
import numpy as np

def initial_state(p):
    return [p['y0'], p['vy0']]

def rhs(t, s, p):
    return [s[1], -p['g'] - p['b'] * s[1]]
"""
CLOSED = """
import numpy as np

def closed_form(t, p):
    t = np.asarray(t)
    return p['y0'] + p['vy0'] * t - 0.5 * p['g'] * t**2
"""
SPECS = {
    "y0": ParamSpec(init=1.0, lower=-5, upper=5, unit="m", kind="initial_condition"),
    "vy0": ParamSpec(init=0.0, lower=-10, upper=10, unit="m/s", kind="initial_condition"),
    "g": ParamSpec(init=9.8, lower=1, upper=30, unit="m/s^2"),
    "b": ParamSpec(init=0.1, lower=0, upper=10, unit="1/s"),
}


def make(code, params=None, **kwargs):
    return ModelSpec(
        id="C1", code=code, params=params or SPECS, observables=["y"], state=["y", "vy"], **kwargs
    )


def test_valid_code_passes():
    assert check_code(GOOD) == []
    assert check_code(CLOSED) == []
    assert check_code("from scipy.special import erf\nimport math\n" + GOOD) == []


@pytest.mark.parametrize(
    "snippet,fragment",
    [
        ("import os", "not allowed"),
        ("import sys", "not allowed"),
        ("import subprocess", "not allowed"),
        ("from os import path", "not allowed"),
        ("import numpy.linalg as la", "not allowed"),
        ("from scipy.optimize import minimize", "not allowed"),
        ("import socket", "not allowed"),
        ("data = open('/etc/passwd').read()", "'open'"),
        ("eval('1+1')", "'eval'"),
        ("exec('x = 1')", "'exec'"),
        ("__import__('os')", "'__import__'"),
        ("x = ().__class__.__bases__", "__class__"),
        ("x = np.load('a.npy')", "'load'"),
        ("x = np.fromfile('a')", "'fromfile'"),
        ("x = np.ctypeslib", "'ctypeslib'"),
        ("x = getattr(np, 'load')", "'getattr'"),
        ("x = os.environ", "'os'"),
        ("x = sys.modules", "'sys'"),
        ("class A:\n    pass", "ClassDef"),
        ("global z", "Global"),
        ("with np.errstate():\n    pass", "With"),
    ],
)
def test_rejects_dangerous_code(snippet, fragment):
    errors = check_code(GOOD + "\n" + snippet)

    assert errors, snippet
    assert any(fragment in e for e in errors), errors


def test_checks_function_signatures_and_required_functions():
    assert any(
        "rhs must take exactly 3" in e
        for e in check_code(GOOD.replace("rhs(t, s, p)", "rhs(t, s)"))
    )
    assert any("define rhs" in e for e in check_code("def helper(x):\n    return x"))
    assert any("needs initial_state" in e for e in check_code("def rhs(t, s, p):\n    return s"))
    assert any("syntax error" in e for e in check_code("def rhs(:"))


def test_oversized_code_is_rejected():
    assert check_code("x = 1\n" * 10_000)


def test_custom_model_matches_the_library_model():
    from ghostline.physics.library import get_model

    custom = CustomModel(make(GOOD), "free_fall", ("y",))
    library = get_model("FF1")
    p = dict(y0=1.5, vy0=-1.0, g=9.81, b=0.8)
    tau = np.linspace(0, 0.5, 50)

    assert np.allclose(custom.predict(tau, p), library.predict(tau, p), atol=1e-8)
    assert dry_run(custom, {}) == []


def test_dry_run_reports_shape_nan_and_runtime_errors():
    bad_shape = GOOD.replace("return [s[1], -p['g'] - p['b'] * s[1]]", "return [s[1]]")
    nan = GOOD.replace("-p['g'] - p['b'] * s[1]", "math.log(-1.0) if False else float('nan')")
    boom = GOOD.replace("-p['g'] - p['b'] * s[1]", "p['missing']")
    wrong_closed = CLOSED.replace("return p['y0']", "return np.zeros((3, len(t))) + p['y0']")

    assert any("shape" in e for e in dry_run(CustomModel(make(bad_shape), "free_fall", ("y",)), {}))
    assert any("NaN" in e for e in dry_run(CustomModel(make(nan), "free_fall", ("y",)), {}))
    assert any("KeyError" in e for e in dry_run(CustomModel(make(boom), "free_fall", ("y",)), {}))
    assert any(
        "shape" in e for e in dry_run(CustomModel(make(wrong_closed), "free_fall", ("y",)), {})
    )
    needs_const = make(GOOD, constants=["L"])
    assert any("constants" in e for e in dry_run(CustomModel(needs_const, "free_fall", ("y",)), {}))


def test_runtime_import_guard_blocks_modules_the_ast_check_would_miss():
    # Even if the static check were bypassed, the restricted __import__ refuses other modules.
    from ghostline.physics.harness.custom import _namespace

    ns = _namespace()
    with pytest.raises(ImportError):
        exec("import os", ns)  # noqa: S102
    with pytest.raises(ImportError):
        exec("from scipy import optimize", ns)  # noqa: S102
    with pytest.raises(NameError):
        exec("open('x')", ns)  # noqa: S102
    exec("from scipy import special\nimport math", ns)  # noqa: S102


def test_harness_validate_job_rejects_malicious_models_without_running_them(tmp_path):
    marker = tmp_path / "pwned.txt"
    attack = f"""
import numpy as np
open({str(marker)!r}, 'w').write('x')
def closed_form(t, p):
    return np.zeros((1, len(t)))
"""
    read_file = GOOD + "\nsecret = open('/etc/hosts').read()\n"
    net = "import socket\n" + GOOD
    job = FitJob(
        mode="validate",
        system="free_fall",
        models=[
            make(attack).model_copy(update={"id": "attack"}),
            make(read_file).model_copy(update={"id": "read"}),
            make(net).model_copy(update={"id": "net"}),
            make(GOOD).model_copy(update={"id": "ok"}),
        ],
    )

    result = InProcessRunner().run(job)

    reports = {r.model_id: r for r in result.validation}
    assert not reports["attack"].ok and not reports["read"].ok and not reports["net"].ok
    assert reports["ok"].ok
    assert not marker.exists()


def test_fit_job_with_invalid_custom_code_marks_the_model_invalid():
    t = np.arange(0, 0.5, 1 / 120)
    data = SeriesData(t=t.tolist(), series={"y": (1.2 - 4.9 * t**2).tolist()}, sigma={"y": 1e-3})
    job = FitJob(
        mode="select",
        system="free_fall",
        data=data,
        models=[
            make("import os\n" + GOOD).model_copy(update={"id": "evil"}),
            ModelSpec(id="FF0", library_ref="FF0"),
        ],
        plots=False,
    )

    result = InProcessRunner().run(job)

    assert result.model("evil").status == "invalid"
    assert "not allowed" in result.model("evil").error
    assert result.model("FF0").status == "ok"


def test_infinite_loop_in_model_code_is_stopped():
    import os

    if not hasattr(__import__("signal"), "setitimer") or os.name == "nt":
        pytest.skip("needs SIGALRM")
    loop = "def closed_form(t, p):\n    while True:\n        pass\n"
    spec = make(loop)
    model = CustomModel(spec, "free_fall", ("y",))

    errors = dry_run(model, {}, limit_s=0.3)

    assert any("longer than" in e for e in errors)
