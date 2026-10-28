"""Static checks for Wonder-written model code (build plan Part 11).

The AST check is a first filter that gives Wonder clear errors. The security boundary is the
Modal Sandbox: no network, no secrets, CPU, memory and time limits.
"""

from __future__ import annotations

import ast

ALLOWED_MODULES = {"math", "numpy", "scipy.special"}
MAX_CODE_BYTES = 20_000

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit", "memoryview",
    "type", "object", "super", "classmethod", "staticmethod", "property",
    "os", "sys", "subprocess", "builtins", "importlib", "socket", "ctypes", "shutil", "pathlib",
    "io", "pickle", "marshal", "inspect", "gc", "threading", "multiprocessing", "signal",
    "requests", "urllib", "http",
}  # fmt: skip
FORBIDDEN_ATTRS = {
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "ctypeslib", "fromregex", "DataSource", "lib", "testing", "f2py",
    "distutils", "core", "os", "sys", "system", "popen", "environ", "modules", "builtins",
}  # fmt: skip
FORBIDDEN_NODES = (
    ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield,
    ast.YieldFrom, ast.AsyncFor, ast.AsyncWith, ast.With, ast.Delete,
)  # fmt: skip

# (name, positional arguments, required)
REQUIRED_SIGNATURES = {
    "initial_state": 1,
    "rhs": 3,
    "closed_form": 2,
    "observe": 1,
}


def _import_errors(node: ast.AST) -> list[str]:
    errors = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in ALLOWED_MODULES:
                errors.append(f"line {node.lineno}: import of {alias.name!r} is not allowed")
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = {alias.name for alias in node.names}
        scipy_special = module == "scipy" and names <= {"special"}
        if node.level or (module not in ALLOWED_MODULES and not scipy_special):
            errors.append(f"line {node.lineno}: import from {module!r} is not allowed")
        if "*" in names:
            errors.append(f"line {node.lineno}: star imports are not allowed")
        for name in sorted(names & FORBIDDEN_ATTRS):
            errors.append(f"line {node.lineno}: importing {name!r} is not allowed")
    return errors


def check_code(code: str) -> list[str]:
    """Return a list of problems with the code; empty means it passes the static checks."""
    if len(code.encode()) > MAX_CODE_BYTES:
        return [f"code is larger than {MAX_CODE_BYTES} bytes"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error on line {exc.lineno}: {exc.msg}"]

    errors: list[str] = []
    for node in ast.walk(tree):
        errors.extend(_import_errors(node))
        if isinstance(node, FORBIDDEN_NODES):
            errors.append(f"line {node.lineno}: {type(node).__name__} statements are not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                errors.append(f"line {node.lineno}: the name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__"):
                errors.append(f"line {node.lineno}: the attribute {node.attr!r} is not allowed")

    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, count in REQUIRED_SIGNATURES.items():
        fn = functions.get(name)
        if fn is None:
            continue
        args = fn.args
        if args.vararg or args.kwarg or args.kwonlyargs or len(args.args) != count:
            errors.append(
                f"line {fn.lineno}: {name} must take exactly {count} positional argument(s)"
            )
    has_ode = "rhs" in functions and "initial_state" in functions
    if "closed_form" not in functions and not has_ode:
        errors.append("define rhs(t, s, p) and initial_state(p), or closed_form(t, p)")
    if "rhs" in functions and "initial_state" not in functions and "closed_form" not in functions:
        errors.append("rhs needs initial_state(p)")
    return sorted(set(errors), key=errors.index)
