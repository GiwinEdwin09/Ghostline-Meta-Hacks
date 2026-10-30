"""Part 28's check: no secret appears in the built JavaScript.

    cd web && npm run build && cd .. && uv run python scripts/check_built_secrets.py

Scans web/.next (or the directories given) for the values of the back end's secrets when they are
set in the environment, for the names of the server-only variables, and for strings shaped like a
key (a Modal proxy token, a database URL with a password, a bearer token). Prints what it found
without printing any value, and exits 1 if there is anything.
"""

import os
import re
import sys
from pathlib import Path

SECRET_VARS = ("MODEL_API_KEY", "REVIEWER_API_KEY", "APP_PASSCODE", "DATABASE_URL")
# Names that only the server reads: seeing one in a browser bundle means it was imported there.
SERVER_ONLY_NAMES = SECRET_VARS + ("REVIEWER_BASE_URL",)
SHAPES = {
    "a Modal proxy token": re.compile(rb"\bwk-[A-Za-z0-9]{6,}\.ws-[A-Za-z0-9]{6,}"),
    "a database URL with a password": re.compile(rb"postgres(?:ql)?://[^\s:@/\"']+:[^\s@/\"']+@"),
    "a bearer token": re.compile(rb"Bearer\s+[A-Za-z0-9._-]{20,}"),
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".ico", ".woff", ".woff2", ".mp4", ".webp", ".svg"}


def scan(roots: list[Path], environ=os.environ) -> list[str]:
    secrets = {
        name: environ[name].encode() for name in SECRET_VARS if len(environ.get(name, "")) >= 6
    }
    findings: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in SKIP_SUFFIXES:
                continue
            data = path.read_bytes()
            for name, value in secrets.items():
                if value in data:
                    findings.append(f"the value of {name} is in {path}")
            for name in SERVER_ONLY_NAMES:
                if name.encode() in data:
                    findings.append(f"the server-only name {name} is in {path}")
            for what, pattern in SHAPES.items():
                if pattern.search(data):
                    findings.append(f"something shaped like {what} is in {path}")
    return findings


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path("web/.next")]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        print(f"nothing to scan: {', '.join(map(str, missing))} (run `npm run build` first)")
        return 2
    findings = scan(roots)
    for line in findings:
        print(line)
    files = sum(1 for r in roots for p in r.rglob("*") if p.is_file())
    print(f"scanned {files} files: {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
