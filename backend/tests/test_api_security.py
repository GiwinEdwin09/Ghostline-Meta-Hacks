"""Part 28 checks that need no network: the built-JavaScript scanner, and the limits on what a
stranger can upload or read."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_built_secrets as scanner  # noqa: E402


def bundle(tmp_path, text: bytes, name="page.js") -> Path:
    folder = tmp_path / ".next" / "static"
    folder.mkdir(parents=True)
    (folder / name).write_bytes(text)
    return tmp_path / ".next"


def test_a_clean_bundle_passes(tmp_path):
    root = bundle(tmp_path, b"const base = process.env.NEXT_PUBLIC_API_BASE; fetch(base)")
    assert scanner.scan([root], environ={"APP_PASSCODE": "hunter2hunter2"}) == []


def test_the_value_of_a_secret_is_caught_without_being_printed(tmp_path):
    root = bundle(tmp_path, b'headers: {"X-Ghostline-Key": "hunter2hunter2"}')
    found = scanner.scan([root], environ={"APP_PASSCODE": "hunter2hunter2"})
    assert found == [f"the value of APP_PASSCODE is in {root / 'static' / 'page.js'}"]
    assert all("hunter2" not in line for line in found)


@pytest.mark.parametrize(
    "text",
    [
        b"const k = process.env.MODEL_API_KEY",
        b'"wk-abcdef123456.ws-abcdef123456"',
        b'"postgresql://postgres:s3cretpass@db.example.com:6543/postgres"',
        b'Authorization: "Bearer abcdefghijklmnopqrstuvwxyz0123"',
    ],
)
def test_names_and_key_shaped_strings_are_caught(tmp_path, text):
    assert scanner.scan([bundle(tmp_path, text)], environ={})


def test_the_scanner_says_when_there_is_no_build(tmp_path, capsys):
    assert scanner.main([str(tmp_path / "missing")]) == 2
    assert "npm run build" in capsys.readouterr().out
