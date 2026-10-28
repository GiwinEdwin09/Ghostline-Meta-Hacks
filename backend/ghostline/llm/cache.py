"""Record/replay cache: one file per call, keyed by a hash of the normalized request.

Layout (mirrors the Volume in build plan Part 22):

    <root>/llm/<hash>.json            Responses and Chat Completions calls, SAM image calls
    <root>/sam/<hash>.json            SAM video call metadata
    <root>/sam/<hash>.events.jsonl    SAM video raw event stream, one event per line

The key covers the model, input, tools and parameters. Secrets are removed, and media is
replaced by content hashes, so a key doesn't depend on a file ID or an auth header.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SECRET_KEYS = {"authorization", "api_key", "api-key", "x-api-key", "token", "secret", "password"}
_DATA_URL = re.compile(r"^data:([^;,]+)(?:;[^,]*)?;base64,(.*)$", re.DOTALL)
CACHE_VERSION = 1


def normalize_request(value: Any, file_hashes: Mapping[str, str] | None = None) -> Any:
    """Return a JSON-able copy of a request with secrets removed and media replaced by hashes.

    `file_hashes` maps an uploaded file's ID to the sha256 of its bytes, so that re-uploading
    the same clip doesn't change the key.
    """
    file_hashes = file_hashes or {}
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if str(key).lower() in _SECRET_KEYS:
                continue
            if key == "file_id" and isinstance(item, str):
                out[key] = f"sha256:{file_hashes[item]}" if item in file_hashes else item
            else:
                out[key] = normalize_request(item, file_hashes)
        return out
    if isinstance(value, list | tuple):
        return [normalize_request(item, file_hashes) for item in value]
    if isinstance(value, str):
        match = _DATA_URL.match(value)
        if match:
            digest = hashlib.sha256(match.group(2).encode()).hexdigest()
            return f"data:{match.group(1)};sha256={digest}"
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def request_hash(kind: str, normalized: Any) -> str:
    text = json.dumps({"kind": kind, "request": normalized}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayCache:
    """A directory of recorded calls."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _folder(self, kind: str) -> Path:
        return self.root / ("sam" if kind.startswith("sam") else "llm")

    def path(self, kind: str, key: str) -> Path:
        return self._folder(kind) / f"{key}.json"

    def events_path(self, kind: str, key: str) -> Path:
        return self._folder(kind) / f"{key}.events.jsonl"

    def has(self, kind: str, key: str) -> bool:
        return self.path(kind, key).exists()

    def get(self, kind: str, key: str) -> dict | None:
        path = self.path(kind, key)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        events_path = self.events_path(kind, key)
        if events_path.exists():
            with events_path.open(encoding="utf-8") as handle:
                entry["events"] = [json.loads(line) for line in handle if line.strip()]
        return entry

    def put(self, kind: str, key: str, entry: dict) -> None:
        """Write an entry. A SAM entry's `events` go to their own JSONL file."""
        entry = {"version": CACHE_VERSION, "kind": kind, "hash": key, **entry}
        events = entry.pop("events", None)
        path = self.path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if events is not None:
            with self.events_path(kind, key).open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        path.write_text(json.dumps(entry, indent=1), encoding="utf-8")  # key order is kept
