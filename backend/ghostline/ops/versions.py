"""Which prompts produced a result (build plan Parts 26 and 29).

The prompts in backend/ghostline/prompts/*.md steer every model call, so a benchmark or eval
number only means something next to the version of the prompts that produced it. The version is a
short hash over the files' names and contents; change one word of one prompt and it changes.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class PromptVersion:
    version: str  # 12 hex characters, or "none" when there are no prompt files
    files: dict[str, str] = field(default_factory=dict)  # file name -> first 12 hex of its sha256

    def __str__(self) -> str:
        return self.version


def _digest(data: bytes) -> str:
    # Line endings must not change the version between a Windows checkout and the Modal image.
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def prompt_version(prompts_dir: Path | str | None = None) -> PromptVersion:
    folder = Path(prompts_dir) if prompts_dir else PROMPTS_DIR
    files = {p.name: _digest(p.read_bytes()) for p in sorted(folder.glob("*.md"))}
    if not files:
        return PromptVersion("none")
    overall = hashlib.sha256()
    for name, digest in files.items():
        overall.update(f"{name}\0{digest}\n".encode())
    return PromptVersion(overall.hexdigest()[:12], {n: d[:12] for n, d in files.items()})
