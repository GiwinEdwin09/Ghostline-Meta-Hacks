"""The recorded Wonder diagnosis (see record.py): its inputs, and a client that replays it."""

from __future__ import annotations

from pathlib import Path

from ghostline.config import get_settings
from ghostline.ghost.diagnose import DiagnosisInputs, DiagnosisReply, ask_wonder
from ghostline.llm import InMemorySpendStore, MetaClient, Runtime

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "diag_inputs"
CACHE = HERE / "cache"


def recorded_inputs() -> DiagnosisInputs:
    """The brief and images Wonder was shown when the call was recorded, byte for byte."""
    import json

    images = sorted(INPUTS.glob("image_*"), key=lambda p: int(p.stem.split("_")[1]))
    return DiagnosisInputs(
        (INPUTS / "brief.md").read_text(),
        [p.read_bytes() for p in images],
        json.loads((INPUTS / "labels.json").read_text()),
    )


def replay_client() -> MetaClient:
    runtime = Runtime.create(
        get_settings(), mode="replay", cache_dir=CACHE, store=InMemorySpendStore()
    )
    return MetaClient(runtime)


def recorded_reply() -> DiagnosisReply:
    return ask_wonder(replay_client(), recorded_inputs())
