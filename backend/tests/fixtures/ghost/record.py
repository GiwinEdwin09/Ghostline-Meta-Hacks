"""Record Wonder's diagnosis of an undamped ghost on a damped pendulum (about $0.1-0.3).

    uv run --env-file .env python backend/tests/fixtures/ghost/record.py

It builds the damped experiment of test_ghost_revision.py, runs the ghost stage, then runs the
revision stage with the Wonder call in `record` mode. The brief and the images Wonder was shown
are saved in `diag_inputs/`, so the replay test sends byte-identical requests whatever machine
renders the frames. Keys are read from the environment and never written down.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[3])]

from fixtures.ghost.rig import pendulum_rig  # noqa: E402

from ghostline.config import get_settings  # noqa: E402
from ghostline.ghost.diagnose import DiagnosisInputs, ask_wonder  # noqa: E402
from ghostline.ghost.stage import diagnose_revision, predict_ghost  # noqa: E402
from ghostline.llm import InMemorySpendStore, MetaClient, Runtime  # noqa: E402
from ghostline.store.records import GhostRender  # noqa: E402
from ghostline.store.volume import VolumePaths  # noqa: E402

INPUTS = HERE / "diag_inputs"
CACHE = HERE / "cache"
GAMMA, D1, D2 = 0.15, 5.0, 10.0


def save(inputs: DiagnosisInputs) -> None:
    INPUTS.mkdir(exist_ok=True)
    (INPUTS / "brief.md").write_text(inputs.text)
    (INPUTS / "labels.json").write_text(json.dumps(inputs.labels, indent=1))
    for k, image in enumerate(inputs.images):
        (INPUTS / f"image_{k}{'.png' if image[:4] == b'\x89PNG' else '.jpg'}").write_bytes(image)


def main() -> None:
    settings = get_settings()
    store = InMemorySpendStore()
    runtime = Runtime.create(settings, mode="record", cache_dir=CACHE, store=store)
    meta = MetaClient(runtime)
    root = Path(tempfile.mkdtemp(prefix="ghostline-ghost-record-"))
    paths = VolumePaths(root / "data")
    rig_settings = dataclasses.replace(settings, volume_root=str(root / "data"))
    rig = pendulum_rig(paths, rig_settings, gamma=GAMMA, theta1=0.5, theta2=0.4, d1=D1, d2=D2)
    v1 = predict_ghost(
        rig.ctx, rig.experiment, rig.fit_trial, rig.prove_trial, rig.scene, rig.trajectory
    )
    print("v1 onset", v1.divergence_onset_s, "tau", v1.threshold_mm, "max", v1.max_error_mm)
    rig.repo.ghost_count = 1
    rig.experiment = rig.experiment.model_copy(update={"scene": rig.scene})
    rig.repo.latest_trial = lambda *a, **k: rig.fit_trial
    ghost = GhostRender(
        id=uuid4(), experiment_id=rig.experiment.id, trial_id=rig.prove_trial.id,
        model_id="P1", version=1, video_path=v1.video_path, track_path=v1.track_path,
    )  # fmt: skip

    def ask(inputs: DiagnosisInputs):
        save(inputs)  # then send exactly what was saved, which is what the replay test sends
        images = sorted(INPUTS.glob("image_*"), key=lambda p: int(p.stem.split("_")[1]))
        saved = DiagnosisInputs(
            (INPUTS / "brief.md").read_text(), [p.read_bytes() for p in images], inputs.labels
        )
        return ask_wonder(meta, saved)

    result = diagnose_revision(rig.ctx, rig.experiment, ghost, ask=ask)
    print(result.diagnosis_md)
    print("adopted", result.adopted, "v2 onset", result.ghost and result.ghost.divergence_onset_s)
    total = sum(r.cost_usd for r in store.records())
    for r in store.records():
        print(
            f"{r.kind:8} {r.model:18} in {r.input_tokens} out {r.output_tokens} ${r.cost_usd:.4f}"
        )
    print(f"total spend: ${total:.4f}")


if __name__ == "__main__":
    main()
