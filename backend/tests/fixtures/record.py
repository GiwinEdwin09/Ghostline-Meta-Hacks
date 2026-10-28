"""Record the replay fixtures in `cache/` (this spends real money: about $0.2 in all).

    uv run --env-file .env python backend/tests/fixtures/record.py              # everything
    uv run --env-file .env python backend/tests/fixtures/record.py --only gravity,scene

It renders the synthetic clip, uploads its derived videos to the Files API, makes each call in
`scenarios.py` once in `record` mode, and writes `media.json`, `truth.json` and `stills/`. The
tests then replay those calls offline. Keys are read from the environment and never written
down; requests are stored with media replaced by hashes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent), str(ROOT / "scripts")]

import smoke  # noqa: E402
from fixtures import scenarios, synthetic_clip  # noqa: E402

from ghostline.config import get_settings  # noqa: E402
from ghostline.llm import (  # noqa: E402
    InMemorySpendStore,
    MetaClient,
    ReviewerClient,
    Runtime,
    sha256_file,
)
from ghostline.vision.frames import encode_image, jpeg_stills, read_frames  # noqa: E402


def prepare(out: Path) -> dict:
    """Render the clip, cut the derived files, and write stills, hashes and ground truth."""
    media = smoke.prepare_media(None, out, scenarios.BALL_PROMPT, scenarios.SHEET_PROMPT)
    (scenarios.FIXTURES / "stills").mkdir(exist_ok=True)
    for path, still in zip(scenarios.STILLS, jpeg_stills(media.master, [0, 150, 299]), strict=True):
        path.write_bytes(still)
    frame = read_frames(media.sam_clip, [45])[45]
    scenarios.SHEET_FRAME.write_bytes(encode_image(frame, ".jpg"))
    scenarios.TRUTH.write_text(media.truth.to_json())
    settings = get_settings()
    info = {
        "wonder_480_sha256": sha256_file(media.wonder_480),
        "sam_3s_sha256": sha256_file(media.sam_clip),
        "frame_size": [synthetic_clip.WIDTH, synthetic_clip.HEIGHT],
        "models": {
            "physicist": settings.physicist_model,
            "scene": settings.scene_model,
            "sam": settings.sam_model,
            "reviewer": settings.reviewer_model,
        },
    }
    scenarios.MEDIA_INFO.write_text(json.dumps(info, indent=1))
    return {"media": media, "info": info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", default="", help="comma-separated scenario names")
    args = parser.parse_args()
    only = {n for n in args.only.split(",") if n}

    store = InMemorySpendStore()
    settings = get_settings()
    runtime = Runtime.create(settings, mode="record", cache_dir=scenarios.CACHE_DIR, store=store)
    meta, reviewer = MetaClient(runtime), ReviewerClient(runtime)

    out = Path(tempfile.mkdtemp(prefix="ghostline-fixtures-"))
    prepared = prepare(out)
    media = prepared["media"]
    uploaded: list[str] = []

    def want(name: str) -> bool:
        return not only or name in only

    def upload(path: Path) -> str:
        file_id = meta.upload_file(path).id
        uploaded.append(file_id)
        return file_id

    try:
        if want("gravity"):
            scenarios.gravity(meta)
        if want("tools"):
            scenarios.tool_roundtrip(meta)
        if want("scene"):
            scenarios.scene(meta, upload(media.wonder_480))
        if want("sam"):
            file_id = upload(media.sam_clip)
            scenarios.sam_video(meta, scenarios.BALL_PROMPT, file_id)
            scenarios.sam_video(meta, scenarios.DOT_PROMPT, file_id)
            scenarios.sam_sheet(meta)
        if want("rescue"):
            scenarios.rephrase(meta)
            scenarios.box(meta)
        if want("reviewer"):
            scenarios.reviewer_reading(reviewer)
    finally:
        for file_id in uploaded:
            try:
                meta.delete_file(file_id)
            except Exception as error:  # uploads expire in 7 days anyway
                print(f"could not delete {file_id}: {type(error).__name__}")

    total = 0.0
    for record in store.records():
        total += record.cost_usd
        print(f"{record.kind:10} {record.phase or '':8} {record.model:22} ${record.cost_usd:.4f}")
    print(f"total spend: ${total:.4f}")


if __name__ == "__main__":
    main()
