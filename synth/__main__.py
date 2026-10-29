"""Command line: uv run python -m synth <system> --out DIR [options]."""

from __future__ import annotations

import argparse
import json

from .render import ClipSpec, render_clip
from .systems import SYSTEMS


def main() -> None:
    p = argparse.ArgumentParser(description="Render a synthetic physics clip with ground truth.")
    p.add_argument("system", choices=sorted(SYSTEMS))
    p.add_argument("--out", required=True, help="output directory (not committed)")
    p.add_argument("--name", default=None)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--capture-fps", type=float, default=60.0)
    p.add_argument("--export-fps", type=float, default=None, help="30 with 240 captured = slow-mo")
    p.add_argument("--size", default="854x480", help="upright WxH")
    p.add_argument("--rotation-tag", type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--hdr", action="store_true")
    p.add_argument("--noise", type=float, default=2.0, help="sensor noise sigma, grey levels")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--params", default="{}", help="system overrides as JSON, e.g. '{\"e\": 0.6}'")
    a = p.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    spec = ClipSpec(
        system=a.system, params=json.loads(a.params), duration_s=a.duration, size=(w, h),
        capture_fps=a.capture_fps, export_fps=a.export_fps, rotation_tag=a.rotation_tag,
        hdr=a.hdr, sensor_sigma=a.noise, seed=a.seed,
    )  # fmt: skip
    result = render_clip(spec, a.out, a.name)
    print(result.video)
    print(result.truth)


if __name__ == "__main__":
    main()
