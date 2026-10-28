"""Decoding with PyAV, so every frame's index and timestamp come from the same pass."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import cv2
import numpy as np


def iter_frames(
    path: str | Path, width: int | None = None, gray: bool = False
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame index, presentation seconds, image) in display order.

    With `width`, frames are scaled to that width (aspect kept). Colour frames are BGR uint8.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else float("nan")
            if width:
                height = max(2, round(frame.height * width / frame.width))
                frame = frame.reformat(width=width, height=height)
            yield i, t, frame.to_ndarray(format="gray" if gray else "bgr24")


def count_frames(path: str | Path) -> int:
    """Number of video frames, from packets (no decoding)."""
    with av.open(str(path)) as container:
        return sum(1 for packet in container.demux(video=0) if packet.size)


def extract_frames(path: str | Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Decode only the wanted frame indices (BGR) in one pass."""
    wanted = set(indices)
    last = max(wanted)
    found: dict[int, np.ndarray] = {}
    for i, _, img in iter_frames(path):
        if i in wanted:
            found[i] = img
        if i >= last:
            break
    return found


def spread_indices(n_frames: int, n: int) -> list[int]:
    return sorted({int(round(v)) for v in np.linspace(0, n_frames - 1, min(n, n_frames))})


def median_image(path: str | Path, n_frames: int, n: int = 15) -> np.ndarray:
    """Per-pixel median of `n` frames spread over the clip: removes the moving object."""
    frames = extract_frames(path, spread_indices(n_frames, n))
    return np.median(np.stack(list(frames.values())), axis=0).astype(np.uint8)


def save_stills(
    path: str | Path, indices: list[int], out_dir: str | Path, quality: int = 92
) -> list[Path]:
    """Cache full-resolution JPEG stills named still_<index>.jpg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, img in sorted(extract_frames(path, indices).items()):
        dest = out / f"still_{i:05d}.jpg"
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paths.append(dest)
    return paths
