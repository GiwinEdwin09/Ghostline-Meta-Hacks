"""Frame access with PyAV: decode on demand, never write every frame to disk (build plan Part 5)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np


def _open(path: str | Path):
    import av

    container = av.open(str(path))
    return container, container.streams.video[0]


def count_frames(path: str | Path) -> int:
    """The number of decodable frames (the container's count is not always filled in)."""
    container, stream = _open(path)
    with container:
        if stream.frames:
            return stream.frames
        return sum(1 for _ in container.decode(stream))


def video_size(path: str | Path) -> tuple[int, int]:
    """(width, height) in pixels as displayed."""
    container, stream = _open(path)
    with container:
        return stream.codec_context.width, stream.codec_context.height


def read_frames(path: str | Path, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decode the frames at `indices` (0-based) into BGR uint8 arrays, in one pass."""
    wanted = set(indices)
    found: dict[int, np.ndarray] = {}
    if not wanted:
        return found
    container, stream = _open(path)
    with container:
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                found[index] = frame.to_ndarray(format="bgr24")
                if len(found) == len(wanted):
                    break
    return found


def spread_indices(n_frames: int, count: int) -> list[int]:
    """`count` frame indices spread evenly over the clip, including the first and last."""
    count = min(count, n_frames)
    if count <= 1:
        return [0]
    return sorted({round(i * (n_frames - 1) / (count - 1)) for i in range(count)})


def median_frame(path: str | Path, count: int = 15, n_frames: int | None = None) -> np.ndarray:
    """The per-pixel median of `count` frames spread over the clip. The median removes the
    moving object, which leaves the static scene and the calibration sheet."""
    n_frames = n_frames or count_frames(path)
    frames = read_frames(path, spread_indices(n_frames, count))
    stack = np.stack([frames[i] for i in sorted(frames)])
    return np.median(stack, axis=0).astype(np.uint8)


def encode_image(image: np.ndarray, ext: str = ".png", jpeg_quality: int = 90) -> bytes:
    """Encode a BGR array as PNG or JPEG bytes."""
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext == ".jpg" else []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"could not encode an image as {ext}")
    return buffer.tobytes()


def jpeg_stills(
    path: str | Path, indices: Sequence[int], max_side: int | None = None
) -> list[bytes]:
    """JPEG bytes for the frames at `indices`, optionally shrunk so the long side is `max_side`."""
    frames = read_frames(path, indices)
    out = []
    for index in indices:
        image = frames[index]
        if max_side and max(image.shape[:2]) > max_side:
            scale = max_side / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(encode_image(image, ".jpg"))
    return out
