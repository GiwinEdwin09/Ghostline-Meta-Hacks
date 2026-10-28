"""Read the ground truth written by `render_clip`, for tests that run without SAM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GroundTruth:
    """Ground truth of one synthetic clip. Pixel units follow OpenCV (centres at integers)."""

    def __init__(self, data: dict, root: Path):
        self.data = data
        self.root = root

    @classmethod
    def load(cls, json_path: str | Path) -> GroundTruth:
        path = Path(json_path)
        return cls(json.loads(path.read_text()), path.parent)

    @property
    def n_frames(self) -> int:
        return self.data["video"]["n_frames"]

    @property
    def size(self) -> tuple[int, int]:
        return self.data["video"]["width"], self.data["video"]["height"]

    @property
    def video_path(self) -> Path:
        return self.root / self.data["video"]["file"]

    @property
    def t_real(self) -> np.ndarray:
        """True capture time of every frame, in real seconds."""
        return np.asarray(self.data["timing"]["t_real_s"])

    @property
    def pts(self) -> np.ndarray:
        """Presentation timestamps stored in the file, in file seconds."""
        return np.asarray(self.data["timing"]["pts_s"])

    @property
    def time_factor(self) -> float:
        return self.data["timing"]["time_factor"]

    @property
    def sheet_corners_px(self) -> np.ndarray:
        """(n_frames, 4, 2) corners in the order top-left, top-right, bottom-right, bottom-left."""
        return np.asarray(self.data["sheet"]["corners_px"])

    @property
    def sheet_corners_world(self) -> np.ndarray:
        return np.asarray(self.data["sheet"]["corners_world_m"])

    @property
    def sheet_mask(self) -> np.ndarray:
        import cv2

        img = cv2.imread(str(self.root / self.data["sheet"]["mask_file"]), cv2.IMREAD_GRAYSCALE)
        return img > 127

    @property
    def plate(self) -> np.ndarray:
        """Background with the sheet and no moving object (BGR)."""
        import cv2

        return cv2.imread(str(self.root / self.data["sheet"]["plate_file"]))

    @property
    def tracks(self) -> list[str]:
        return list(self.data["tracks"])

    def track(self, key: str = "object") -> dict[str, np.ndarray]:
        """centroid_px, mask_centroid_px, area_px, visible and xyz_m (physics frame, metres)."""
        key = self.data["primary"] if key == "object" and key not in self.data["tracks"] else key
        return {k: np.asarray(v) for k, v in self.data["tracks"][key].items()}

    def series(self, name: str) -> np.ndarray:
        return np.asarray(self.data["series"][name])

    @property
    def params(self) -> dict:
        return self.data["params"]

    @property
    def events(self) -> dict:
        return self.data["events"]

    def object_mask(self, key: str, frame: int) -> np.ndarray:
        with np.load(self.root / self.data["masks_file"]) as z:
            h, w = z["shape"]
            return np.unpackbits(z[key][frame])[: h * w].reshape(h, w).astype(bool)

    def noisy_centroids(self, key: str, sigma_px: float, seed: int = 0) -> np.ndarray:
        """Mask centroids plus Gaussian noise, a stand-in for a noisier segmentation."""
        c = self.track(key)["centroid_px"]
        return c + np.random.default_rng(seed).normal(0, sigma_px, c.shape)
