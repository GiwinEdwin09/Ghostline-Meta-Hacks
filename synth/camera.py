"""Pinhole camera for the synthetic scenes.

World frame: origin at the sheet's centre, X right, Y up, Z toward the camera. The sheet lies in
the plane Z = 0; the moving object lives in the plane Z = d (d metres in front of the sheet).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

A4_M = (0.210, 0.297)


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    f_px: float
    center: np.ndarray  # camera centre in world coordinates, shape (3,)
    R: np.ndarray  # world -> camera rotation; rows are right, down, forward
    shift: tuple[float, float] = (0.0, 0.0)  # extra principal-point shift in pixels

    @classmethod
    def look_at(
        cls,
        width: int,
        height: int,
        target: tuple[float, float],
        distance: float,
        f35: float = 26.0,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> Camera:
        """Camera `distance` m from `target` (on the sheet plane), looking at it.

        Azimuth moves the camera sideways and elevation upward; both give perspective.
        """
        az, el, roll = np.radians([azimuth_deg, elevation_deg, roll_deg])
        tgt = np.array([target[0], target[1], 0.0])
        offset = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        center = tgt + distance * offset
        forward = (tgt - center) / np.linalg.norm(tgt - center)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        right, down = (
            np.cos(roll) * right + np.sin(roll) * down,
            -np.sin(roll) * right + np.cos(roll) * down,
        )
        return cls(width, height, width * f35 / 36.0, center, np.stack([right, down, forward]))

    @property
    def K(self) -> np.ndarray:
        cx = (self.width - 1) / 2 + self.shift[0]  # OpenCV convention: pixel centres at integers
        cy = (self.height - 1) / 2 + self.shift[1]
        return np.array([[self.f_px, 0, cx], [0, self.f_px, cy], [0, 0, 1.0]])

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(pts) - self.center) @ self.R.T

    def depth(self, pts: np.ndarray) -> np.ndarray:
        return self.to_camera(pts)[:, 2]

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2)."""
        pc = self.to_camera(pts)
        k = self.K
        return np.stack(
            [k[0, 2] + self.f_px * pc[:, 0] / pc[:, 2], k[1, 2] + self.f_px * pc[:, 1] / pc[:, 2]],
            axis=1,
        )

    def plane_homography(self, z: float = 0.0) -> np.ndarray:
        """3x3 matrix taking world (X, Y) on the plane Z = z to pixels."""
        r1, r2, r3 = self.R[:, 0], self.R[:, 1], self.R[:, 2]
        t = -self.R @ self.center
        return self.K @ np.stack([r1, r2, r3 * z + t], axis=1)

    def with_shift(self, dx: float, dy: float) -> Camera:
        return Camera(self.width, self.height, self.f_px, self.center, self.R, (dx, dy))


def sheet_corners_world(
    center: tuple[float, float] = (0.0, 0.0),
    rot_deg: float = 0.0,
    landscape: bool = False,
    size_m: tuple[float, float] = A4_M,
) -> np.ndarray:
    """Sheet corners (4, 3) in world coordinates, counter-clockwise seen from the camera."""
    w, h = (size_m[1], size_m[0]) if landscape else size_m
    local = np.array([[-w / 2, h / 2], [w / 2, h / 2], [w / 2, -h / 2], [-w / 2, -h / 2]])
    a = np.radians(rot_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    xy = local @ rot.T + np.asarray(center)
    return np.column_stack([xy, np.zeros(4)])
