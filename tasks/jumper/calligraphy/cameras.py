"""The two shots, as one definition the renderer and the ink export both read.

The ink is painted afterwards, onto the rendered frames, so the stroke export has
to put each point on the **same pixel** the renderer put the floor under it. Two
copies of a camera -- one in the render, one in the export -- would agree until
one was moved. So both read `Shot`, and `project` is checked against the
renderer by `tools/render.py --check` (a marker drawn at a known point has to land
on the pixel `project` gives it).

## The convention

A MuJoCo free camera looks along

    forward = (cos(el) cos(az), cos(el) sin(az), sin(el))

from `lookat - distance * forward`, with the image's up the world's +z projected
onto the image plane -- and, looking straight down, the horizontal heading `az`.
So `top` at az = 0 puts world +x at the top of the frame: the character's up
(see `hanzi.py`), reading the right way round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Shot:
    name: str
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float    # deg
    elevation: float  # deg
    fovy: float = 45.0  # deg
    width: int = 960
    height: int = 720

    def forward(self) -> np.ndarray:
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                         math.sin(el)])

    def position(self) -> np.ndarray:
        return np.asarray(self.lookat) - self.distance * self.forward()

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, forward), world frame."""
        f = self.forward()
        right = np.cross(f, (0.0, 0.0, 1.0))
        if np.linalg.norm(right) < 1e-6:  # straight down: up is the heading
            az = math.radians(self.azimuth)
            right = np.cross(f, (math.cos(az), math.sin(az), 0.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        return right, up, f

    def focal_px(self) -> float:
        return (self.height / 2) / math.tan(math.radians(self.fovy) / 2)

    def intrinsics(self) -> np.ndarray:
        """The 3x3 pinhole matrix, pixels, origin at the top-left corner."""
        fp = self.focal_px()
        return np.array([[fp, 0.0, self.width / 2], [0.0, fp, self.height / 2],
                         [0.0, 0.0, 1.0]])

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (n, 3) -> pixels (n, 2), x right, y down."""
        right, up, f = self.basis()
        d = np.asarray(points, dtype=float) - self.position()
        z = d @ f
        fp = self.focal_px()
        return np.column_stack([self.width / 2 + fp * (d @ right) / z,
                                self.height / 2 - fp * (d @ up) / z])

    def mjv_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.distance = self.distance
        cam.azimuth = self.azimuth
        cam.elevation = self.elevation
        return cam

    def to_json(self) -> dict:
        right, up, f = self.basis()
        return {
            "name": self.name, "width": self.width, "height": self.height,
            "fovy_deg": self.fovy, "position": self.position().round(6).tolist(),
            "right": right.round(6).tolist(), "up": up.round(6).tolist(),
            "forward": f.round(6).tolist(), "intrinsics": self.intrinsics().round(4).tolist(),
            "pixel_from_world": "u = cx + f*(d.right)/(d.forward), "
                                "v = cy - f*(d.up)/(d.forward), d = p - position",
        }


def shots(origin: tuple[float, float], size: float) -> tuple[Shot, Shot]:
    """The overhead shot framing the character and where the robot stands to
    write it, and a low shot from the robot's side, near the floor."""
    cx, cy = origin
    # Overhead: the character in the upper part of the frame, and below it the
    # band 0.1-0.4 m behind (towards -x) where the robot stands to write; 4:3,
    # +x up. The robot walks out of frame at the bottom between stretches.
    top = Shot("top", (cx - 0.12, cy, 0.0), distance=0.55 + size, azimuth=0.0,
               elevation=-90.0)
    # Low: from beyond the character's top-left corner, about 15 cm off the floor,
    # looking back across the ink at the robot writing towards the camera.
    low = Shot("low", (cx - 0.08, cy, 0.02), distance=0.35 + size, azimuth=-150.0,
               elevation=-12.0)
    return top, low
