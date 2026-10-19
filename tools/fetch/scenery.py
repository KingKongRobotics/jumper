"""Dunes, palms and rocks around the beach: a desert to wander rather than a void.

Dressing, not props: everything here is added through the scene's `decorate` hook
(`scenes/registry.py::_install_spec_edits`), so it is drawn and it is in the depth
sensor's picture, and the robot cannot touch it (`contype` 0). The layout keeps
everything at least two metres off the wander path: `sense.py` takes the biggest
cluster within 1.5 m as the mug, and a palm trunk inside that range would be a
mug. Obstacles the crab has to walk round come later, as physical props.

Shapes are MuJoCo primitives: a dune is an ellipsoid sunk to a third of its
height; a palm is a tilted capsule trunk with a rosette of drooping ellipsoid
fronds and three coconuts; a rock is a squashed ellipsoid. The sand colour is
the beach ground's.
"""
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

SAND = [0.80, 0.70, 0.52, 1.0]
TRUNK = [0.42, 0.30, 0.18, 1.0]
FROND = [0.18, 0.46, 0.20, 1.0]
COCONUT = [0.36, 0.24, 0.12, 1.0]
ROCK = [0.46, 0.42, 0.38, 1.0]

#: (x, y, rx, ry, rz, yaw deg): big smooth mounds well beyond the sensor's search range
#: (the dunes at (9, 3) and (15, 1) reached the route's centreline between x 7 and 9: the tray
#: of the single-shot film stood on a slope, the room walk steered off it, film4 -- moved out)
DUNES = [(10.5, 5.0, 5.0, 3.0, 1.1, 20), (11.0, -4.5, 6.0, 3.5, 1.4, -15), (4.0, 7.5, 4.0, 2.5, 0.8, 70),
         (-5.0, -6.0, 5.0, 3.0, 1.0, 40), (-7.0, 4.0, 4.5, 3.0, 1.2, -30), (19.0, 2.0, 7.0, 4.0, 1.6, 5),
         (3.0, -8.0, 4.0, 2.5, 0.9, 100)]
#: (x, y, height, lean deg, lean toward deg): trunks
PALMS = [(1.8, 3.2, 2.2, 10, 200), (5.0, -2.8, 2.6, 8, 60), (6.5, 2.4, 2.0, 14, 300), (-2.5, 2.6, 2.4, 6, 120),
         (9.5, -0.5, 2.8, 10, 250)]
#: (x, y, rx, ry, rz, yaw deg)
ROCKS = [(2.2, -2.6, 0.45, 0.34, 0.30, 30), (-1.6, -2.2, 0.36, 0.28, 0.26, -50), (7.0, 3.0, 0.55, 0.40, 0.34, 80),
         (3.6, 3.4, 0.32, 0.26, 0.24, 10)]   # (the big one stood at (7.0, 0.8), on the film's tray: moved out)   # all taller than a mug: a 16 cm rock was "sighted" once (wander3)
FRONDS = 7
#: (x, y, rx, ry, rz): low bushes, a cluster of three ellipsoids each
BUSHES = [(3.2, 2.6, 0.30, 0.26, 0.22), (6.3, -2.5, 0.34, 0.28, 0.24), (0.8, -2.8, 0.28, 0.24, 0.20),
          (8.8, 2.7, 0.32, 0.26, 0.22), (-1.5, 2.9, 0.30, 0.25, 0.21), (11.5, -2.9, 0.33, 0.27, 0.23)]
BUSH = [0.36, 0.42, 0.22, 1.0]
if os.environ.get("SCENERY_DENSE") == "1":
    # The single-shot film walks 8 m: more of everything along the route, all of it still
    # 2.2 m or more off the centreline (the mug search reaches 1.5 m, the flag 2.5 m... a
    # palm trunk inside that would be a pillar to the room walk).
    PALMS = PALMS + [(3.0, -3.1, 2.3, 9, 150), (7.2, 3.3, 2.5, 12, 320), (9.8, -2.7, 2.1, 7, 40),
                     (0.6, 3.6, 2.4, 10, 220), (-2.2, -3.2, 2.2, 8, 100), (12.5, 2.6, 2.7, 11, 290)]
    ROCKS = ROCKS + [(4.6, 2.5, 0.38, 0.30, 0.27, 50), (6.0, -2.7, 0.42, 0.33, 0.30, -20), (8.6, 2.3, 0.35, 0.28, 0.25, 120),
                     (2.4, -2.5, 0.40, 0.31, 0.28, 75), (10.2, -2.3, 0.36, 0.29, 0.26, -60), (0.4, 2.7, 0.33, 0.27, 0.24, 15),
                     (11.8, 2.4, 0.45, 0.35, 0.31, 95)]


def _quat_z(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _quat(axis: np.ndarray, deg: float) -> list[float]:
    axis = axis / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return [math.cos(h), *(math.sin(h) * axis)]


def _mul(a: list[float], b: list[float]) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _geom(spec: Any, **kw: Any) -> None:
    spec.worldbody.add_geom(contype=0, conaffinity=0, mass=0.0, group=2, **kw)


def _palm(spec: Any, i: int, x: float, y: float, h: float, lean: float, toward: float) -> None:
    lean_axis = np.array([-math.sin(math.radians(toward)), math.cos(math.radians(toward)), 0.0])
    q = _quat(lean_axis, lean)
    tip = np.array([x, y, 0.0]) + h * np.array([math.sin(math.radians(lean)) * math.cos(math.radians(toward)),
                                                math.sin(math.radians(lean)) * math.sin(math.radians(toward)),
                                                math.cos(math.radians(lean))])
    mid = (np.array([x, y, 0.0]) + tip) / 2
    _geom(spec, name=f"palm{i}_trunk", type=3, size=[0.07, h / 2, 0.0], pos=mid.tolist(), quat=q, rgba=TRUNK)
    for k in range(FRONDS):
        yaw = 360.0 * k / FRONDS + 17 * i
        droop = 28.0 + 8 * ((k * 5 + i) % 3)
        length = 1.1
        d = np.array([math.cos(math.radians(yaw)) * math.cos(math.radians(droop)),
                      math.sin(math.radians(yaw)) * math.cos(math.radians(droop)), -math.sin(math.radians(droop))])
        centre = tip + 0.5 * length * d
        qf = _mul(_quat_z(yaw), _quat(np.array([0.0, 1.0, 0.0]), droop))
        _geom(spec, name=f"palm{i}_frond{k}", type=4, size=[length / 2, 0.20, 0.035], pos=centre.tolist(),
              quat=qf, rgba=FROND)
    for k in range(3):
        a = math.radians(120 * k + 40 * i)
        _geom(spec, name=f"palm{i}_nut{k}", type=2, size=[0.09, 0.0, 0.0],
              pos=(tip + np.array([0.12 * math.cos(a), 0.12 * math.sin(a), -0.12])).tolist(), rgba=COCONUT)


def decorate(spec: Any) -> None:
    import mujoco

    for i, (x, y, rx, ry, rz, yaw) in enumerate(DUNES):
        _geom(spec, name=f"dune{i}", type=4, size=[rx, ry, rz], pos=[x, y, -rz * 0.35], quat=_quat_z(yaw), rgba=SAND)
    for i, p in enumerate(PALMS):
        _palm(spec, i, *p)
    for i, (x, y, rx, ry, rz) in enumerate(BUSHES):
        for k, (dx, dy, f) in enumerate(((0.0, 0.0, 1.0), (rx * 0.8, ry * 0.3, 0.7), (-rx * 0.5, -ry * 0.7, 0.6))):
            _geom(spec, name=f"bush_{i}_{k}", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[rx * f, ry * f, rz * f],
                  pos=[x + dx, y + dy, rz * f * 0.6], rgba=BUSH)
    for i, (x, y, rx, ry, rz, yaw) in enumerate(ROCKS):
        _geom(spec, name=f"rock{i}", type=4, size=[rx, ry, rz], pos=[x, y, rz * 0.55], quat=_quat_z(yaw), rgba=ROCK)


def apply_scenery(env_cfg: Any) -> None:
    from scenes.registry import _install_spec_edits

    _install_spec_edits(env_cfg.scene, None, decorate)
