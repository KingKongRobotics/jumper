"""The brush in the carried claw: a thick handle gripped by the closed claw, and a
black cone of hair that may sink into the floor.

## Where it sits

In `LF_palm_link`'s frame the palm runs along +y to y = 0.144 m, the fixed jaw is
on +z and the finger closes from -z. The handle is 13 mm thick and lies along the
fixed jaw's channel (`MOUTH_Z`), and the finger closes **onto** it. Measured on
the V1.6 visual meshes along the handle's whole length, and the largest 无 the
reach band then holds (`write.py`, the legs held, 2026-10-09):

    FINGER_HOLD   finger against the handle   无
       0.10       6.4 mm into it               13.6 cm   the finger's limit; the
                                                         claw closed through the brush
       0.00       3.3 mm into it               13.6 cm
      -0.025      0.8 mm into it -- touching   12.2 cm
      -0.05       1.7 mm clear                 10.6 cm

Opening it costs size because the finger's tip hangs lower and meets the floor in
more of the writing poses. The handle starts at y = 55 mm, inside the mouth: from
30 mm it ran 2.4 mm into the finger's hinge at any opening.

The brush is **welded to the palm**, not held by the finger's friction: a grip
that could slip would turn every stroke into a measurement of the grip.

## The hair, and why nothing collides

The hair is a cone, apex down, black. It does not collide with anything: when it
was a colliding tip, pressing it into the floor pushed back on the arm, the
trunk turned under it and the stroke wandered (无's long middle stroke turned the
trunk 48 deg with a fixed-depth press). Now the apex is simply put **below** the
floor, the way wet hair splays, and the ink is the cone's section at the floor:
`section_width`. Deeper is wider; nothing pushes the robot.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

PALM_BODY = "LF_palm_link"
BRUSH_BODY = "brush"
#: The brush's geoms; neither collides, `arm.self_clearance` measures them.
GEOMS = ("brush_handle", "brush_hair")
TIP_SITE = "brush_tip"      # the cone's apex
BASE_SITE = "brush_base"    # the centre of the cone's base, where the hair meets the handle

#: The handle, in the palm's frame: a capsule along +y, centred between the jaws.
MOUTH_Z = -0.0034
HANDLE_Y = (0.055, 0.160)
HANDLE_RADIUS = 0.0065
#: The hair: a cone from its base at CONE_Y to its apex CONE_LENGTH further out.
CONE_Y = 0.160
CONE_LENGTH = 0.026
CONE_RADIUS = 0.014
TIP_Y = CONE_Y + CONE_LENGTH
#: tan of the cone's half-angle: the section's radius per metre of hair below the floor.
CONE_SLOPE = CONE_RADIUS / CONE_LENGTH

#: A bamboo handle and wet hair: ~10 g and ~5 g.
HANDLE_MASS = 0.010
HAIR_MASS = 0.005

#: Finger target with the handle in the mouth, rad: closed onto the handle, not through it.
FINGER_HOLD = -0.025

CONE_SEGMENTS = 24


def add_brush(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    palm = spec.body(PALM_BODY)
    if palm is None:
        raise KeyError(f"{PALM_BODY} is not in the robot's spec")
    ring = [(CONE_RADIUS * math.cos(a), 0.0, CONE_RADIUS * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, CONE_SEGMENTS, endpoint=False)]
    mesh = spec.add_mesh(name="brush_hair")
    # In the brush body's frame: base ring at y = 0, apex at y = CONE_LENGTH.
    mesh.uservert = [c for v in [*ring, (0.0, CONE_LENGTH, 0.0)] for c in v]
    body = palm.add_body(name=BRUSH_BODY)
    body.add_geom(
        name="brush_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, HANDLE_Y[0], MOUTH_Z, 0, HANDLE_Y[1], MOUTH_Z],
        size=[HANDLE_RADIUS, 0, 0], mass=HANDLE_MASS,
        contype=0, conaffinity=0, group=2, rgba=[0.62, 0.45, 0.24, 1],
    )
    body.add_geom(
        name="brush_hair", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="brush_hair",
        pos=[0, CONE_Y, MOUTH_Z], mass=HAIR_MASS,
        contype=0, conaffinity=0, group=2, rgba=[0.04, 0.04, 0.04, 1],
    )
    body.add_site(name=TIP_SITE, pos=[0, TIP_Y, MOUTH_Z], size=[0.002, 0, 0], group=4)
    body.add_site(name=BASE_SITE, pos=[0, CONE_Y, MOUTH_Z], size=[0.002, 0, 0], group=4)
    return spec


def section_width(apex: np.ndarray, base: np.ndarray) -> float:
    """Width of the ink the cone leaves: twice its radius where its axis meets the
    floor, 0 while the apex is above it. The section of a tilted cone is an
    ellipse; this is its width across the stroke's direction when the brush leans
    along the stroke, and within the ellipse's spread otherwise -- close enough for
    ink, and the same rule the renderer and the ink export use."""
    if apex[2] >= 0.0:
        return 0.0
    axis = base - apex
    if axis[2] <= 1e-9:
        return 2.0 * CONE_RADIUS  # lying flat: the whole cone is down
    t = min(1.0, -apex[2] / axis[2])  # fraction of the way from apex to base
    return 2.0 * CONE_SLOPE * t * float(np.linalg.norm(axis))


def ink_point(apex: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Where the ink is centred: the cone's axis at the floor, or under the apex
    while the apex is above it. A leaning brush sunk 10 mm puts this about a
    centimetre from the point under its apex, so this -- not the apex -- is what
    has to follow the stroke."""
    axis = base - apex
    if apex[2] >= 0.0 or axis[2] <= 1e-9:
        return np.array([apex[0], apex[1], 0.0])
    t = min(1.0, -apex[2] / axis[2])
    return apex + t * axis


def apply(cfg) -> None:
    """Put the brush on `jumper.five_foot`'s robot in `cfg`."""
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn
    robot.spec_fn = lambda: add_brush(base_spec_fn())
