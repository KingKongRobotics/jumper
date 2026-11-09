"""The brush in the carried claw: a thick handle gripped by the closed claw, and a
black tuft of hair that may sink into the floor.

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

The hair is black, tip down. It does not collide with anything: when it was a
colliding tip, pressing it into the floor pushed back on the arm, the trunk
turned under it and the stroke wandered (无's long middle stroke turned the
trunk 48 deg with a fixed-depth press). Now the tip is simply put **below** the
floor, the way wet hair splays, and the ink is the hair's section at the floor:
`section_width`. Deeper is wider; nothing pushes the robot.

## Its shape

As wide as the handle at its base and bellied towards a point, like a real
brush's (笔肚, 笔锋): `hair_radius`. As a straight cone that wide, its section
grows only in proportion to the depth, and the share of it under the floor is
the stroke's width over the base's whatever the lean -- 58% of its length for a
7.8 mm stroke (median; 84% at the heads, 跳跳, claw2_tt, 2026-10-10), a brush
seen sunk into the stone. Bellied, the section widens fastest near the tip.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

PALM_BODY = "LF_palm_link"
BRUSH_BODY = "brush"
#: The brush's geoms; neither collides, `arm.self_clearance` measures them.
GEOMS = ("brush_handle", "brush_hair")
TIP_SITE = "brush_tip"      # the hair's tip
BASE_SITE = "brush_base"    # the centre of the hair's base, where it meets the handle

#: The handle, in the palm's frame: a capsule along +y, centred between the jaws.
MOUTH_Z = -0.0034
HANDLE_Y = (0.055, 0.160)
HANDLE_RADIUS = 0.0065
#: The hair: from its base at HAIR_Y to its tip HAIR_LENGTH further out, as wide as
#: the handle it grows from (`hair_radius`). It was a cone 28 mm wide at first,
#: twice the handle: a tuft rather than a brush in the film.
HAIR_Y = 0.160
HAIR_LENGTH = 0.026
HAIR_RADIUS = HANDLE_RADIUS
TIP_Y = HAIR_Y + HAIR_LENGTH

#: A bamboo handle and wet hair: ~10 g and ~5 g.
HANDLE_MASS = 0.010
HAIR_MASS = 0.005

#: Finger target with the handle in the mouth, rad: closed onto the handle, not through it.
FINGER_HOLD = -0.025

HAIR_SEGMENTS = 24
HAIR_RINGS = 12


def hair_radius(s: float) -> float:
    """The hair's radius `s` from its tip along its axis: HAIR_RADIUS at the base
    and sin-shaped towards the tip, where it is a point with the slope of a 21 deg
    half-angle cone."""
    return HAIR_RADIUS * math.sin(0.5 * math.pi * min(max(s / HAIR_LENGTH, 0.0), 1.0))


def hair_reach(radius: float) -> float:
    """How far from its tip the hair is `radius` wide: `hair_radius` inverted."""
    return HAIR_LENGTH * 2.0 / math.pi * math.asin(min(max(radius / HAIR_RADIUS, 0.0), 1.0))


def add_brush(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    palm = spec.body(PALM_BODY)
    if palm is None:
        raise KeyError(f"{PALM_BODY} is not in the robot's spec")
    # In the brush body's frame: the base ring at y = 0, the tip at y = HAIR_LENGTH.
    # Convex (the radius is concave along the axis), so its hull is itself.
    verts = [(hair_radius(s) * math.cos(a), HAIR_LENGTH - s, hair_radius(s) * math.sin(a))
             for s in np.linspace(HAIR_LENGTH, 0.0, HAIR_RINGS, endpoint=False)
             for a in np.linspace(0, 2 * math.pi, HAIR_SEGMENTS, endpoint=False)]
    mesh = spec.add_mesh(name="brush_hair")
    mesh.uservert = [c for v in [*verts, (0.0, HAIR_LENGTH, 0.0)] for c in v]
    body = palm.add_body(name=BRUSH_BODY)
    body.add_geom(
        name="brush_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, HANDLE_Y[0], MOUTH_Z, 0, HANDLE_Y[1], MOUTH_Z],
        size=[HANDLE_RADIUS, 0, 0], mass=HANDLE_MASS,
        contype=0, conaffinity=0, group=2, rgba=[0.62, 0.45, 0.24, 1],
    )
    body.add_geom(
        name="brush_hair", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="brush_hair",
        pos=[0, HAIR_Y, MOUTH_Z], mass=HAIR_MASS,
        contype=0, conaffinity=0, group=2, rgba=[0.04, 0.04, 0.04, 1],
    )
    body.add_site(name=TIP_SITE, pos=[0, TIP_Y, MOUTH_Z], size=[0.002, 0, 0], group=4)
    body.add_site(name=BASE_SITE, pos=[0, HAIR_Y, MOUTH_Z], size=[0.002, 0, 0], group=4)
    return spec


def section_width(apex: np.ndarray, base: np.ndarray) -> float:
    """Width of the ink the hair leaves: twice its radius where its axis meets the
    floor, 0 while the tip is above it. The section of a tilted brush is close to
    an ellipse; this is its width across the stroke's direction when the brush
    leans along the stroke, and within the ellipse's spread otherwise -- close
    enough for ink, and the same rule the renderer and the ink export use."""
    if apex[2] >= 0.0:
        return 0.0
    axis = base - apex
    if axis[2] <= 1e-9:
        return 2.0 * HAIR_RADIUS  # lying flat: the whole hair is down
    t = min(1.0, -apex[2] / axis[2])  # fraction of the way from tip to base
    return 2.0 * hair_radius(t * float(np.linalg.norm(axis)))


def depth_for(width: float, apex: np.ndarray, base: np.ndarray) -> float:
    """How far below the floor the tip goes for ink `width` wide, the brush leaning
    as it does from `apex` to `base`: `section_width` inverted."""
    axis = base - apex
    n = float(np.linalg.norm(axis))
    return hair_reach(width / 2.0) * max(float(axis[2]) / n, 0.0) if n > 0 else 0.0


def ink_point(apex: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Where the ink is centred: the hair's axis at the floor, or under the apex
    while the apex is above it. A leaning brush sunk 10 mm puts this about a
    centimetre from the point under its apex, so this -- not the apex -- is what
    has to follow the stroke."""
    axis = base - apex
    if apex[2] >= 0.0 or axis[2] <= 1e-9:
        return np.array([apex[0], apex[1], 0.0])
    t = min(1.0, -apex[2] / axis[2])
    return apex + t * axis


#: Put down before the jump (`write.py --finale run`), the brush lies this far out
#: from where the claw held it. Laid where it was held, the claw landed 6.4 mm into
#: it after the jump; 5 cm out, its meshes stayed 23.5 mm off it (无, 2026-10-11,
#: the claw's collision meshes against the lying brush's axis).
PUT_ASIDE = 0.05                  # m


def lying(m: mujoco.MjModel, d: mujoco.MjData,
          prefix: str = "robot/") -> tuple[np.ndarray, np.ndarray]:
    """The brush body's pose put down on the floor, (position, rotation), from where
    the claw holds it in `d` (its kinematics done): the axis turned level in the
    vertical plane it was in, the handle resting on the stone, the middle where it
    was but PUT_ASIDE out on the brush's +z, the side away from the forearm (5 cm
    down -z of it in the jump's stance). The simulation has no loose brush:
    `render.py` draws it there and `write.py` runs its laps around it."""
    b = m.body(prefix + BRUSH_BODY).id
    rot = d.xmat[b].reshape(3, 3)
    along = np.array([rot[0, 1], rot[1, 1], 0.0])
    along /= np.linalg.norm(along)
    aside = np.array([rot[0, 2], rot[1, 2], 0.0])
    aside -= (aside @ along) * along
    aside /= np.linalg.norm(aside)
    up = np.array([0.0, 0.0, 1.0])
    level = np.column_stack([np.cross(along, up), along, up])
    mid = (HANDLE_Y[0] + TIP_Y) / 2
    centre = d.xpos[b] + rot @ np.array([0.0, mid, MOUTH_Z]) + PUT_ASIDE * aside
    origin = np.array([*(centre[:2] - mid * along[:2]), HANDLE_RADIUS - MOUTH_Z])
    return origin, level


def lying_ends(pose: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(2, 3): the lying brush's handle end and hair tip, world, from `lying`."""
    origin, rot = pose
    return np.array([origin + rot @ np.array([0.0, y, MOUTH_Z]) for y in (HANDLE_Y[0], TIP_Y)])


def apply(cfg) -> None:
    """Put the brush on `jumper.five_foot`'s robot in `cfg`."""
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn
    robot.spec_fn = lambda: add_brush(base_spec_fn())
