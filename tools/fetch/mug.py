"""A mug with its handle toward the robot: a prop the claw can hang, not only pinch.

The row's props are friction-only holds and a 0.6 N*m squeeze drops every one of
them on the lift (tools/fetch/fetch.py's history). A handle gives form closure:
the jaws pinch the handle's outer bar across the mouth, and when the body pitches
up the mug slides down the bar until the handle's lower arm meets the pads from
below. From then on the weight hangs on the jaws and the pinch only has to keep
the bar between them.

The look is Andrew's model (`Coffee Mug.glb`, converted once into
`assets/mug_mesh.npz` + `assets/mug_tex.png` by `convert_mug.py`): a textured
mesh drawn over the collision shapes, which are the row's own kind of geom. The
cup is the model's; the handle is a loop drawn from the same numbers as its
collision bars (convert_mug.py says why the model's own C handle is not used), so
the claw meets what the picture shows.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from tasks.jumper.five_foot import objects as O

ASSETS = Path(__file__).resolve().parent / "assets"

BODY_R, BODY_HALF_H = 0.040, 0.050   # the model's cup: 80 mm across, 100 tall
BAR = 0.004                          # the loop's bars, 8 mm square section
HANDLE_OUT = 0.046                   # the outer bar's centre this far outside the wall: leaves the anvil 2 cm off the wall
LOOP_HALF_Z = 0.022                  # the top and bottom bars at +-22 mm: the lower one hangs on the finger's underside,
                                     # 17 mm under the pinch, so the lifted mug hangs level (mug run 6: 2-4 deg)
MUG_X = float(os.environ.get("FETCH_MUG_X", "0.587"))   # body centre: the outer bar's centre then stands at x = 0.501
MUG_Y = float(os.environ.get("FETCH_MUG_Y", str(0.064)))  # the row's y by default; the wander puts it metres away
MUG_YAW = 0.0                        # rad: squared to the robot. Turned 11 deg (beach6) the lower bar slid off the
                                     # finger's underside on the lift; the fixed jaw clears the cup without it, the
                                     # bar standing 86 mm from the axis
MASS = float(os.environ.get("FETCH_MUG_MASS", "0.075"))   # kg, with the handle: an empty enamel mug. 0.104 sagged 20-30 deg in the
                                     # folded drink and slipped out one time in three (auto14-48)
#: The handle is a silicone grip: the pinch holds the mug level only while its friction
#: torque beats the weight on the handle's lever (0.104 kg x 86 mm = 0.088 N*m). On the
#: row's contact (sliding 1.0 against the jaw, torsional 0.0002) it rotated to 60 deg in
#: every lift; a grip's sliding 2.0 and torsional 0.005 m (a 5 mm patch) roughly triple
#: the capacity. Priority 2 so the grip's numbers win over the jaw's.
GRIP_FRICTION, GRIP_PRIORITY = (2.0, 0.005, 0.0001), 2
#: The handle's contact time constant, 10 ms (twice the 5 ms step, the integrator's
#: floor) against the row's 20 ms. On the row's contact the pinched bar CREEPS along the
#: jaw toward its tip while the crab walks -- the gait's jolts part the compliant contact
#: a hair at a time -- at 0.5-0.9 mm/s level, 1.1 at a 0.75 walk, 6-17 mm in one 3 s
#: sidestep at nose 0.2, from 25 mm behind the mouth to off the tip in ~20 s (every
#: carry past ~25 s dropped the mug: w31/w32, probe_pinch 2026-10-13, 3 of 4 trials).
#: At 10 ms the same 19 s of walking and sidestep moved it 2 mm in both trials and the
#: hang stayed under 17 deg; a 50 g mug on the 20 ms contact crept exactly as the 75 g
#: one (probe_pinch light/stiff runs). 10 ms is the setting `objects.py` calls the edge of
#: stability rather than a margin; the handle is three small boxes against the jaw faces
#: and sixty seconds of pinch, lift, walk and sidestep showed nothing of it, so the edge
#: is taken for the one prop the film cannot drop. FETCH_MUG_SOLREF overrides it for the probes.
GRIP_SOLREF = [float(v) for v in os.environ.get("FETCH_MUG_SOLREF", "0.01,1.0").split(",")]
GRIP_RGBA = [0.16, 0.16, 0.18, 1.0]

#: What the claw aims at, in the mug's frame: the outer bar's centre, and its half
#: extents across x and y. `hold` is how far inside the anvil the bar should sit --
#: where the finger closes (2.6 cm behind the anvil, measured from the jaw clouds).
FEATURE = {"offset": (-(BODY_R + HANDLE_OUT), 0.0), "half_x": BAR, "half_y": BAR, "hold": 0.026,
           "gap": 0.025}   # the jaw lands this far beside the bar and the finger pushes the mug the rest of the
                           # way to the anvil (no tipping: 40/50 > 0.35). 0.015 was tried for the 75 g mug and
                           # the pinch landed worse: two drinks dropped and an empty close (light21-23)


def handle_boxes() -> list[tuple[list[float], list[float]]]:
    """(centre, half-size) of the loop's three bars, in the mug's frame."""
    bar_x = -(BODY_R + HANDLE_OUT)
    reach_half = (BODY_R + HANDLE_OUT - (BODY_R - BAR)) / 2      # from inside the wall to the outer bar
    reach_mid = -(BODY_R - BAR) - reach_half
    return [([reach_mid, 0.0, LOOP_HALF_Z], [reach_half, BAR, BAR]),
            ([reach_mid, 0.0, -LOOP_HALF_Z], [reach_half, BAR, BAR]),
            ([bar_x, 0.0, 0.0], [BAR, BAR, LOOP_HALF_Z + BAR])]


#: A geom group of the mug's own: `LiveViewer._collision_only_groups` turns a group off
#: when all its geoms collide and their bodies carry an appearance geom, which is how
#: the recorder draws the mesh and not the boxes. Alpha 0 as well, for any other viewer.
COLLISION_GROUP = 5
HIDDEN = [0.0, 0.0, 0.0, 0.0]


def _skin(spec: Any, body: Any) -> None:
    """The model's mesh and its artwork, over the collision shapes, colliding with nothing."""
    mesh = np.load(ASSETS / "mug_mesh.npz")
    from PIL import Image

    img = Image.open(ASSETS / "mug_tex.png").convert("RGB")
    tex = spec.add_texture(name="mug_art", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           width=img.width, height=img.height, nchannel=3)
    tex.data = np.asarray(img, dtype=np.uint8).tobytes()
    mat = spec.add_material(name="mug_art", roughness=0.6, metallic=0.0, specular=0.3)
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "mug_art"
    uv = mesh["uv"].astype(np.float64)
    uv[:, 1] = 1.0 - uv[:, 1]   # trimesh keeps OBJ's lower-left origin; MuJoCo's texcoord row 0 is the image's top
    m = spec.add_mesh(name="mug_skin")
    m.uservert = mesh["vert"].astype(np.float64).ravel().tolist()
    m.userface = mesh["face"].astype(np.int32).ravel().tolist()
    m.usertexcoord = uv.ravel().tolist()
    body.add_geom(name="mug_skin", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mug_skin",
                  material="mug_art", contype=0, conaffinity=0, group=2, mass=0.0)
    h = spec.add_mesh(name="mug_grip")
    h.uservert = mesh["hvert"].astype(np.float64).ravel().tolist()
    h.userface = mesh["hface"].astype(np.int32).ravel().tolist()
    body.add_geom(name="mug_grip", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mug_grip",
                  rgba=GRIP_RGBA, contype=0, conaffinity=0, group=2, mass=0.0)


def mug_spec() -> Any:
    spec = O._prop_spec()
    body = O._free_body(spec, "mug")
    O._prop_geom(body, name="mug_body", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                 size=[BODY_R, BODY_HALF_H, 0.0], pos=[0.0, 0.0, 0.0], rgba=HIDDEN, mass=MASS - 0.024,
                 group=COLLISION_GROUP)
    boxes = handle_boxes()
    for i, (pos, half) in enumerate(boxes):
        O._prop_geom(body, name=f"mug_handle_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=half, pos=pos, rgba=HIDDEN, mass=0.024 / len(boxes), group=COLLISION_GROUP,
                     friction=list(GRIP_FRICTION), priority=GRIP_PRIORITY, solref=list(GRIP_SOLREF))
    _skin(spec, body)
    return spec


def apply_mug(env_cfg: Any) -> None:
    """The mug in front of the robot, on the row's friction model and lighting."""
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), O.OBJECTS_NCONMAX)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), O.OBJECTS_NJMAX)
    env_cfg.sim.mujoco.cone = O.PROP_CONE
    env_cfg.sim.mujoco.impratio = O.PROP_IMPRATIO
    from math import cos, sin

    from mjlab.entity import EntityCfg

    props = {"mug": EntityCfg(spec_fn=mug_spec, init_state=EntityCfg.InitialStateCfg(
        pos=(MUG_X, MUG_Y, BODY_HALF_H), rot=(cos(MUG_YAW / 2), 0.0, 0.0, sin(MUG_YAW / 2))))}
    env_cfg.scene.entities.update(props)
    O._place_props(env_cfg, props)
    O._place_the_robot(env_cfg)
    # The look is the scene's (`--scene beach`, `studio`...): the row strips the
    # terrain's textures and lights for its own flat studio, and this does not.
