"""The shared room the cleaning race is run in: two crabs, a line down the middle,
and the objects strewn evenly over both halves.

Andrew (2026-10-15): "a race to clean a room between two crabs where they share a
room and there is a line down the middle, there are objects evenly distributed
throughout the room and the goal of the red crab is to put everything on the right
side and the blue crab tries to put everything on the left side, in the end the
winner is the crab with the most amount of objects on their target side." And the
objects "should have some variation. We can have boxes, but then some balls that
roll would be nice and maybe some other shapes as well."

The world is the battle's two crabs (`tools/battle/arena.py` builds the second one
and makes the two collide) on a flat floor instead of a plank: a painted rink,
`ROOM_X` by `ROOM_Y` metres, walled by a low kerb the crabs cannot climb and the
balls cannot roll over, with the line along y through x = 0. The camera sits on
-y (azimuth 90) looking along +y, so +x is the RIGHT of the picture: the red crab's
side is x > 0 and the blue crab's x < 0. Each crab starts at its own side's far
end on the line's axis, facing the other.

The objects are props (`tasks/jumper/five_foot/objects.py`'s contact model: free
bodies on the prop channel, so a crab's legs and shell meet them): round pucks,
square pucks and small balls, in a seeded layout that is MIRRORED through the
line so both halves start with the same objects in the same places -- a race is
only fair if it starts even. Their sizes are picked against the crab: the
shell's underside is ~0.08 m up at the stand, so a 7 cm puck meets the legs and
the shell's front edge rather than passing underneath, and nothing is taller
than 12 cm, so a cluster that runs out of the top of the depth picture is the
kerb or the other crab (see `pilots/sweeper.py` for what the measured pushes did).

Numbers are environment variables where a take may want them changed, as the
battle's are (`RACE_*`).
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import importlib.util as _ilu
import math
import os
from typing import Any

_b = _ilu.spec_from_file_location("_beside", __file__.rsplit("/", 1)[0] + "/_beside.py")
_m = _ilu.module_from_spec(_b); _b.loader.exec_module(_m); _m.install()   # sibling imports, no sys.path

from arena import (
    BLUE,
    RED,
    SHELL_BLUE,
    SHELL_RED,
    _pin,
    _recoloured,
    _second_robots_events,
    _tof_for_blue,
    cross_collide,
)

ROOM_X = float(os.environ.get("RACE_ROOM_X", "4.0"))      # m across the line (the picture's width)
ROOM_Y = float(os.environ.get("RACE_ROOM_Y", "2.6"))      # m along the line (toward and away from the camera)
KERB_H, KERB_T = 0.35, 0.05                               # the wall round the rink: taller than a crab (0.20) and every
                                                          # object (0.12), so the depth image cannot mistake it for either
                                                          # (at 10 cm it was a two-zone sliver at range, told from a cube
                                                          # by width alone, and the crabs "lost the walls", Andrew 2026-10-15)
GLASS = 0.22                                              # the camera-side wall's alpha: at 0.35 m it would hide 0.4 m of floor
CHAMFER = float(os.environ.get("RACE_CHAMFER", "0.45"))   # each corner cut off by a wall at 45 degrees this long (along each
                                                          # side): a thing in a square corner has two free faces and both
                                                          # shove it into a wall -- no crab wider than the gap can get behind
                                                          # it -- on the diagonal it is pushed along the diagonal and out
                                                          # (Andrew, 2026-10-15: "the crabs fail when an object is in a corner")
LINE_W = 0.05                                             # the painted line's width
START_X = float(os.environ.get("RACE_START_X", str(ROOM_X / 2 - 0.45)))   # each crab's mark, out from the line
N_PER_SIDE = int(os.environ.get("RACE_OBJECTS", "6"))     # objects on each half at the start
OBJECT_MARGIN = 0.40                                      # m: objects keep this far from the kerb...
OBJECT_GAP = 0.45                                         # ...and from each other
START_CLEAR = 0.55                                        # ...and from a crab's mark (its stance is ~0.45 m across)
LINE_CLEAR = 0.25                                         # ...and from the line (an object ON the line is nobody's)

#: What is on the floor: name -> (geom type, size, mass kg, how it is placed). `half_z`
#: is the resting height of the body origin. A puck's weight sits on a face; a ball rolls.
#: All of them LOW, 12 cm at most (cubes of 15, cans of 17 and balls of 17 cm before, 2026-10-15):
#: the sensor's picture reaches 0.137 + 0.384 r m up the room at range r, so nothing here
#: runs out of its top at any range, and a cluster that does is the kerb or the other crab
#: whatever else it looks like (Andrew: "an object can never fully take up the crab's view
#: and in this case we know if something does obscure the full view it must be a wall").
#: `sight.OBJECT_HEIGHT` is the line.
KINDS = {
    "puck": {"type": "cylinder", "size": (0.095, 0.035, 0.0), "mass": 0.2, "half_z": 0.035},
    "square": {"type": "box", "size": (0.08, 0.08, 0.035), "mass": 0.25, "half_z": 0.035},
    "ball": {"type": "sphere", "size": (0.06, 0.0, 0.0), "mass": 0.08, "half_z": 0.06},
}
#: Which kinds, in what order, on each half; the layout cycles through it.
MIX = os.environ.get("RACE_MIX", "puck,square,ball,square,puck,ball").split(",")
#: Saturated, and none of them the teams' red or blue, so an object never reads as a side.
PALETTE = {
    "puck": ((0.1, 0.1, 0.1, 1.0), (1.0, 0.95, 0.6, 1.0), (0.1, 0.85, 0.9, 1.0)),
    "square": ((1.0, 0.85, 0.0, 1.0), (0.1, 0.8, 0.3, 1.0), (0.95, 0.3, 0.75, 1.0)),
    "ball": ((1.0, 1.0, 1.0, 1.0), (1.0, 0.5, 0.0, 1.0)),
}
#: A ball is light and rolls: the prop model's friction (0.35 sliding) with the rolling
#: term kept, so a shoved ball carries on across the floor; everything else slides on
#: the prop model as the fetch's props do.
BALL_FRICTION = (0.35, 0.0002, 0.0001)

#: Bold (Andrew: "more bold colors and a nicer presentation of the play area and its
#: environment"): each half in its team's colour, a white line and marks, a white
#: kerb, and round the rink a dark floor, three walls (none on the camera's side),
#: a banner in each team's colour on its own end wall, and corner posts with flags.
FLOOR = (0.92, 0.92, 0.90)
TINT = {RED: (0.88, 0.16, 0.12), BLUE: (0.12, 0.35, 0.92)}
TINT_MIX = 0.85                                           # how much of the tint over the floor
KERB = (0.96, 0.96, 0.94, 1.0)
LINE = (1.0, 1.0, 1.0)
SET_FLOOR = (0.22, 0.22, 0.24, 1.0)
WALL = (0.80, 0.78, 0.74, 1.0)
WALL_H, WALL_OUT = 1.4, 0.9                               # the set's walls: tall enough to keep the horizon out of frame
SUN_DIR = (0.12, 0.08, -1.0)                              # the daylight scene's sun turned overhead: a 0.35 m kerb lit from
                                                          # the side lays a shadow across the rink (Andrew: "a lighting
                                                          # adjustment so there aren't shadows in the arena")
POST = (0.95, 0.95, 0.95, 1.0)


def layout(seed: int, n: int = N_PER_SIDE) -> list[dict]:
    """The objects, mirrored through the line: `n` on the blue half (x < 0) and the
    same `n` reflected onto the red half. Each is {name, kind, x, y, rgba}.
    `RACE_LAYOUT="puck:-1.8:0.3;ball:0.5:-0.2"` stages exactly these instead (a probe)."""
    import numpy as np

    staged = os.environ.get("RACE_LAYOUT")
    if staged:
        out = []
        for i, item in enumerate(staged.split(";")):
            kind, x, y = item.split(":")
            out.append({"name": f"obj_{kind}_{i}", "kind": kind, "x": float(x), "y": float(y),
                        "rgba": PALETTE[kind][i % len(PALETTE[kind])]})
        return out
    rng = np.random.default_rng(seed)
    placed: list[tuple[float, float]] = []
    out = []
    lo_x, hi_x = LINE_CLEAR, ROOM_X / 2 - OBJECT_MARGIN
    hi_y = ROOM_Y / 2 - OBJECT_MARGIN
    for i in range(n):
        kind = MIX[i % len(MIX)]
        for _ in range(2000):
            x, y = float(rng.uniform(lo_x, hi_x)), float(rng.uniform(-hi_y, hi_y))
            if math.hypot(x - START_X, y) < START_CLEAR or not free(x, y, OBJECT_MARGIN * 0.6):
                continue
            if all(math.hypot(x - px, y - py) >= OBJECT_GAP for px, py in placed):
                break
        else:
            raise ValueError(f"no room for object {i} of {n}: smaller RACE_OBJECTS or a bigger room")
        placed.append((x, y))
        rgba = PALETTE[kind][i % len(PALETTE[kind])]
        out.append({"name": f"obj_{kind}_{i}_blue", "kind": kind, "x": -x, "y": y, "rgba": rgba})
        out.append({"name": f"obj_{kind}_{i}_red", "kind": kind, "x": x, "y": -y, "rgba": rgba})
    return out


def _object_spec(kind: str, rgba: tuple) -> Any:
    import mujoco

    from tasks.jumper.five_foot import objects as O

    k = KINDS[kind]
    spec = O._prop_spec()
    body = O._free_body(spec, kind)
    gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
             "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[k["type"]]
    extra = {"friction": list(BALL_FRICTION)} if kind == "ball" else {}
    O._prop_geom(body, name=f"{kind}_geom", type=gtype, size=list(k["size"]), pos=[0.0, 0.0, 0.0],
                 rgba=list(rgba), mass=k["mass"], **extra)
    return spec


def free(x: float, y: float, margin: float = 0.0) -> bool:
    """Whether a point is on the rink floor `margin` clear of every wall, chamfers included."""
    hx, hy = ROOM_X / 2 - margin, ROOM_Y / 2 - margin
    if abs(x) > hx or abs(y) > hy:
        return False
    return (ROOM_X / 2 - abs(x)) + (ROOM_Y / 2 - abs(y)) > CHAMFER + margin * math.sqrt(2)


def side_of(x: float) -> str | None:
    """Whose half an x is on: the red crab's (x > 0), the blue crab's, or None on the line."""
    if x > LINE_W / 2:
        return RED
    if x < -LINE_W / 2:
        return BLUE
    return None


def _floor_texture(spec: Any) -> None:
    """The rink painted onto a 1 mm sheet over the ground: the two halves faintly
    tinted, the line down the middle, a mark where each crab starts. A 2D texture on
    a box is mapped per face, so the top face carries the whole picture."""
    import mujoco
    import numpy as np

    n = 1024
    ny = int(n * ROOM_Y / ROOM_X)
    img = np.zeros((ny, n, 3))
    xs = (np.arange(n) + 0.5) / n * ROOM_X - ROOM_X / 2
    ys = (np.arange(ny) + 0.5) / ny * ROOM_Y - ROOM_Y / 2
    X, Y = np.meshgrid(xs, ys)
    base = np.array(FLOOR)
    img[:] = base
    for crab, tint in TINT.items():
        mask = (X > 0) if crab == RED else (X < 0)
        img[mask] = (1 - TINT_MIX) * base + TINT_MIX * np.array(tint)
    rng = np.random.default_rng(5)
    img *= (1.0 + 0.03 * rng.standard_normal((ny, n, 1)))                # a little grain, so the floor is not a flat card
    img[np.abs(X) < LINE_W / 2] = LINE
    # the rink is the octagon inside the kerb: the touchline follows it, and the floor the
    # chamfers cut off is the set's (it was the rink's, a tinted pocket behind each chamfer)
    edge = np.minimum(np.minimum(ROOM_X / 2 - np.abs(X), ROOM_Y / 2 - np.abs(Y)),
                      ((ROOM_X / 2 - np.abs(X)) + (ROOM_Y / 2 - np.abs(Y)) - CHAMFER) / math.sqrt(2))
    img[edge < 0.04] = LINE                                                # a touchline round the rink
    img[edge < 0.0] = SET_FLOOR[:3]
    for sx in (-START_X, START_X):
        ring = np.abs(np.hypot(X - sx, Y) - 0.30) < 0.025
        img[ring] = LINE
    tex = spec.add_texture(name="rink", type=mujoco.mjtTexture.mjTEXTURE_2D, width=n, height=ny, nchannel=3)
    tex.data = (np.clip(img, 0.0, 1.0)[::-1] * 255).astype(np.uint8).tobytes()   # row 0 is the image's top: +y
    mat = spec.add_material(name="rink", roughness=0.9, metallic=0.0, specular=0.1)
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "rink"
    spec.worldbody.add_geom(name="rink", type=mujoco.mjtGeom.mjGEOM_BOX, size=[ROOM_X / 2, ROOM_Y / 2, 0.0005],
                            pos=[0.0, 0.0, 0.0005], material="rink", contype=0, conaffinity=0, group=2, mass=0.0)


def decorate(spec: Any) -> None:
    """The scene's last edit: the crabs made to collide, the rink and its kerb."""
    import mujoco

    cross_collide(spec)
    _floor_texture(spec)
    box = mujoco.mjtGeom.mjGEOM_BOX
    hx, hy, hz, c = ROOM_X / 2, ROOM_Y / 2, KERB_H / 2, CHAMFER
    # The kerb is an octagon: eight walls of one thickness, each box's inner face on its
    # edge of the rink and each long enough to reach the outer corner of its mitre with the
    # next (KERB_T tan 22.5 degrees past the inner corner, buried in the neighbour's box),
    # so the corners close with no pocket and no gap. It was a rectangle with a wall laid
    # across each corner: a tinted pocket of rink behind every chamfer, and the chamfer a
    # touch short of the kerb at both ends, where a puck was once pushed through
    # (probe_fix10_chamfer). Andrew, 2026-10-16: "drop the unused corner pockets (it looks
    # cheap)". The mitred outer faces meet at an angle, so nothing flickers (two faces in
    # one place did, still4).
    corners = [(hx - c, -hy), (hx, -hy + c), (hx, hy - c), (hx - c, hy),
               (-hx + c, hy), (-hx, hy - c), (-hx, -hy + c), (-hx + c, -hy)]       # counter-clockwise
    ext = KERB_T * math.tan(math.radians(22.5))
    for i in range(8):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 8]
        ang = math.atan2(y1 - y0, x1 - x0)
        nx, ny = math.sin(ang), -math.cos(ang)                 # outward: to the right of the way round
        cx, cy = (x0 + x1) / 2 + nx * KERB_T / 2, (y0 + y1) / 2 + ny * KERB_T / 2
        quat = [math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)]
        glass = (y0 + y1) / 2 < -hy + c * 0.75                 # the camera's side and its two chamfers
        rgba = list(KERB[:3]) + [GLASS if glass else 1.0]
        spec.worldbody.add_geom(name=f"kerb{i}", type=box, size=[math.hypot(x1 - x0, y1 - y0) / 2 + ext, KERB_T / 2, hz],
                                pos=[cx, cy, hz], quat=quat, rgba=rgba, contype=1, conaffinity=1, group=2, mass=0.0)
    _set(spec)
    for light in spec.lights:
        if light.castshadow:
            light.dir = list(SUN_DIR)


def _set(spec: Any) -> None:
    """The set round the rink, drawn only: a dark floor out to the walls, three walls
    (the camera's side open), a banner in each team's colour on its end wall, four
    corner posts with pennants. Nothing here collides, and nothing stands inside
    the sensor's reach of the rink but the walls, which are taller than the kerb
    and never read as it (`sight.KERB_TOP`)."""
    import mujoco

    box = mujoco.mjtGeom.mjGEOM_BOX
    ox, oy = ROOM_X / 2 + KERB_T + WALL_OUT, ROOM_Y / 2 + KERB_T + WALL_OUT
    off = {"contype": 0, "conaffinity": 0, "group": 2, "mass": 0.0}
    spec.worldbody.add_geom(name="set_floor", type=box, size=[ox + 3.0, oy + 3.0, 0.0004], pos=[0.0, 0.0, 0.0004],
                            rgba=list(SET_FLOOR), **off)
    t = 0.05
    walls = (("wall_back", (0.0, oy + t, WALL_H / 2), (ox + 2 * t, t, WALL_H / 2)),
             ("wall_right", (ox + t, 0.0, WALL_H * 0.35), (t, oy, WALL_H * 0.35)),
             ("wall_left", (-(ox + t), 0.0, WALL_H * 0.35), (t, oy, WALL_H * 0.35)))
    for name, pos, size in walls:
        spec.worldbody.add_geom(name=name, type=box, size=list(size), pos=list(pos), rgba=list(WALL), **off)
    # the banners: each team's colour along its half of the back wall, a white stripe
    # through them (on the side walls they filled the frame's edges, still2)
    for crab, sx in ((RED, 1.0), (BLUE, -1.0)):
        c = TINT[crab]
        spec.worldbody.add_geom(name=f"banner_{crab}", type=box, size=[ox * 0.46, 0.012, 0.16],
                                pos=[sx * ox * 0.5, oy + t - 0.06, 0.62], rgba=[*c, 1.0], **off)
        spec.worldbody.add_geom(name=f"banner_{crab}_stripe", type=box, size=[ox * 0.46, 0.013, 0.022],
                                pos=[sx * ox * 0.5, oy + t - 0.06, 0.62], rgba=[1.0, 1.0, 1.0, 1.0], **off)
    # a dark band low round the walls, and a post at each of the set's corners
    spec.worldbody.add_geom(name="wall_back_band", type=box, size=[ox, 0.012, 0.08], pos=[0.0, oy + t - 0.06, 0.08],
                            rgba=list(SET_FLOOR), **off)
    for i, (px, py) in enumerate(((-ox, -oy), (ox, -oy), (ox, oy), (-ox, oy))):
        spec.worldbody.add_geom(name=f"post{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.45, 0.0],
                                pos=[px, py, 0.45], rgba=list(POST), **off)


def apply_race(env_cfg: Any, *, seed: int, senses: bool = False, n_objects: int = N_PER_SIDE) -> list[dict]:
    """The second crab, the rink, the objects and a camera on all of it. Returns
    the layout (`layout(seed)`), whose names are the scene's entity names."""
    from mjlab.entity import EntityCfg
    from mjlab.viewer.viewer_config import ViewerConfig

    from scenes.registry import _install_spec_edits
    from tasks.jumper.common.constants import STAND_Z
    from tasks.jumper.five_foot import objects as O

    entities = env_cfg.scene.entities
    red = entities[RED]
    blue = copy.deepcopy(red)
    blue.spec_fn = _recoloured(blue.spec_fn, SHELL_RED, SHELL_BLUE)
    # red at +x facing -x (its own half behind it), blue the mirror
    red.init_state = dataclasses.replace(red.init_state, pos=(START_X, 0.0, STAND_Z), rot=(0.0, 0.0, 0.0, 1.0))
    blue.init_state = dataclasses.replace(blue.init_state, pos=(-START_X, 0.0, STAND_Z), rot=(1.0, 0.0, 0.0, 0.0))
    entities[BLUE] = blue
    cloned = _second_robots_events(env_cfg)
    if "reset_base" not in env_cfg.events:
        raise ValueError("the race pins the spawns, and this config has no 'reset_base'")
    _pin(env_cfg.events["reset_base"])
    _pin(env_cfg.events["reset_base2"])
    moved = []
    for name, term in env_cfg.terminations.items():
        if "limit_angle" in (term.params or {}):
            term.params = {**term.params, "limit_angle": math.pi}
            moved.append(name)
    tof = _tof_for_blue(env_cfg, senses)

    objs = layout(seed, n_objects)
    props = {}
    for o in objs:
        k = KINDS[o["kind"]]
        props[o["name"]] = EntityCfg(spec_fn=functools.partial(_object_spec, o["kind"], o["rgba"]),
                                     init_state=EntityCfg.InitialStateCfg(pos=(o["x"], o["y"], k["half_z"] + 0.002)))
    entities.update(props)
    O._place_props(env_cfg, props)
    env_cfg.sim.mujoco.cone = O.PROP_CONE
    env_cfg.sim.mujoco.impratio = O.PROP_IMPRATIO
    _install_spec_edits(env_cfg.scene, None, decorate)
    # Two hexapods, a dozen props and four kerbs: more contacts than one robot on a plane.
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 2048)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 8192)

    viewer = env_cfg.viewer
    viewer.origin_type = ViewerConfig.OriginType.WORLD
    viewer.lookat = (0.0, float(os.environ.get("RACE_CAM_Y", "0.2")), 0.05)    # the whole rink in the lower two thirds
    viewer.distance = float(os.environ.get("RACE_CAM_DIST", "4.9"))           # of the frame, under the readouts (still2)
    viewer.azimuth = float(os.environ.get("RACE_CAM_AZ", "90"))
    viewer.elevation = float(os.environ.get("RACE_CAM_EL", "-40"))
    print(f"[room] rink {ROOM_X:g} x {ROOM_Y:g} m, the line along y at x = 0; red at x {START_X:+.2f} facing -x "
          f"(its side x > 0), blue at {-START_X:+.2f} facing +x; {len(objs)} objects, {n_objects} a side, seed {seed}")
    print(f"[room] cloned for the blue crab: {', '.join(cloned) or 'nothing'}"
          + ("; its dToF is `tof2`" if tof else "; no depth sensors (no pilot looks)")
          + (f"; fall resets left to the referee: {', '.join(moved)}" if moved else ""))
    return objs
