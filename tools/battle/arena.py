"""Two crabs on a plank: the world the duel is fought in.

The toolkit runs one robot per world -- one `robot` entity, one controller, one
pad. This adds the second: a copy of the task's own robot entity under the name
`robot2`, its shell recoloured blue, standing at the far end of a plank facing
the red one, with every event the task gives the red robot cloned for it so a
reset puts both back on their marks. Nothing here reads the task's rewards or
observations, which stay the red robot's: the duel is played through two
`AppPlayer`s (`duel.py`), and the environment is only the physics.

Three things the second robot needs that are invisible in the config:

* **The crabs could not touch.** The robot's collision masks (`constants.py`)
  give each leg a bit and the base none, so two copies of the robot collide
  leg-against-leg only where the leg indices differ, and base against anything
  not at all: a shove would pass through the other crab's shell. `_decorate`
  ORs one bit per crab into every collision geom's `contype` and the *other*
  crab's bit into its `conaffinity`, so every geom of one meets every geom of
  the other and nothing within a crab changes.
* **The world's fall reset would hide the fall.** The family's `fell_over`
  termination teleports a robot upright the step it tips past the limit, and
  `play --app` moves that limit to 80 degrees. A crab shoved off a 0.25 m plank
  tips past that as it lands, so the camera would see the loser vanish mid-air.
  The terminations are moved out of reach and the referee (`referee.py`) decides
  who fell, lets the fall play out, and resets the round itself.
* **An entity with no reset event stays where it was.** mjlab places entities
  in `reset_root_state_uniform`, so `robot2` needs the task's `reset_base` and
  joint reset cloned under its own name (`_second_robots_events`); the clone
  carries the task's own parameters, so the blue crab settles exactly as the
  red one does.

The plank is a static box on the world body -- not a prop entity, since it
never moves -- on two trestles, over whatever ground the scene puts underneath
(`--scene beach` for the sand). Its numbers are environment variables so a take
can be framed without editing this file, the way `tools/fetch/mug.py` places
the mug. Round the disc goes the dressing (`dress`): everything in it is drawn
and nothing in it collides, so the fight on the dressed table is the fight on
the bare one.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import importlib.util
import inspect
import itertools
import math
import os
from pathlib import Path
from typing import Any

#: The scene's names for the two crabs. The red one is the task's own robot, so
#: everything the task says about `robot` stays true of it.
RED, BLUE = "robot", "robot2"
#: jumper.xml's shell colour (every `rgba="0.85 0.08 0.05 1"` geom is the red
#: shell), and the blue it becomes on the second crab.
SHELL_RED = (0.85, 0.08, 0.05, 1.0)
SHELL_BLUE = (0.10, 0.32, 0.88, 1.0)

ARENA = os.environ.get("BATTLE_ARENA", "disc")                      # disc (Andrew, 2026-10-14: "I prefer that the
                                                                    # platform is circular so the crabs have more
                                                                    # ability to dodge to the side") or plank
DISC_R = float(os.environ.get("BATTLE_DISC_R", "0.8"))              # m: the disc's radius; a crab's stance is ~0.4 m
PLANK_LEN = float(os.environ.get("BATTLE_PLANK_LEN", "2.0"))        # m, along x
PLANK_WIDTH = float(os.environ.get("BATTLE_PLANK_WIDTH", "0.7"))    # m: a crab is ~0.4 m across its stance; at 0.5 the first shove settled it
PLANK_TOP = float(os.environ.get("BATTLE_PLANK_TOP", "0.25"))       # m above the ground: the drop
PLANK_THICK = 0.04
START_GAP = float(os.environ.get("BATTLE_START_GAP", "0.9"))        # m between the two bases at the bell
WOOD = (0.62, 0.45, 0.26, 1.0)
TRESTLE = (0.35, 0.30, 0.25, 1.0)

#: The dressing (Andrew, 2026-10-14: "dress it up to make the space and platform
#: look more interesting"): a decked top with a painted ring and the two marks, a
#: brass lip and a dark skirt round the rim, a stone plinth under the pedestal,
#: four tiki torches round the table each with a warm light of its own, a line of
#: red and blue pennants behind it, shells and a starfish on the sand, and the
#: beach's own dunes, palms and rocks (`tools/fetch/scenery.py`). The camera sits
#: on -y (azimuth 90), so behind is +y and the torches flank the shot.
STONE = (0.58, 0.50, 0.40, 1.0)
BRASS = (0.85, 0.66, 0.24, 1.0)
SKIRT = (0.13, 0.10, 0.08, 1.0)
PAINT = (0.93, 0.90, 0.84, 1.0)
BAMBOO = (0.74, 0.60, 0.34, 1.0)
BINDING = (0.28, 0.18, 0.10, 1.0)
FLAME = ((1.0, 0.92, 0.45, 1.0), (1.0, 0.52, 0.10, 0.55))
SHELL = ((0.94, 0.90, 0.80, 1.0), (0.93, 0.76, 0.70, 1.0), (0.80, 0.72, 0.60, 1.0))
STARFISH = (0.88, 0.46, 0.24, 1.0)
TORCH_R, TORCH_AZ = 1.3, (25.0, 155.0, 205.0, 335.0)       # m out from the centre, degrees round from +x
PENNANT_POSTS = ((-1.6, 1.4), (1.6, 1.4))                   # behind the table, as the camera sees it
PENNANTS = 15

#: Each crab's depth sensor, as the scene names it (`_tof_for_blue`).
TOF = {RED: "tof", BLUE: "tof2"}

#: One contact bit per crab, above the ground bit (1), the six leg bits (2..7)
#: and the prop channel (`objects.py::PROP_BIT`, 1 << 11).
CROSS_BITS = {RED: 1 << 13, BLUE: 1 << 14}


def _recoloured(spec_fn: Any, old: tuple, new: tuple) -> Any:
    """`spec_fn` with every geom of colour `old` turned `new`."""
    def build():
        spec = spec_fn()
        for geom in spec.geoms:
            if all(abs(float(a) - b) < 0.02 for a, b in zip(geom.rgba, old)):
                geom.rgba = list(new)
        return spec
    return build


def _pin(reset: Any) -> None:
    """Narrow a `reset_base` to its mark: the entity's own `init_state` is the
    place, and the event's x, y and yaw ranges are what would move it off."""
    pose = dict(reset.params.get("pose_range", {}))
    pose.update(x=(0.0, 0.0), y=(0.0, 0.0), yaw=(0.0, 0.0))
    reset.params["pose_range"] = pose


def _second_robots_events(env_cfg: Any) -> list[str]:
    """Clone every startup and reset event the task gives `robot` for `robot2`.

    An event names its robot in `asset_cfg`, or leaves it to the function's
    default, which is `robot`; one whose function takes no `asset_cfg` cannot be
    pointed at the second robot and is skipped, by name, on the terminal.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    cloned, skipped = [], []
    for name, term in list(env_cfg.events.items()):
        if term.mode not in ("startup", "reset"):
            continue
        asset = (term.params or {}).get("asset_cfg")
        if asset is None:
            if "asset_cfg" not in inspect.signature(term.func).parameters:
                skipped.append(name)
                continue
            asset2 = SceneEntityCfg(BLUE)
        elif asset.name == RED:
            asset2 = dataclasses.replace(asset, name=BLUE)
        else:
            continue
        clone = copy.deepcopy(term)
        clone.params = {**(clone.params or {}), "asset_cfg": asset2}
        env_cfg.events[f"{name}2"] = clone
        cloned.append(name)
    if skipped:
        print(f"[arena] not cloned for the blue crab (no asset_cfg): {', '.join(skipped)}")
    return cloned


def _tof_for_blue(env_cfg: Any, senses: bool) -> bool:
    """The blue crab's dToF beside the red one's, when the task mounts one and a
    pilot looks through it; otherwise neither crab's, since 2 x 2268 rays a step
    are cast for nobody."""
    sensors = list(env_cfg.scene.sensors or ())
    tof = next((s for s in sensors if type(s).__name__ == "ToFSensorCfg" and s.name == "tof"), None)
    if tof is None:
        return False
    if not senses:
        env_cfg.scene.sensors = tuple(s for s in sensors if s is not tof)
        return False
    frame = dataclasses.replace(tof.frame, entity=BLUE)
    pattern = dataclasses.replace(tof.pattern, camera=dataclasses.replace(tof.pattern.camera, entity=BLUE),
                                  body=frame)
    sensors.append(dataclasses.replace(tof, name="tof2", frame=frame, pattern=pattern))
    env_cfg.scene.sensors = tuple(sensors)
    return True


def cross_collide(spec: Any) -> None:
    """Every collision geom of one crab meets every geom of the other (the module
    docstring's first point); nothing within a crab changes. Shared with the race
    (`tools/race/room.py`), which builds a different world round the same two crabs."""
    for geom in spec.geoms:
        name = geom.name or ""
        for crab, bit in CROSS_BITS.items():
            if name.startswith(f"{crab}/") and (geom.contype or geom.conaffinity):
                other = next(b for c, b in CROSS_BITS.items() if c != crab)
                geom.contype |= bit
                geom.conaffinity |= other


def _decorate(spec: Any, dressed: bool = True) -> None:
    """The scene's last edit: the crabs made to collide, the plank, and the dressing."""
    import mujoco

    cross_collide(spec)
    box = mujoco.mjtGeom.mjGEOM_BOX
    leg_h = (PLANK_TOP - PLANK_THICK) / 2
    if ARENA == "disc":
        # a round table: the top a cylinder, on one pedestal
        cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
        top = spec.worldbody.add_geom(name="plank", type=cyl, size=[DISC_R, PLANK_THICK / 2, 0.0],
                                      pos=[0.0, 0.0, PLANK_TOP - PLANK_THICK / 2], rgba=list(WOOD),
                                      contype=1, conaffinity=1, group=2)
        spec.worldbody.add_geom(name="trestle0", type=cyl, size=[0.18, leg_h, 0.0],
                                pos=[0.0, 0.0, leg_h], rgba=list(STONE if dressed else TRESTLE),
                                contype=1, conaffinity=1, group=2)
        if dressed:
            dress(spec, top)
        return
    spec.worldbody.add_geom(name="plank", type=box, size=[PLANK_LEN / 2, PLANK_WIDTH / 2, PLANK_THICK / 2],
                            pos=[0.0, 0.0, PLANK_TOP - PLANK_THICK / 2], rgba=list(WOOD),
                            contype=1, conaffinity=1, group=2)
    for i, x in enumerate((-(PLANK_LEN / 2 - 0.25), PLANK_LEN / 2 - 0.25)):
        spec.worldbody.add_geom(name=f"trestle{i}", type=box, size=[0.04, PLANK_WIDTH / 2 - 0.06, leg_h],
                                pos=[x, 0.0, leg_h], rgba=list(TRESTLE), contype=1, conaffinity=1, group=2)


def _deck(spec: Any) -> None:
    """Decking over the table's top: planks across it, each its own shade, with
    grain along them and a dark seam between, drawn into a 2D texture and laid on
    a 1 mm disc mesh with planar texture coordinates. (MuJoCo maps a texture onto
    a cylinder's cap radially: on the cylinder itself the planks came out as a
    pinwheel.) The feet stand on the cylinder under it; this collides with nothing."""
    import mujoco
    import numpy as np

    n, planks = 512, 9
    rng = np.random.default_rng(3)
    v = np.arange(n)[:, None] / n
    u = np.arange(n)[None, :] / n
    band = np.floor(v * planks)
    shade = rng.uniform(0.82, 1.12, planks)[band.astype(int)]
    grain = 0.05 * np.sin(u * 2 * math.pi * 23 + band * 1.7 + 0.8 * np.sin(v * 2 * math.pi * 11)) \
        + 0.03 * rng.standard_normal((n, n))
    seam = (np.abs(v * planks - np.round(v * planks)) < 0.012) & (band > 0) & (band < planks)
    base = np.array(WOOD[:3])
    img = base[None, None, :] * (shade + grain)[:, :, None]
    img[np.broadcast_to(seam, (n, n))] *= 0.45
    tex = spec.add_texture(name="deck", type=mujoco.mjtTexture.mjTEXTURE_2D, width=n, height=n, nchannel=3)
    tex.data = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8).tobytes()
    mat = spec.add_material(name="deck", roughness=0.8, metallic=0.0, specular=0.2)
    mat.textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "deck"
    segments, half = 72, 0.0005
    a = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([DISC_R * np.cos(a), DISC_R * np.sin(a)], axis=1)
    verts = np.concatenate([[[0.0, 0.0, half]], np.column_stack([ring, np.full(segments, half)]),
                            [[0.0, 0.0, -half]], np.column_stack([ring, np.full(segments, -half)])])
    uv = np.column_stack([verts[:, 0] / (2 * DISC_R) + 0.5, 0.5 - verts[:, 1] / (2 * DISC_R)])
    top, bottom = 1 + np.arange(segments), segments + 2 + np.arange(segments)
    nxt = (np.arange(segments) + 1) % segments
    faces = np.concatenate([np.column_stack([np.zeros(segments, int), top, top[nxt]]),
                            np.column_stack([np.full(segments, segments + 1), bottom[nxt], bottom]),
                            np.column_stack([top, bottom, bottom[nxt]]),
                            np.column_stack([top, bottom[nxt], top[nxt]])])
    mesh = spec.add_mesh(name="deck")
    mesh.uservert = verts.ravel().tolist()
    mesh.userface = faces.astype(np.int32).ravel().tolist()
    mesh.usertexcoord = uv.ravel().tolist()
    _dressing(spec, name="deck", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="deck", material="deck",
              pos=[0.0, 0.0, PLANK_TOP + half])


def _dressing(spec: Any, **kw: Any) -> Any:
    return spec.worldbody.add_geom(contype=0, conaffinity=0, mass=0.0, group=2, **kw)


def _quat_z(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _torch(spec: Any, i: int, x: float, y: float) -> None:
    import mujoco

    cap, cyl, ell = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ELLIPSOID
    _dressing(spec, name=f"torch{i}_pole", type=cap, size=[0.022, 0.4, 0.0], pos=[x, y, 0.4], rgba=list(BAMBOO))
    for k, z in enumerate((0.25, 0.45, 0.65)):
        _dressing(spec, name=f"torch{i}_band{k}", type=cyl, size=[0.027, 0.01, 0.0], pos=[x, y, z], rgba=list(BINDING))
    _dressing(spec, name=f"torch{i}_cup", type=cyl, size=[0.065, 0.055, 0.0], pos=[x, y, 0.85], rgba=list(BINDING))
    _dressing(spec, name=f"torch{i}_flame", type=ell, size=[0.03, 0.03, 0.07], pos=[x, y, 0.96], rgba=list(FLAME[0]))
    _dressing(spec, name=f"torch{i}_glow", type=ell, size=[0.055, 0.055, 0.11], pos=[x, y, 0.97], rgba=list(FLAME[1]))
    # a warm glow that reaches the table and little further (a third of it at 1.3 m, a tenth at 3)
    light = spec.worldbody.add_light(name=f"torch{i}", pos=[x, y, 1.05], dir=[0.0, 0.0, -1.0], castshadow=False)
    if hasattr(light, "type"):
        light.type = mujoco.mjtLightType.mjLIGHT_POINT
    light.diffuse = [0.55, 0.32, 0.12]
    light.specular = [0.2, 0.12, 0.05]
    light.attenuation = [1.0, 0.5, 1.0]


def _pennants(spec: Any) -> None:
    import mujoco

    cap, box = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX
    (x0, y0), (x1, y1) = PENNANT_POSTS
    for i, (x, y) in enumerate(PENNANT_POSTS):
        _dressing(spec, name=f"post{i}", type=cap, size=[0.035, 0.68, 0.0], pos=[x, y, 0.68], rgba=list(BINDING))
    height, sag = 1.35, 0.22
    pts = [(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, height - sag * (1 - (2 * f - 1) ** 2)) for f in
           (k / 16 for k in range(17))]
    for i, (a, b) in enumerate(itertools.pairwise(pts)):
        mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]
        d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
        length = math.sqrt(sum(c * c for c in d))
        # a capsule stands along its z; tilt it to the segment
        axis = [-d[1], d[0], 0.0]
        angle = math.acos(max(-1.0, min(1.0, d[2] / length)))
        n = math.hypot(axis[0], axis[1]) or 1.0
        q = [math.cos(angle / 2), math.sin(angle / 2) * axis[0] / n, math.sin(angle / 2) * axis[1] / n, 0.0]
        _dressing(spec, name=f"line{i}", type=cap, size=[0.006, length / 2, 0.0], pos=mid, quat=q, rgba=list(SKIRT))
    for i in range(PENNANTS):
        f = (i + 0.5) / PENNANTS
        x, y, z = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, height - sag * (1 - (2 * f - 1) ** 2)
        colour = SHELL_RED if i % 2 == 0 else SHELL_BLUE
        _dressing(spec, name=f"pennant{i}", type=box, size=[0.055, 0.003, 0.075], pos=[x, y, z - 0.08],
                  rgba=list(colour))


def _shells(spec: Any) -> None:
    import mujoco
    import numpy as np

    ell, cap = mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CAPSULE
    rng = np.random.default_rng(11)
    for i in range(12):
        r, a = rng.uniform(0.95, 1.6), rng.uniform(-150.0, -30.0) if i < 7 else rng.uniform(0.0, 360.0)
        x, y = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        if min(math.hypot(x - TORCH_R * math.cos(math.radians(t)), y - TORCH_R * math.sin(math.radians(t)))
               for t in TORCH_AZ) < 0.15:
            continue
        _dressing(spec, name=f"shell{i}", type=ell, size=[0.038, 0.03, 0.014], pos=[x, y, 0.012],
                  quat=_quat_z(float(rng.uniform(0.0, 180.0))), rgba=list(SHELL[i % 3]))
    for i, (x, y, yaw) in enumerate(((0.75, -1.15, 20.0), (-1.2, 0.3, 70.0))):
        for k in range(5):
            a = math.radians(yaw + 72 * k)
            _dressing(spec, name=f"starfish{i}_{k}", type=cap, size=[0.012, 0.045, 0.0],
                      pos=[x + 0.05 * math.cos(a), y + 0.05 * math.sin(a), 0.012],
                      quat=[math.cos(math.pi / 4) * math.cos(a / 2), -math.sin(math.pi / 4) * math.sin(a / 2),
                            math.sin(math.pi / 4) * math.cos(a / 2), math.cos(math.pi / 4) * math.sin(a / 2)],
                      rgba=list(STARFISH))


def dress(spec: Any, top: Any) -> None:
    """The dressing round the disc; `top` is the table's top, which gets the decking."""
    import mujoco

    cyl, box = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
    _deck(spec)
    # the rim: a brass lip at the top edge over a dark skirt
    _dressing(spec, name="lip", type=cyl, size=[DISC_R + 0.006, 0.006, 0.0], pos=[0.0, 0.0, PLANK_TOP - 0.006],
              rgba=list(BRASS))
    _dressing(spec, name="skirt", type=cyl, size=[DISC_R + 0.012, (PLANK_THICK - 0.012) / 2, 0.0],
              pos=[0.0, 0.0, PLANK_TOP - 0.012 - (PLANK_THICK - 0.012) / 2], rgba=list(SKIRT))
    # the paint: a ring at the dodges' radius, the centre spot, and a mark the colour of each crab
    z = PLANK_TOP + 0.0016
    for i in range(48):
        a = 360.0 * i / 48
        r = DISC_R * 0.68
        _dressing(spec, name=f"ring{i}", type=box, size=[r * math.pi / 48 + 0.002, 0.012, 0.0005],
                  pos=[r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), z],
                  quat=_quat_z(a + 90.0), rgba=list(PAINT))
    _dressing(spec, name="spot", type=cyl, size=[0.03, 0.0005, 0.0], pos=[0.0, 0.0, z], rgba=list(PAINT))
    for name, x, colour in (("mark_red", -START_GAP / 2, SHELL_RED), ("mark_blue", START_GAP / 2, SHELL_BLUE)):
        _dressing(spec, name=name, type=cyl, size=[0.09, 0.0005, 0.0], pos=[x, 0.0, z], rgba=list(colour))
    # the plinth under the pedestal
    _dressing(spec, name="plinth0", type=cyl, size=[0.55, 0.03, 0.0], pos=[0.0, 0.0, 0.03], rgba=list(STONE))
    _dressing(spec, name="plinth1", type=cyl, size=[0.36, 0.025, 0.0], pos=[0.0, 0.0, 0.085],
              rgba=[c * 0.92 for c in STONE[:3]] + [1.0])
    for i, a in enumerate(TORCH_AZ):
        _torch(spec, i, TORCH_R * math.cos(math.radians(a)), TORCH_R * math.sin(math.radians(a)))
    _pennants(spec)
    _shells(spec)
    scenery = Path(__file__).resolve().parents[1] / "fetch" / "scenery.py"
    loader = importlib.util.spec_from_file_location("_fetch_scenery", scenery)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    module.decorate(spec)


def margin(x: float, y: float) -> float:
    """How far the point (x, y) is inside the arena's top, in metres (negative outside):
    the disc's radius less the distance from its centre, or the plank's nearer edge."""
    if ARENA == "disc":
        return DISC_R - math.hypot(x, y)
    return min(PLANK_LEN / 2 - abs(x), PLANK_WIDTH / 2 - abs(y))


def apply_arena(env_cfg: Any, *, senses: bool = False, dressed: bool = True) -> None:
    """The second crab, the plank, and a camera on the two of them. `senses`
    keeps the crabs' depth sensors, for a pilot that looks for the other crab;
    `dressed` is the look round the disc (`dress`), which collides with nothing."""
    from mjlab.viewer.viewer_config import ViewerConfig

    from scenes.registry import _install_spec_edits
    from tasks.jumper.common.constants import STAND_Z

    entities = env_cfg.scene.entities
    red = entities[RED]
    blue = copy.deepcopy(red)
    blue.spec_fn = _recoloured(blue.spec_fn, SHELL_RED, SHELL_BLUE)
    z = PLANK_TOP + STAND_Z
    red.init_state = dataclasses.replace(red.init_state, pos=(-START_GAP / 2, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0))
    blue.init_state = dataclasses.replace(blue.init_state, pos=(START_GAP / 2, 0.0, z), rot=(0.0, 0.0, 0.0, 1.0))
    entities[BLUE] = blue

    cloned = _second_robots_events(env_cfg)
    if "reset_base" not in env_cfg.events:
        raise ValueError("the arena pins the spawns, and this config has no 'reset_base'")
    _pin(env_cfg.events["reset_base"])
    _pin(env_cfg.events["reset_base2"])

    # The referee's, not the world's: see the module docstring.
    moved = []
    for name, term in env_cfg.terminations.items():
        if "limit_angle" in (term.params or {}):
            term.params = {**term.params, "limit_angle": math.pi}
            moved.append(name)

    tof = _tof_for_blue(env_cfg, senses)
    _install_spec_edits(env_cfg.scene, None, functools.partial(_decorate, dressed=dressed))

    # Two hexapods and a plank: more contacts than one robot on a plane.
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 512)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 4096)

    # A camera on the plank rather than on one crab: the recorder and the live
    # viewer both keep a `WORLD` origin as the config gives it.
    viewer = env_cfg.viewer
    viewer.origin_type = ViewerConfig.OriginType.WORLD
    viewer.lookat = (float(os.environ.get("BATTLE_CAM_X", "0.0")), 0.0, PLANK_TOP + 0.05)   # along the plank
    viewer.distance = float(os.environ.get("BATTLE_CAM_DIST", "2.3"))
    viewer.azimuth = float(os.environ.get("BATTLE_CAM_AZ", "90"))
    viewer.elevation = float(os.environ.get("BATTLE_CAM_EL", "-12"))

    print(f"[arena] plank {PLANK_LEN:g} x {PLANK_WIDTH:g} m, top at {PLANK_TOP:g} m; red at x {-START_GAP / 2:+.2f} "
          f"facing +x, blue at {START_GAP / 2:+.2f} facing -x")
    print(f"[arena] cloned for the blue crab: {', '.join(cloned) or 'nothing'}"
          + ("; its dToF is `tof2`" if tof else "; no depth sensors (no pilot looks)")
          + (f"; fall resets left to the referee: {', '.join(moved)}" if moved else ""))
