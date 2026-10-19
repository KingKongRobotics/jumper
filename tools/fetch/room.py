"""The obstacle room for beat two: pillars the crab has to walk round, and the tray
it sets the mug on. Props, not dressing: everything here collides (the default
contype/conaffinity), so the robot bumps into a pillar it does not avoid and the mug
stands on the tray, and all of it is in the depth sensor's picture.

Laid out in the world frame the crab starts in (origin, facing +x): the mug stands at
its usual spot ahead, the pillars fill the next two metres with no straight line
through, and the tray waits beyond them on the centreline. The pillars are 0.4 m tall
and 0.15 m square -- taller than anything `sense.py` takes for a mug; the tray is
0.30 m square and 0.07 m tall, which keeps its top 4 cm above the sensor's floor
clearance (FLOOR_CLEAR 0.03) so it reads as a low, wide object, and low enough for
the mug hanging from the raised claw to clear it (see auto.py's `place`).
"""
from __future__ import annotations

import os
from typing import Any

STONE = [0.62, 0.56, 0.48, 1.0]
TRAY = [0.16, 0.14, 0.13, 1.0]
TRAY_RIM = [0.30, 0.26, 0.22, 1.0]

PILLAR_HALF = (0.075, 0.075, 0.30)   # 0.6 m: taller than the flag pole, which is what tells them apart
#: (x, y) of each pillar's centre
#: A slalom: three pairs, each pair framing a gap 0.2 m off the centreline, alternating
#: sides, so the straight line clips a pillar's face by a few centimetres at every pair
#: and the walk has to bend round it. (The first layout, six pillars with no straight
#: line through, was a maze for a 55 deg sensor with no map: beat2a-f.)
#: Two pairs, not three: the mug ratchets round the pinch with every bounce of the walk
#: and a 40 s carry through three pairs arrived hanging 150 deg under its handle (beat2r);
#: the hold keeps it near level for ~15-20 s.
PILLARS = [(1.4, -0.35), (1.4, 0.75), (2.2, 0.35), (2.2, -0.75)]
if os.environ.get("ROOM_PILLARS") == "aside":
    # The short straight delivery: the pillars stand beside the path as set dressing the
    # sensor still sees, and nothing has to be walked round.
    PILLARS = [(1.2, -0.9), (1.2, 0.9), (2.0, -0.9), (2.0, 0.9)]
ROOM_X0 = float(os.environ.get("ROOM_X0", "0.0"))          # the whole room moved along x (the film puts it past the mug)
if os.environ.get("ROOM_PILLARS") == "film":
    # The single shot: two pairs framing a corridor the carry walks down, close enough for
    # the sensor to show them and the walk to keep off them, with no bend to ratchet the mug.
    # (centred on y 0.3, where the mug stands and the carry starts, 0.7 m each side: at
    # +-0.55 about y 0 the two pillars filled the view at 0.7 m and the walk froze, film3)
    # One pillar straight in the path (the carry's line is y 0.3), two off to the sides
    # as dressing: the avoidance is one deliberate sidestep round the one that matters
    # ("if we want to show obstacle avoidance we need an obstacle straight in the path").
    PILLARS = [(0.9, 0.3), (0.7, -0.75), (1.5, 1.25)]
PILLARS = [(x + ROOM_X0, y) for x, y in PILLARS]
TRAY_XY = (float(os.environ.get("ROOM_TRAY_X", "3.4")) + ROOM_X0, float(os.environ.get("ROOM_TRAY_Y", "0.0")))   # 3.0 put the
#: final turn-in-place with a leg against the second pair's pillar, and it did not turn (beat2s)
TRAY_HALF = (0.30, 0.30, 0.004)  # 60 cm square (40 until 2026-10-14: Andrew -- "make the tray a little larger, people
                                 # just need to see it can deliver the object to the target") and 8 mm tall, a mat
                                 # with a drawn rim: where the mug lands depends on
                                 # how far it hangs off level (+-8 cm, film15), so the tray absorbs +-17 cm; and it
                                 # has no edge to lift the mug over -- walking with the nose raised for a 3 cm edge
                                 # flipped a mug hanging 20 deg to 90 within two seconds, twice (w21/w22)  # 3 cm tall: the mug is set down from the LEVEL carry (the nose going up to clear
                                 # a 7 cm tray flung the mug off the pinch, beat2w/x), and level it hangs ~5 cm up
TRAY_TOP = 2 * TRAY_HALF[2]
#: The flag: a thin pole just behind the tray's far edge with a pennant on top. The
#: tray itself (7 cm) is a two-row sliver to a sensor whose lower edge, with the nose up
#: for the carry, meets the ground a metre out (beat2h walked straight past it); the pole
#: shows from across the room, and nothing else in the room is that thin and that tall.
FLAG_BACK = 0.0                  # the pole stands AT the tray's centre (Andrew: "put the flag in the middle of the
                                 # tray"); the mug's spot is MUG_SPOT_SIDE to its left (auto.py's FLAG_SIDE is the
                                 # same number from the spot's side), and a
                                 # pole 0.45 m behind the tray turned every heading error at the tray into a 0.45 m
                                 # lever (30 deg -> the mug 20 cm off, w3); from the centre the lever is 10 cm...
MUG_SPOT_BACK = 0.22             # the mug's spot this far in FRONT of the pole (toward the crab): with the spot level
                                 # with the pole the crab stopped with its face 8 cm from the flag, "pushing all the
                                 # way in" (Andrew, film52/55); now it stops 0.44 m from the pole, facing it
MUG_SPOT_SIDE = 0.12             # the mug's spot this far to the LEFT of the pole, 10 cm inside the 60 cm mat's
                                 # edge; 16 cm from the pole left the mug leaning on it in two placings of five
                                 # (w47/w48, the spread being +-10 cm), 24 kept clear in nine of nine (w54-w68)
FLAG_SIDE = 0.0                  # the pole's own offset from the tray's centre (none: it stands in the middle)
_OLD_FLAG_SIDE_NOTE = -0.24      # ...24 cm to the right of the mug's spot, 4 cm past the mat's edge: at 16 the mug
                                 # came to rest leaning on the pole in two placings of five (w47/w48, its spread
                                 # being +-10 cm); (at 10 cm a mug set down 3 cm off its spot
                                 # came to rest leaning on the pole, w25): the carried mug hangs left of the nose and
                                 # hid a pole dead ahead from the sensor (beat2k); the mug lands 10 cm to its left
FLAG_R, FLAG_H = 0.012, 0.42     # short enough for the pennant to be in the low camera's frame
POLE = [0.92, 0.90, 0.85, 1.0]
PENNANT = [0.95, 0.75, 0.10, 1.0]


def decorate(spec: Any) -> None:
    import mujoco

    for x, y in PILLARS:
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(PILLAR_HALF), pos=[x, y, PILLAR_HALF[2]],
                                rgba=STONE, group=2, mass=0.0)
    tx, ty = TRAY_XY
    # The mat is drawn only (contype/conaffinity 0): collidable, its 8 mm edge caught the
    # right claw's step as the crab closed on it and the body lurched while the mug was
    # being set down, every take (Andrew, film36-45). The mug and the feet rest on the
    # sand through it; 8 mm does not show on camera.
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TRAY_HALF), pos=[tx, ty, TRAY_HALF[2]],
                            rgba=TRAY, group=2, mass=0.0, contype=0, conaffinity=0)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[FLAG_R, FLAG_H / 2, 0.0],
                            pos=[tx + FLAG_BACK, ty + FLAG_SIDE, FLAG_H / 2], rgba=POLE, group=2, mass=0.0)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.003, 0.05],
                            pos=[tx + FLAG_BACK + 0.09, ty + FLAG_SIDE, FLAG_H - 0.05], rgba=PENNANT, group=2, mass=0.0,
                            contype=0, conaffinity=0)
    # a rim round the top, drawn only, so the tray reads as a tray on camera
    for dx, dy, sx, sy in ((TRAY_HALF[0], 0.0, 0.008, TRAY_HALF[1]), (-TRAY_HALF[0], 0.0, 0.008, TRAY_HALF[1]),
                           (0.0, TRAY_HALF[1], TRAY_HALF[0], 0.008), (0.0, -TRAY_HALF[1], TRAY_HALF[0], 0.008)):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.008], pos=[tx + dx, ty + dy, TRAY_TOP + 0.006],
                                rgba=TRAY_RIM, group=2, mass=0.0, contype=0, conaffinity=0)


def apply_room(env_cfg: Any) -> None:
    from scenes.registry import _install_spec_edits

    # More contacts than the mug alone needs: pillars, tray and pole each add theirs, and a
    # mug standing on the tray tipped over 3 s into the sleep gesture with the crab a
    # metre away (film18) -- the body settling onto the sand fills the buffer, and a
    # contact dropped silently is a mug that slides.
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 1024)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 8192)
    _install_spec_edits(env_cfg.scene, None, decorate)
